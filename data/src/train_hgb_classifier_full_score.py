from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from features import get_feature_columns, add_season_rank_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_NBA_DATASET_PATH = PROCESSED_DIR / "all_nba_dataset.csv"
ALL_ROOKIE_DATASET_PATH = PROCESSED_DIR / "all_rookie_dataset.csv"

REPORT_PATH = MODELS_DIR / "hgb_classifier_full_score_report.json"
UNIFIED_OUTPUT_DIR = MODELS_DIR / "unified_best_experiments"

MIN_TRAIN_SEASON = 2000
MAX_TRAIN_SEASON = 2025
BACKTEST_START_SEASON = 2010
TARGET_SEASON = 2026
RANDOM_STATE = 42

AWARD_CONFIGS = {
    "all_nba": {
        "dataset_path": ALL_NBA_DATASET_PATH,
        "label_col": "ALL_NBA_LABEL",
        "max_score": 270,
        "num_players": 15,
        "slot_labels": [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
        "classes": [0, 1, 2, 3],
    },
    "all_rookie": {
        "dataset_path": ALL_ROOKIE_DATASET_PATH,
        "label_col": "ALL_ROOKIE_LABEL",
        "max_score": 180,
        "num_players": 10,
        "slot_labels": [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
        "classes": [0, 1, 2],
    },
}


def load_dataset(path: Path, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)
    required_columns = ["SEASON_END_YEAR", "PLAYER_NAME", "PLAYER_NAME_KEY", label_col]

    for column in required_columns:
        if column not in df.columns:
            raise RuntimeError(f"Missing column in {path}: {column}")

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def make_hgb_classifier(config: dict) -> Pipeline:
    model = HistGradientBoostingClassifier(
        max_iter=config["max_iter"],
        learning_rate=config["learning_rate"],
        max_leaf_nodes=config["max_leaf_nodes"],
        l2_regularization=config["l2_regularization"],
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def make_sample_weight(y: pd.Series, mode: str) -> np.ndarray:
    y_values = y.astype(int).to_numpy()

    if mode == "none":
        return np.ones(len(y_values), dtype=float)

    if mode == "sqrt_class_balance":
        counts = pd.Series(y_values).value_counts().to_dict()
        max_count = max(counts.values())
        weights = np.ones(len(y_values), dtype=float)

        for label, count in counts.items():
            weights[y_values == label] = np.sqrt(max_count / count)

        return weights

    if mode == "positive_boost":
        weights = np.ones(len(y_values), dtype=float)
        weights[y_values > 0] = 6.0
        return weights

    if mode == "team_weighted":
        weights = np.ones(len(y_values), dtype=float)
        weights[y_values == 1] = 4.0
        weights[y_values == 2] = 5.0
        weights[y_values == 3] = 6.0
        return weights

    raise ValueError(f"Unknown weight mode: {mode}")


def predict_proba_for_classes(
    model: Pipeline,
    x: pd.DataFrame,
    wanted_classes: list[int],
) -> np.ndarray:
    raw_proba = model.predict_proba(x)
    model_classes = model.named_steps["model"].classes_

    proba = np.zeros((len(x), len(wanted_classes)), dtype=float)
    class_to_index = {label: index for index, label in enumerate(wanted_classes)}

    for source_index, label in enumerate(model_classes):
        label = int(label)

        if label in class_to_index:
            proba[:, class_to_index[label]] = raw_proba[:, source_index]

    return proba


def apply_pool_filter(df: pd.DataFrame, pool_filter: str) -> pd.DataFrame:
    if pool_filter == "none":
        return df.copy()

    if pool_filter == "rotation_players":
        filtered_df = df[(df["GP"] >= 40) & (df["MIN"] >= 18)].copy()

        if filtered_df["PLAYER_NAME_KEY"].nunique() >= 10:
            return filtered_df

        return df.copy()

    if pool_filter == "strong_rotation_players":
        filtered_df = df[(df["GP"] >= 50) & (df["MIN"] >= 24)].copy()

        if filtered_df["PLAYER_NAME_KEY"].nunique() >= 10:
            return filtered_df

        return df.copy()

    raise ValueError(f"Unknown pool_filter: {pool_filter}")


def select_top_unique_players(
    df: pd.DataFrame,
    score_col: str,
    top_n: int,
) -> pd.DataFrame:
    ranked_df = df.sort_values(score_col, ascending=False)
    ranked_df = ranked_df.drop_duplicates(subset=["PLAYER_NAME_KEY"], keep="first")

    return ranked_df.head(top_n).copy()


def add_classifier_scores(
    df: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
    award_cfg: dict,
) -> pd.DataFrame:
    df = df.copy()

    classes = award_cfg["classes"]
    positive_classes = [label for label in classes if label > 0]
    proba = predict_proba_for_classes(model, df[feature_columns], classes)

    for index, label in enumerate(classes):
        df[f"P_CLASS_{label}"] = proba[:, index]

    df["P_AWARD"] = 0.0
    df["EXPECTED_LABEL"] = 0.0

    for label in positive_classes:
        df["P_AWARD"] += df[f"P_CLASS_{label}"]
        df["EXPECTED_LABEL"] += label * df[f"P_CLASS_{label}"]

    predicted_labels = sorted(set(award_cfg["slot_labels"]), reverse=True)

    for predicted_label in predicted_labels:
        value = np.zeros(len(df), dtype=float)

        for true_label in positive_classes:
            diff = abs(predicted_label - true_label)

            if diff == 0:
                points = 10.0
            elif diff == 1:
                points = 8.0
            elif diff == 2:
                points = 6.0
            else:
                points = 0.0

            value += points * df[f"P_CLASS_{true_label}"].to_numpy()

        df[f"VALUE_LABEL_{predicted_label}"] = value

    value_columns = [f"VALUE_LABEL_{label}" for label in predicted_labels]
    df["BEST_TEAM_VALUE"] = df[value_columns].max(axis=1)

    return df


def assign_selected_players(
    selected_df: pd.DataFrame,
    award_cfg: dict,
) -> pd.DataFrame:
    selected_df = selected_df.copy().reset_index(drop=True)
    slot_labels = award_cfg["slot_labels"]

    profit_matrix = np.vstack(
        [selected_df[f"VALUE_LABEL_{label}"].to_numpy() for label in slot_labels]
    )

    row_indexes, column_indexes = linear_sum_assignment(-profit_matrix)
    rows = []

    for row_index, column_index in sorted(zip(row_indexes, column_indexes)):
        row = selected_df.iloc[column_index].copy()
        row["PREDICTED_LABEL"] = slot_labels[row_index]
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def assign_by_expected_label(
    selected_df: pd.DataFrame,
    award_cfg: dict,
) -> pd.DataFrame:
    selected_df = selected_df.sort_values("EXPECTED_LABEL", ascending=False)
    selected_df = selected_df.reset_index(drop=True)
    selected_df["PREDICTED_LABEL"] = 0

    start = 0
    labels = sorted(set(award_cfg["slot_labels"]), reverse=True)

    for label in labels:
        end = start + award_cfg["slot_labels"].count(label)
        selected_df.loc[start:end - 1, "PREDICTED_LABEL"] = label
        start = end

    return selected_df


def make_prediction(
    test_df: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
    config: dict,
    award_cfg: dict,
) -> pd.DataFrame:
    filtered_df = apply_pool_filter(test_df, config["pool_filter"])
    scored_df = add_classifier_scores(filtered_df, model, feature_columns, award_cfg)

    score_columns = {
        "p_award": "P_AWARD",
        "expected_label": "EXPECTED_LABEL",
        "best_team_value": "BEST_TEAM_VALUE",
    }

    selection_mode = config["selection_mode"]

    if selection_mode not in score_columns:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    selected_df = select_top_unique_players(
        scored_df,
        score_columns[selection_mode],
        award_cfg["num_players"],
    )

    team_assignment_mode = config["team_assignment_mode"]

    if team_assignment_mode == "sort_expected_label":
        return assign_by_expected_label(selected_df, award_cfg)

    if team_assignment_mode == "assignment":
        return assign_selected_players(selected_df, award_cfg)

    raise ValueError(f"Unknown team_assignment_mode: {team_assignment_mode}")


def score_player(player_key: str, predicted_label: int, true_labels: dict[str, int]) -> tuple[int, int]:
    true_label = int(true_labels.get(player_key, 0))

    if true_label == 0:
        return 0, 0

    diff = abs(predicted_label - true_label)

    if diff == 0:
        return 10, 1

    if diff == 1:
        return 8, 0

    if diff == 2:
        return 6, 0

    return 0, 0


def team_score_details(
    predicted_player_keys: list[str],
    predicted_label: int,
    true_labels: dict[str, int],
) -> dict:
    points = 0
    exact_count = 0

    for player_key in predicted_player_keys:
        player_points, player_exact = score_player(player_key, predicted_label, true_labels)
        points += player_points
        exact_count += player_exact

    bonus_by_exact_count = {
        0: 0,
        1: 0,
        2: 5,
        3: 10,
        4: 20,
        5: 40,
    }

    return {
        "points_without_bonus": points,
        "exact_count": exact_count,
        "bonus": bonus_by_exact_count.get(exact_count, 40),
        "total": points + bonus_by_exact_count.get(exact_count, 40),
    }


def score_prediction(
    prediction_df: pd.DataFrame,
    true_df: pd.DataFrame,
    label_col: str,
    award_cfg: dict,
) -> dict:
    true_df = true_df[true_df[label_col] > 0]
    true_labels = dict(zip(true_df["PLAYER_NAME_KEY"], true_df[label_col]))

    score_by_label = {}
    exact_by_label = {}
    total_score = 0

    labels = sorted(set(award_cfg["slot_labels"]), reverse=True)

    for label in labels:
        predicted_keys = prediction_df.loc[
            prediction_df["PREDICTED_LABEL"] == label,
            "PLAYER_NAME_KEY",
        ].tolist()

        details = team_score_details(predicted_keys, label, true_labels)
        score_by_label[label] = details["total"]
        exact_by_label[label] = details["exact_count"]
        total_score += details["total"]

    predicted_keys = set(prediction_df["PLAYER_NAME_KEY"])
    true_keys = set(true_labels)

    return {
        "score": total_score,
        "top_hits": len(predicted_keys & true_keys),
        "score_by_label": score_by_label,
        "exact_by_label": exact_by_label,
    }


def evaluate_award_config(
    df: pd.DataFrame,
    award_name: str,
    config: dict,
) -> dict:
    award_cfg = AWARD_CONFIGS[award_name]
    label_col = award_cfg["label_col"]
    feature_columns = get_feature_columns(df, config["feature_set"], verbose=True)

    season_results = []

    for test_season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        train_df = df[
            (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
            & (df["SEASON_END_YEAR"] < test_season)
        ].copy()

        test_df = df[df["SEASON_END_YEAR"] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        y_train = train_df[label_col].astype(int)
        sample_weight = make_sample_weight(y_train, config["weight_mode"])
        model = make_hgb_classifier(config)

        model.fit(
            train_df[feature_columns],
            y_train,
            model__sample_weight=sample_weight,
        )

        prediction_df = make_prediction(
            test_df=test_df,
            model=model,
            feature_columns=feature_columns,
            config=config,
            award_cfg=award_cfg,
        )

        score_info = score_prediction(
            prediction_df=prediction_df,
            true_df=test_df,
            label_col=label_col,
            award_cfg=award_cfg,
        )

        season_results.append({"season": test_season, **score_info})

        exact_text = ", ".join(
            f"{label}:{score_info['exact_by_label'].get(label, 0)}"
            for label in sorted(set(award_cfg["slot_labels"]), reverse=True)
        )

        print(
            f"  {award_name} {test_season}: "
            f"score={score_info['score']:>3}/{award_cfg['max_score']} | "
            f"hits={score_info['top_hits']:>2}/{award_cfg['num_players']} | "
            f"exact=({exact_text})",
            flush=True,
        )

    avg_score = float(np.mean([row["score"] for row in season_results]))
    avg_hits = float(np.mean([row["top_hits"] for row in season_results]))

    return {
        "award_name": award_name,
        "label_col": label_col,
        "max_score": award_cfg["max_score"],
        "num_players": award_cfg["num_players"],
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "avg_score": avg_score,
        "avg_score_pct": avg_score / award_cfg["max_score"],
        "avg_hits": avg_hits,
        "season_results": season_results,
    }


def rookie_feature_set_for_full_score(feature_set: str) -> str:
    if feature_set.startswith("previous_team_share"):
        return "previous_team_share"

    if feature_set.startswith("compact_plus_previous_team_share"):
        return "compact_plus_previous_team_share"

    return feature_set


def evaluate_full_config(
    all_nba_df: pd.DataFrame,
    all_rookie_df: pd.DataFrame,
    config: dict,
) -> dict:
    print()
    print("-" * 80)
    print("ALL-NBA EVALUATION")
    print("-" * 80)

    all_nba_result = evaluate_award_config(all_nba_df, "all_nba", config)

    print()
    print("-" * 80)
    print("ALL-ROOKIE EVALUATION")
    print("-" * 80)

    rookie_config = config.copy()
    rookie_config["feature_set"] = rookie_feature_set_for_full_score(config["feature_set"])

    if rookie_config["feature_set"] != config["feature_set"]:
        print(f"All-Rookie feature_set forced to {rookie_config['feature_set']}")

    all_rookie_result = evaluate_award_config(all_rookie_df, "all_rookie", rookie_config)

    nba_by_season = {row["season"]: row for row in all_nba_result["season_results"]}
    rookie_by_season = {row["season"]: row for row in all_rookie_result["season_results"]}

    common_seasons = sorted(set(nba_by_season) & set(rookie_by_season))
    season_total_results = []

    for season in common_seasons:
        nba_score = nba_by_season[season]["score"]
        rookie_score = rookie_by_season[season]["score"]

        season_total_results.append(
            {
                "season": season,
                "all_nba_score": nba_score,
                "all_rookie_score": rookie_score,
                "total_score": nba_score + rookie_score,
            }
        )

    avg_total_score = float(np.mean([row["total_score"] for row in season_total_results]))

    return {
        "config": config,
        "all_nba": all_nba_result,
        "all_rookie": all_rookie_result,
        "avg_total_score": avg_total_score,
        "avg_total_score_pct": avg_total_score / 450.0,
        "season_total_results": season_total_results,
    }


def build_config(args: argparse.Namespace) -> dict:
    return {
        "feature_set": args.feature_set,
        "model_name": "hgb_classifier",
        "weight_mode": args.weight_mode,
        "selection_mode": args.selection_mode,
        "team_assignment_mode": args.team_assignment_mode,
        "pool_filter": args.pool_filter,
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2_regularization": args.l2_regularization,
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def print_dataset_info(name: str, df: pd.DataFrame, label_col: str, path: Path) -> None:
    print()
    print("=" * 80)
    print(f"DATASET INFO: {name}")
    print("=" * 80)
    print(f"Dataset: {path}")
    print(f"Shape: {df.shape}")
    print(
        "Seasons:",
        int(df["SEASON_END_YEAR"].min()),
        "-",
        int(df["SEASON_END_YEAR"].max()),
    )
    print()
    print(f"{label_col} counts:")
    print(df[label_col].value_counts().sort_index())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default="previous_team_share_allstar")
    parser.add_argument(
        "--weight-mode",
        default="positive_boost",
        choices=["none", "sqrt_class_balance", "positive_boost", "team_weighted"],
    )
    parser.add_argument(
        "--selection-mode",
        default="expected_label",
        choices=["p_award", "expected_label", "best_team_value"],
    )
    parser.add_argument(
        "--team-assignment-mode",
        default="sort_expected_label",
        choices=["sort_expected_label", "assignment"],
    )
    parser.add_argument(
        "--pool-filter",
        default="none",
        choices=["none", "rotation_players", "strong_rotation_players"],
    )
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)

    return parser.parse_args()


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_nba_df = load_dataset(
        AWARD_CONFIGS["all_nba"]["dataset_path"],
        AWARD_CONFIGS["all_nba"]["label_col"],
    )
    all_rookie_df = load_dataset(
        AWARD_CONFIGS["all_rookie"]["dataset_path"],
        AWARD_CONFIGS["all_rookie"]["label_col"],
    )

    all_nba_df = add_season_rank_features(all_nba_df)
    all_rookie_df = add_season_rank_features(all_rookie_df)

    return all_nba_df, all_rookie_df


def main() -> None:
    args = parse_args()
    config = build_config(args)

    all_nba_df, all_rookie_df = load_all_data()

    print_dataset_info(
        "All-NBA",
        all_nba_df,
        AWARD_CONFIGS["all_nba"]["label_col"],
        AWARD_CONFIGS["all_nba"]["dataset_path"],
    )
    print_dataset_info(
        "All-Rookie",
        all_rookie_df,
        AWARD_CONFIGS["all_rookie"]["label_col"],
        AWARD_CONFIGS["all_rookie"]["dataset_path"],
    )

    print()
    print("=" * 80)
    print("HGB CLASSIFIER FULL SCORE")
    print("=" * 80)
    print(config)

    result = evaluate_full_config(all_nba_df, all_rookie_df, config)

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print(
        f"TOTAL={result['avg_total_score']:.2f}/450 "
        f"({100.0 * result['avg_total_score_pct']:.2f}%)"
    )
    print(f"All-NBA={result['all_nba']['avg_score']:.2f}/270")
    print(f"All-Rookie={result['all_rookie']['avg_score']:.2f}/180")
    print(f"All-NBA hits={result['all_nba']['avg_hits']:.2f}/15")
    print(f"All-Rookie hits={result['all_rookie']['avg_hits']:.2f}/10")

    report = {
        "all_nba_dataset_path": str(ALL_NBA_DATASET_PATH),
        "all_rookie_dataset_path": str(ALL_ROOKIE_DATASET_PATH),
        "backtest_start_season": BACKTEST_START_SEASON,
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
        "results": [result],
        "best": result,
    }

    REPORT_PATH.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print(f"Report saved: {REPORT_PATH}")

    from unified_best_output import write_from_hgb_full_report

    write_from_hgb_full_report(
        report_path=REPORT_PATH,
        out_dir=UNIFIED_OUTPUT_DIR,
        experiment="hgb_classifier",
    )


if __name__ == "__main__":
    main()