from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from features import (
    get_feature_columns,
    add_season_rank_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_NBA_DATASET_PATH = PROCESSED_DIR / "all_nba_dataset.csv"

REPORT_PATH = MODELS_DIR / "feature_set_comparison_report.json"

MIN_TRAIN_SEASON = 2000
MAX_TRAIN_SEASON = 2025
BACKTEST_START_SEASON = 2010
TARGET_SEASON = 2026

RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    if not ALL_NBA_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {ALL_NBA_DATASET_PATH}")

    df = pd.read_csv(ALL_NBA_DATASET_PATH)

    required_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
        "ALL_NBA_LABEL",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def encode_target(df: pd.DataFrame, target_scheme: str) -> pd.Series:
    label = df["ALL_NBA_LABEL"].astype(int)

    if target_scheme == "team_321":
        return label.astype(float)

    if target_scheme == "vote_531":
        return label.map(
            {
                0: 0.0,
                1: 1.0,
                2: 3.0,
                3: 5.0,
            }
        ).astype(float)

    raise ValueError(f"Unknown target_scheme: {target_scheme}")


def make_sample_weight(y: pd.Series, mode: str) -> np.ndarray:
    y_values = np.asarray(y, dtype=float)

    if mode == "none":
        return np.ones(len(y_values), dtype=float)

    if mode == "sqrt_balance":
        positive_mask = y_values > 0
        positive_count = positive_mask.sum()
        negative_count = len(y_values) - positive_count

        weights = np.ones(len(y_values), dtype=float)

        if positive_count > 0:
            weights[positive_mask] = np.sqrt(negative_count / positive_count)

        return weights

    if mode == "label_weighted":
        weights = np.ones(len(y_values), dtype=float)
        weights[y_values == 1] = 3.0
        weights[y_values == 2] = 4.0
        weights[y_values == 3] = 5.0
        return weights

    raise ValueError(f"Unknown weight_mode: {mode}")


def make_model(model_name: str):
    if model_name == "grad_boost":
        model = GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        )

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def apply_pool_filter(df: pd.DataFrame, pool_filter: str) -> pd.DataFrame:
    if pool_filter == "none":
        return df.copy()

    if pool_filter == "rotation_players":
        filtered = df[
            (df["GP"] >= 40)
            & (df["MIN"] >= 18)
        ].copy()

        if len(filtered.drop_duplicates(subset=["PLAYER_NAME_KEY"])) >= 15:
            return filtered

        return df.copy()

    if pool_filter == "strong_rotation_players":
        filtered = df[
            (df["GP"] >= 50)
            & (df["MIN"] >= 24)
        ].copy()

        if len(filtered.drop_duplicates(subset=["PLAYER_NAME_KEY"])) >= 15:
            return filtered

        return df.copy()

    raise ValueError(f"Unknown pool_filter: {pool_filter}")


def select_top_unique_players(
    df: pd.DataFrame,
    score_col: str,
    top_n: int,
) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).copy()

    ranked = ranked.drop_duplicates(
        subset=["PLAYER_NAME_KEY"],
        keep="first",
    )

    return ranked.head(top_n).copy()


def predict_top_15(
    model,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    pool_filter: str,
) -> pd.DataFrame:
    test_df = test_df.copy()

    filtered_df = apply_pool_filter(test_df, pool_filter=pool_filter)

    filtered_df["MODEL_SCORE"] = model.predict(filtered_df[feature_columns])

    prediction_df = select_top_unique_players(
        filtered_df,
        score_col="MODEL_SCORE",
        top_n=15,
    )

    prediction_df = prediction_df.sort_values(
        "MODEL_SCORE",
        ascending=False,
    ).reset_index(drop=True)

    prediction_df["PREDICTED_LABEL"] = 0
    prediction_df.loc[0:4, "PREDICTED_LABEL"] = 3
    prediction_df.loc[5:9, "PREDICTED_LABEL"] = 2
    prediction_df.loc[10:14, "PREDICTED_LABEL"] = 1

    return prediction_df


def team_score_details(
    predicted_player_keys: list[str],
    predicted_label: int,
    true_labels: dict[str, int],
) -> dict:
    points = 0
    exact_count = 0

    for player_key in predicted_player_keys:
        true_label = int(true_labels.get(player_key, 0))

        if true_label == 0:
            player_points = 0
        else:
            diff = abs(predicted_label - true_label)

            if diff == 0:
                player_points = 10
                exact_count += 1
            elif diff == 1:
                player_points = 8
            elif diff == 2:
                player_points = 6
            else:
                player_points = 0

        points += player_points

    bonus_by_exact_count = {
        0: 0,
        1: 0,
        2: 5,
        3: 10,
        4: 20,
        5: 40,
    }

    bonus = bonus_by_exact_count.get(exact_count, 40)

    return {
        "points_without_bonus": points,
        "exact_count": exact_count,
        "bonus": bonus,
        "total": points + bonus,
    }


def score_prediction(
    prediction_df: pd.DataFrame,
    true_df: pd.DataFrame,
) -> dict:
    true_labels = dict(
        zip(
            true_df.loc[true_df["ALL_NBA_LABEL"] > 0, "PLAYER_NAME_KEY"],
            true_df.loc[true_df["ALL_NBA_LABEL"] > 0, "ALL_NBA_LABEL"],
        )
    )

    first_keys = prediction_df.loc[
        prediction_df["PREDICTED_LABEL"] == 3,
        "PLAYER_NAME_KEY",
    ].tolist()

    second_keys = prediction_df.loc[
        prediction_df["PREDICTED_LABEL"] == 2,
        "PLAYER_NAME_KEY",
    ].tolist()

    third_keys = prediction_df.loc[
        prediction_df["PREDICTED_LABEL"] == 1,
        "PLAYER_NAME_KEY",
    ].tolist()

    first = team_score_details(first_keys, 3, true_labels)
    second = team_score_details(second_keys, 2, true_labels)
    third = team_score_details(third_keys, 1, true_labels)

    total_score = first["total"] + second["total"] + third["total"]

    predicted_keys = set(prediction_df["PLAYER_NAME_KEY"].tolist())
    true_keys = set(true_labels.keys())

    return {
        "score": total_score,
        "top_hits": len(predicted_keys & true_keys),
        "first_exact": first["exact_count"],
        "second_exact": second["exact_count"],
        "third_exact": third["exact_count"],
        "first_score": first["total"],
        "second_score": second["total"],
        "third_score": third["total"],
    }


def evaluate_config(
    df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    target_scheme: str,
    weight_mode: str,
    pool_filter: str,
) -> dict:
    feature_columns = get_feature_columns(
        df,
        feature_set=feature_set,
        verbose=True,
    )

    season_results = []

    for test_season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        train_df = df[
            (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
            & (df["SEASON_END_YEAR"] < test_season)
        ].copy()

        test_df = df[df["SEASON_END_YEAR"] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        y_train = encode_target(train_df, target_scheme=target_scheme)
        sample_weight = make_sample_weight(y_train, mode=weight_mode)

        model = make_model(model_name)

        model.fit(
            train_df[feature_columns],
            y_train,
            model__sample_weight=sample_weight,
        )

        prediction_df = predict_top_15(
            model=model,
            test_df=test_df,
            feature_columns=feature_columns,
            pool_filter=pool_filter,
        )

        score_info = score_prediction(
            prediction_df=prediction_df,
            true_df=test_df,
        )

        season_results.append(
            {
                "season": test_season,
                **score_info,
            }
        )

        print(
            f"  {test_season}: "
            f"score={score_info['score']:>3} | "
            f"hits={score_info['top_hits']:>2}/15 | "
            f"exact=({score_info['first_exact']}, "
            f"{score_info['second_exact']}, "
            f"{score_info['third_exact']})",
            flush=True,
        )

    avg_score = float(np.mean([row["score"] for row in season_results]))
    avg_hits = float(np.mean([row["top_hits"] for row in season_results]))
    avg_first_exact = float(np.mean([row["first_exact"] for row in season_results]))
    avg_second_exact = float(np.mean([row["second_exact"] for row in season_results]))
    avg_third_exact = float(np.mean([row["third_exact"] for row in season_results]))

    return {
        "feature_set": feature_set,
        "model_name": model_name,
        "target_scheme": target_scheme,
        "weight_mode": weight_mode,
        "pool_filter": pool_filter,
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "avg_score": avg_score,
        "avg_hits": avg_hits,
        "avg_first_exact": avg_first_exact,
        "avg_second_exact": avg_second_exact,
        "avg_third_exact": avg_third_exact,
        "season_results": season_results,
    }


def train_final_model(
    df: pd.DataFrame,
    best_result: dict,
) -> Path:
    feature_columns = best_result["feature_columns"]

    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] <= MAX_TRAIN_SEASON)
    ].copy()

    y_train = encode_target(
        train_df,
        target_scheme=best_result["target_scheme"],
    )

    sample_weight = make_sample_weight(
        y_train,
        mode=best_result["weight_mode"],
    )

    model = make_model(best_result["model_name"])

    model.fit(
        train_df[feature_columns],
        y_train,
        model__sample_weight=sample_weight,
    )

    model_path = MODELS_DIR / f"all_nba_{best_result['feature_set']}_{best_result['model_name']}.joblib"

    artifact = {
        "award_type": "all_nba",
        "model_type": "feature_set_comparison_best",
        "model": model,
        "feature_set": best_result["feature_set"],
        "feature_columns": feature_columns,
        "config": {
            "model_name": best_result["model_name"],
            "target_scheme": best_result["target_scheme"],
            "weight_mode": best_result["weight_mode"],
            "pool_filter": best_result["pool_filter"],
        },
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
    }

    joblib.dump(artifact, model_path)

    return model_path


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=[
            "compact",
            "previous_team_share",
            "previous_team_share_bref",
            "previous_team_share_bref_allstar",
        ],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "grad_boost",
        ],
        choices=[
            "grad_boost",
            "random_forest",
        ],
    )

    parser.add_argument(
        "--target-schemes",
        nargs="+",
        default=[
            "team_321",
        ],
        choices=[
            "team_321",
            "vote_531",
        ],
    )

    parser.add_argument(
        "--weight-modes",
        nargs="+",
        default=[
            "sqrt_balance",
        ],
        choices=[
            "none",
            "sqrt_balance",
            "label_weighted",
        ],
    )

    parser.add_argument(
        "--pool-filters",
        nargs="+",
        default=[
            "rotation_players",
        ],
        choices=[
            "none",
            "rotation_players",
            "strong_rotation_players",
        ],
    )

    parser.add_argument(
        "--save-best-model",
        action="store_true",
    )

    args = parser.parse_args()

    df = load_dataset()
    df = add_season_rank_features(df)

    print()
    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"Dataset: {ALL_NBA_DATASET_PATH}")
    print(f"Shape: {df.shape}")
    print(
        "Seasons:",
        int(df["SEASON_END_YEAR"].min()),
        "-",
        int(df["SEASON_END_YEAR"].max()),
    )

    print()
    print("ALL_NBA_LABEL counts:")
    print(df["ALL_NBA_LABEL"].value_counts().sort_index())

    configs = []

    for feature_set in args.feature_sets:
        for model_name in args.models:
            for target_scheme in args.target_schemes:
                for weight_mode in args.weight_modes:
                    for pool_filter in args.pool_filters:
                        configs.append(
                            {
                                "feature_set": feature_set,
                                "model_name": model_name,
                                "target_scheme": target_scheme,
                                "weight_mode": weight_mode,
                                "pool_filter": pool_filter,
                            }
                        )

    print()
    print("=" * 80)
    print("FEATURE SET COMPARISON")
    print("=" * 80)
    print(f"Configs to test: {len(configs)}")

    results = []

    for idx, config in enumerate(configs, start=1):
        print()
        print("=" * 80)
        print(f"START [{idx:03d}/{len(configs):03d}] {config}")
        print("=" * 80)

        result = evaluate_config(
            df=df,
            feature_set=config["feature_set"],
            model_name=config["model_name"],
            target_scheme=config["target_scheme"],
            weight_mode=config["weight_mode"],
            pool_filter=config["pool_filter"],
        )

        results.append(result)

        print()
        print(
            f"DONE [{idx:03d}/{len(configs):03d}] "
            f"feature_set={result['feature_set']} | "
            f"features={result['num_features']} | "
            f"score={result['avg_score']:.2f}/270 | "
            f"hits={result['avg_hits']:.2f}/15 | "
            f"exact=({result['avg_first_exact']:.2f}, "
            f"{result['avg_second_exact']:.2f}, "
            f"{result['avg_third_exact']:.2f})"
        )

    best = max(results, key=lambda row: row["avg_score"])

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    results_sorted = sorted(results, key=lambda row: row["avg_score"], reverse=True)

    for row in results_sorted:
        print(
            f"{row['avg_score']:>7.2f}/270 | "
            f"hits={row['avg_hits']:>5.2f}/15 | "
            f"features={row['num_features']:>3} | "
            f"{row['feature_set']} | "
            f"{row['model_name']} | "
            f"{row['target_scheme']} | "
            f"{row['weight_mode']} | "
            f"{row['pool_filter']}"
        )

    print()
    print("=" * 80)
    print("BEST CONFIG")
    print("=" * 80)
    print(f"score={best['avg_score']:.2f}/270")
    print(f"hits={best['avg_hits']:.2f}/15")
    print(f"features={best['num_features']}")
    print(
        {
            "feature_set": best["feature_set"],
            "model_name": best["model_name"],
            "target_scheme": best["target_scheme"],
            "weight_mode": best["weight_mode"],
            "pool_filter": best["pool_filter"],
        }
    )

    report = {
        "dataset_path": str(ALL_NBA_DATASET_PATH),
        "backtest_start_season": BACKTEST_START_SEASON,
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
        "results": results_sorted,
        "best": best,
    }

    REPORT_PATH.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Report saved: {REPORT_PATH}")

    if args.save_best_model:
        model_path = train_final_model(
            df=df,
            best_result=best,
        )

        print(f"Best model saved: {model_path}")


if __name__ == "__main__":
    main()