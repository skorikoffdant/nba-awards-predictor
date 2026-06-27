from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pathlib import Path
import json
from itertools import product

import joblib
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_NBA_DATASET_PATH = PROCESSED_DIR / "all_nba_dataset.csv"

MODEL_PATH = MODELS_DIR / "all_nba_binary_multiclass_assignment.joblib"
REPORT_PATH = MODELS_DIR / "validation_report_binary_multiclass_assignment.json"

MIN_TRAIN_SEASON = 2000
MAX_TRAIN_SEASON = 2025
TARGET_SEASON = 2026
BACKTEST_START_SEASON = 2010
RANDOM_STATE = 42


CORE_RAW_FEATURES = [
    "AGE",
    "GP",
    "W",
    "L",
    "W_PCT",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PTS",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
    "DD2",
    "TD3",
]


RANK_SOURCE_FEATURES = [
    "GP",
    "W",
    "W_PCT",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
    "DD2",
    "TD3",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "REB_PCT",
    "EFG_PCT",
    "TS_PCT",
    "USG_PCT",
    "PIE",
]


ALL_NBA_PREVIOUS_FEATURES = [
    "PLAYED_PREVIOUS_SEASON",
    "PREV_WAS_ALL_NBA",
    "PREV_ALL_NBA_VOTE_SCORE",
]


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
            raise RuntimeError(f"Missing column: {col}. Run build_dataset.py first.")

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rank_cols = {}

    for col in existing_columns(df, RANK_SOURCE_FEATURES):
        rank_col = f"{col}_SEASON_RANK_PCT"

        rank_cols[rank_col] = (
            df.groupby("SEASON_END_YEAR")[col]
            .rank(method="average", ascending=False, pct=True)
        )

    if rank_cols:
        df = pd.concat([df, pd.DataFrame(rank_cols, index=df.index)], axis=1)

    return df


def add_previous_award_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    player_key_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PLAYER_NAME_KEY"

    df = df.sort_values([player_key_col, "SEASON_END_YEAR"]).reset_index(drop=True)

    prev_season = df.groupby(player_key_col)["SEASON_END_YEAR"].shift(1)
    valid_prev = prev_season == (df["SEASON_END_YEAR"] - 1)

    df["PLAYED_PREVIOUS_SEASON"] = valid_prev.astype(int)

    prev_all_nba = df.groupby(player_key_col)["ALL_NBA_LABEL"].shift(1)
    prev_all_nba = prev_all_nba.where(valid_prev, 0).fillna(0)

    df["PREV_WAS_ALL_NBA"] = (prev_all_nba > 0).astype(float)

    vote_map = {
        0.0: 0.0,
        1.0: 1.0,
        2.0: 3.0,
        3.0: 5.0,
    }

    df["PREV_ALL_NBA_VOTE_SCORE"] = (
        prev_all_nba.map(vote_map).fillna(0.0).astype(float)
    )

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rank_features(df)
    df = add_previous_award_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_compact_feature_columns(df: pd.DataFrame) -> list[str]:
    core_features = existing_columns(df, CORE_RAW_FEATURES)

    rank_features = [
        f"{col}_SEASON_RANK_PCT"
        for col in RANK_SOURCE_FEATURES
        if f"{col}_SEASON_RANK_PCT" in df.columns
    ]

    previous_features = existing_columns(df, ALL_NBA_PREVIOUS_FEATURES)

    features = core_features + rank_features + previous_features

    seen = set()
    unique_features = []

    for col in features:
        if col not in seen:
            unique_features.append(col)
            seen.add(col)

    return unique_features


def make_classifier(model_name: str):
    if model_name == "hist_gbc":
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )

    elif model_name == "grad_boost":
        model = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unknown classifier: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def make_binary_sample_weight(y: pd.Series, mode: str) -> np.ndarray:
    y_values = np.asarray(y, dtype=int)

    if mode == "none":
        return np.ones(len(y_values), dtype=float)

    if mode == "sqrt_balance":
        positives = y_values == 1
        pos_count = positives.sum()
        neg_count = len(y_values) - pos_count

        weights = np.ones(len(y_values), dtype=float)

        if pos_count > 0:
            weights[positives] = np.sqrt(neg_count / pos_count)

        return weights

    raise ValueError(f"Unknown binary weight mode: {mode}")


def make_multiclass_sample_weight(y: pd.Series, mode: str) -> np.ndarray:
    y_values = np.asarray(y, dtype=int)

    if mode == "none":
        return np.ones(len(y_values), dtype=float)

    if mode == "sqrt_class_balance":
        counts = pd.Series(y_values).value_counts().to_dict()
        max_count = max(counts.values())

        weights = np.ones(len(y_values), dtype=float)

        for cls, count in counts.items():
            cls_weight = np.sqrt(max_count / count)
            weights[y_values == cls] = cls_weight

        return weights

    if mode == "positive_boost":
        weights = np.ones(len(y_values), dtype=float)
        weights[y_values == 1] = 5.0
        weights[y_values == 2] = 5.0
        weights[y_values == 3] = 5.0
        return weights

    raise ValueError(f"Unknown multiclass weight mode: {mode}")


def fit_classifier(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target: pd.Series,
    model_name: str,
    sample_weight: np.ndarray | None,
):
    model = make_classifier(model_name)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight

    model.fit(
        train_df[feature_columns],
        target,
        **fit_kwargs,
    )

    return model


def fit_stage_models(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
):
    binary_target = (train_df["ALL_NBA_LABEL"] > 0).astype(int)

    binary_weight = make_binary_sample_weight(
        binary_target,
        mode=config["binary_weight_mode"],
    )

    binary_model = fit_classifier(
        train_df=train_df,
        feature_columns=feature_columns,
        target=binary_target,
        model_name=config["binary_model"],
        sample_weight=binary_weight,
    )

    multiclass_target = train_df["ALL_NBA_LABEL"].astype(int)

    multiclass_weight = make_multiclass_sample_weight(
        multiclass_target,
        mode=config["multiclass_weight_mode"],
    )

    multiclass_model = fit_classifier(
        train_df=train_df,
        feature_columns=feature_columns,
        target=multiclass_target,
        model_name=config["multiclass_model"],
        sample_weight=multiclass_weight,
    )

    return binary_model, multiclass_model


def predict_class_probability(
    model,
    X: pd.DataFrame,
    wanted_classes: list[int],
) -> np.ndarray:
    proba = model.predict_proba(X)

    estimator = model.named_steps["model"]
    classes = estimator.classes_

    out = np.zeros((len(X), len(wanted_classes)), dtype=float)

    class_to_idx = {
        int(cls): idx
        for idx, cls in enumerate(classes)
    }

    for wanted_idx, wanted_class in enumerate(wanted_classes):
        if wanted_class in class_to_idx:
            out[:, wanted_idx] = proba[:, class_to_idx[wanted_class]]

    return out


def apply_all_nba_pool_filter(
    df: pd.DataFrame,
    pool_filter: str,
) -> pd.DataFrame:
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

    raise ValueError(f"Unknown pool filter: {pool_filter}")


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


def make_candidate_pool(
    test_df: pd.DataFrame,
    binary_model,
    feature_columns: list[str],
    config: dict,
) -> pd.DataFrame:
    test_df = test_df.copy()

    test_df["P_ALL_NBA_BINARY"] = predict_class_probability(
        binary_model,
        test_df[feature_columns],
        wanted_classes=[1],
    )[:, 0]

    filtered_df = apply_all_nba_pool_filter(
        test_df,
        pool_filter=config["pool_filter"],
    )

    candidate_df = select_top_unique_players(
        filtered_df,
        score_col="P_ALL_NBA_BINARY",
        top_n=config["candidate_top_k"],
    )

    if len(candidate_df) < 15:
        candidate_df = select_top_unique_players(
            test_df,
            score_col="P_ALL_NBA_BINARY",
            top_n=max(15, config["candidate_top_k"]),
        )

    return candidate_df


def add_multiclass_scores(
    candidate_df: pd.DataFrame,
    multiclass_model,
    feature_columns: list[str],
    config: dict,
) -> pd.DataFrame:
    candidate_df = candidate_df.copy()

    proba = predict_class_probability(
        multiclass_model,
        candidate_df[feature_columns],
        wanted_classes=[0, 1, 2, 3],
    )

    candidate_df["P_NONE_MULTI"] = proba[:, 0]
    candidate_df["P_THIRD_MULTI"] = proba[:, 1]
    candidate_df["P_SECOND_MULTI"] = proba[:, 2]
    candidate_df["P_FIRST_MULTI"] = proba[:, 3]

    p1 = candidate_df["P_THIRD_MULTI"].to_numpy()
    p2 = candidate_df["P_SECOND_MULTI"].to_numpy()
    p3 = candidate_df["P_FIRST_MULTI"].to_numpy()

    candidate_df["P_ALL_NBA_MULTI"] = p1 + p2 + p3
    candidate_df["EXPECTED_LABEL"] = 1.0 * p1 + 2.0 * p2 + 3.0 * p3

    candidate_df["FIRST_VALUE"] = 10.0 * p3 + 8.0 * p2 + 6.0 * p1
    candidate_df["SECOND_VALUE"] = 8.0 * p3 + 10.0 * p2 + 8.0 * p1
    candidate_df["THIRD_VALUE"] = 6.0 * p3 + 8.0 * p2 + 10.0 * p1

    candidate_df["BEST_TEAM_VALUE"] = candidate_df[
        ["FIRST_VALUE", "SECOND_VALUE", "THIRD_VALUE"]
    ].max(axis=1)

    if config["selection_score_mode"] == "binary_only":
        candidate_df["FINAL_SELECT_SCORE"] = candidate_df["P_ALL_NBA_BINARY"]

    elif config["selection_score_mode"] == "multi_positive":
        candidate_df["FINAL_SELECT_SCORE"] = candidate_df["P_ALL_NBA_MULTI"]

    elif config["selection_score_mode"] == "expected_best":
        candidate_df["FINAL_SELECT_SCORE"] = candidate_df["BEST_TEAM_VALUE"]

    elif config["selection_score_mode"] == "binary_multi_blend":
        binary_rank = candidate_df["P_ALL_NBA_BINARY"].rank(
            method="average",
            ascending=True,
            pct=True,
        )

        multi_rank = candidate_df["P_ALL_NBA_MULTI"].rank(
            method="average",
            ascending=True,
            pct=True,
        )

        alpha = config["binary_blend_weight"]

        candidate_df["FINAL_SELECT_SCORE"] = (
            alpha * binary_rank
            + (1.0 - alpha) * multi_rank
        )

    else:
        raise ValueError(f"Unknown selection score mode: {config['selection_score_mode']}")

    return candidate_df


def assign_selected_15_to_teams(selected_df: pd.DataFrame) -> pd.DataFrame:
    selected_df = selected_df.copy().reset_index(drop=True)

    slot_labels = [3] * 5 + [2] * 5 + [1] * 5

    first_profit = selected_df["FIRST_VALUE"].to_numpy()
    second_profit = selected_df["SECOND_VALUE"].to_numpy()
    third_profit = selected_df["THIRD_VALUE"].to_numpy()

    profit_matrix = []

    for slot_label in slot_labels:
        if slot_label == 3:
            profit_matrix.append(first_profit)
        elif slot_label == 2:
            profit_matrix.append(second_profit)
        elif slot_label == 1:
            profit_matrix.append(third_profit)
        else:
            raise ValueError(slot_label)

    profit_matrix = np.vstack(profit_matrix)

    row_ind, col_ind = linear_sum_assignment(-profit_matrix)

    assigned_rows = []

    for row_idx, col_idx in sorted(zip(row_ind, col_ind), key=lambda pair: pair[0]):
        row = selected_df.iloc[col_idx].copy()
        row["PREDICTED_LABEL"] = slot_labels[row_idx]
        row["ASSIGNMENT_SLOT"] = row_idx
        row["ASSIGNMENT_PROFIT"] = profit_matrix[row_idx, col_idx]
        assigned_rows.append(row)

    assigned_df = pd.DataFrame(assigned_rows).reset_index(drop=True)

    return assigned_df


def sort_selected_15_to_teams(selected_df: pd.DataFrame) -> pd.DataFrame:
    selected_df = selected_df.sort_values(
        ["EXPECTED_LABEL", "P_ALL_NBA_BINARY"],
        ascending=[False, False],
    ).reset_index(drop=True)

    selected_df["PREDICTED_LABEL"] = 0
    selected_df.loc[0:4, "PREDICTED_LABEL"] = 3
    selected_df.loc[5:9, "PREDICTED_LABEL"] = 2
    selected_df.loc[10:14, "PREDICTED_LABEL"] = 1

    return selected_df


def make_final_prediction(
    candidate_df: pd.DataFrame,
    multiclass_model,
    feature_columns: list[str],
    config: dict,
) -> pd.DataFrame:
    candidate_df = add_multiclass_scores(
        candidate_df=candidate_df,
        multiclass_model=multiclass_model,
        feature_columns=feature_columns,
        config=config,
    )

    selected_df = select_top_unique_players(
        candidate_df,
        score_col="FINAL_SELECT_SCORE",
        top_n=15,
    )

    if config["team_assignment_mode"] == "assignment":
        prediction_df = assign_selected_15_to_teams(selected_df)

    elif config["team_assignment_mode"] == "sort_expected_label":
        prediction_df = sort_selected_15_to_teams(selected_df)

    else:
        raise ValueError(f"Unknown team assignment mode: {config['team_assignment_mode']}")

    return prediction_df


def team_score_details(
    predicted_player_keys: list[str],
    predicted_label: int,
    true_labels: dict[str, int],
) -> dict:
    points = 0
    exact_count = 0
    player_details = []

    for player_key in predicted_player_keys:
        true_label = int(true_labels.get(player_key, 0))

        if true_label == 0:
            player_points = 0
            diff = None
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

        player_details.append(
            {
                "player_key": player_key,
                "predicted_label": predicted_label,
                "true_label": true_label,
                "points": player_points,
                "diff": diff,
            }
        )

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
        "players": player_details,
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

    first_details = team_score_details(first_keys, 3, true_labels)
    second_details = team_score_details(second_keys, 2, true_labels)
    third_details = team_score_details(third_keys, 1, true_labels)

    total_score = (
        first_details["total"]
        + second_details["total"]
        + third_details["total"]
    )

    predicted_keys = set(prediction_df["PLAYER_NAME_KEY"].tolist())
    true_keys = set(true_labels.keys())

    return {
        "score": total_score,
        "top_hits": len(predicted_keys & true_keys),
        "first": first_details,
        "second": second_details,
        "third": third_details,
    }


def evaluate_config(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
) -> dict:
    season_results = []

    for test_season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        train_df = df[
            (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
            & (df["SEASON_END_YEAR"] < test_season)
        ].copy()

        test_df = df[df["SEASON_END_YEAR"] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        binary_model, multiclass_model = fit_stage_models(
            train_df=train_df,
            feature_columns=feature_columns,
            config=config,
        )

        candidate_df = make_candidate_pool(
            test_df=test_df,
            binary_model=binary_model,
            feature_columns=feature_columns,
            config=config,
        )

        prediction_df = make_final_prediction(
            candidate_df=candidate_df,
            multiclass_model=multiclass_model,
            feature_columns=feature_columns,
            config=config,
        )

        score_info = score_prediction(
            prediction_df=prediction_df,
            true_df=test_df,
        )

        true_keys = set(
            test_df.loc[test_df["ALL_NBA_LABEL"] > 0, "PLAYER_NAME_KEY"].tolist()
        )

        candidate_keys = set(candidate_df["PLAYER_NAME_KEY"].tolist())
        prediction_keys = set(prediction_df["PLAYER_NAME_KEY"].tolist())

        candidate_hits = len(candidate_keys & true_keys)
        final_hits = len(prediction_keys & true_keys)

        season_results.append(
            {
                "season": test_season,
                "score": score_info["score"],
                "top_15_hits": score_info["top_hits"],
                "candidate_pool_hits": candidate_hits,
                "final_hits": final_hits,
                "first_exact": score_info["first"]["exact_count"],
                "second_exact": score_info["second"]["exact_count"],
                "third_exact": score_info["third"]["exact_count"],
            }
        )

    avg_score = float(np.mean([row["score"] for row in season_results]))
    avg_top_hits = float(np.mean([row["top_15_hits"] for row in season_results]))
    avg_candidate_hits = float(
        np.mean([row["candidate_pool_hits"] for row in season_results])
    )
    avg_final_hits = float(np.mean([row["final_hits"] for row in season_results]))

    return {
        "config": config,
        "avg_score": avg_score,
        "avg_top_hits": avg_top_hits,
        "avg_candidate_hits": avg_candidate_hits,
        "avg_final_hits": avg_final_hits,
        "season_results": season_results,
    }


def build_configs() -> list[dict]:
    binary_models = [
        "hist_gbc",
        "grad_boost",
    ]

    multiclass_models = [
        "hist_gbc",
        "grad_boost",
    ]

    selection_settings = [
        {
            "selection_score_mode": "binary_only",
            "binary_blend_weight": 1.0,
        },
        {
            "selection_score_mode": "multi_positive",
            "binary_blend_weight": 0.0,
        },
        {
            "selection_score_mode": "expected_best",
            "binary_blend_weight": 0.0,
        },
        {
            "selection_score_mode": "binary_multi_blend",
            "binary_blend_weight": 0.25,
        },
    ]

    configs = []

    team_assignment_modes = [
        "assignment",
        "sort_expected_label",
    ]

    for binary_model, multiclass_model, selection_setting, team_assignment_mode in product(
        binary_models,
        multiclass_models,
        selection_settings,
        team_assignment_modes,
    ):
        config = {
            "approach": "binary_candidate_pool_plus_multiclass_selector_assignment",
            "feature_set": "compact",
            "binary_model": binary_model,
            "binary_weight_mode": "sqrt_balance",
            "multiclass_model": multiclass_model,
            "multiclass_weight_mode": "sqrt_class_balance",
            "pool_filter": "rotation_players",
            "candidate_top_k": 30,
            "team_assignment_mode": team_assignment_mode,
        }

        config.update(selection_setting)
        configs.append(config)

    return configs


def find_best_config(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    configs = build_configs()

    print()
    print("=" * 80)
    print("ALL-NBA BINARY CANDIDATE + MULTICLASS SELECTOR SEARCH")
    print("=" * 80)
    print(f"Configs to test: {len(configs)}", flush=True)

    results = []

    for idx, config in enumerate(configs, start=1):
        print()
        print(f"START [{idx:03d}/{len(configs):03d}] {config}", flush=True)

        result = evaluate_config(
            df=df,
            feature_columns=feature_columns,
            config=config,
        )

        results.append(result)

        print(
            f"DONE  [{idx:03d}/{len(configs):03d}] "
            f"score={result['avg_score']:.2f}/270 | "
            f"hits={result['avg_top_hits']:.2f}/15 | "
            f"candidate_hits={result['avg_candidate_hits']:.2f}/15 | "
            f"final_hits={result['avg_final_hits']:.2f}/15 | "
            f"{config}",
            flush=True,
        )

    best = max(results, key=lambda row: row["avg_score"])

    print()
    print("=" * 80)
    print("BEST ALL-NBA BINARY CANDIDATE + MULTICLASS SELECTOR CONFIG")
    print("=" * 80)
    print(f"score={best['avg_score']:.2f}/270")
    print(f"hits={best['avg_top_hits']:.2f}/15")
    print(f"candidate_hits={best['avg_candidate_hits']:.2f}/15")
    print(f"final_hits={best['avg_final_hits']:.2f}/15")
    print(best["config"])

    return best


def train_final_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    best_config: dict,
) -> dict:
    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] <= MAX_TRAIN_SEASON)
    ].copy()

    binary_model, multiclass_model = fit_stage_models(
        train_df=train_df,
        feature_columns=feature_columns,
        config=best_config,
    )

    artifact = {
        "award_type": "all_nba",
        "model_type": "binary_candidate_pool_plus_multiclass_selector_assignment",
        "dataset_path": str(ALL_NBA_DATASET_PATH),
        "binary_model": binary_model,
        "multiclass_model": multiclass_model,
        "feature_columns": feature_columns,
        "config": best_config,
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
    }

    joblib.dump(artifact, MODEL_PATH)

    return artifact


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


def save_report(
    best: dict,
    feature_columns: list[str],
) -> None:
    report = {
        "model_type": "binary_candidate_pool_plus_multiclass_selector_assignment",
        "dataset_path": str(ALL_NBA_DATASET_PATH),
        "model_path": str(MODEL_PATH),
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
        "backtest_start_season": BACKTEST_START_SEASON,
        "feature_set": "compact",
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "best": best,
        "note": (
            "Binary classifier creates a wide top-30 candidate pool. "
            "Multiclass classifier is trained on all players with labels 0/1/2/3. "
            "Top-15 is selected first, then assignment only distributes those 15 players "
            "between first, second and third teams."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_dataset_info(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    print()
    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"Dataset: {ALL_NBA_DATASET_PATH}")
    print(f"Shape: {df.shape}")
    print(
        "Seasons:",
        df["SEASON_END_YEAR"].min(),
        "-",
        df["SEASON_END_YEAR"].max(),
    )
    print(f"Features: {len(feature_columns)}")

    print()
    print("ALL_NBA_LABEL counts:")
    print(df["ALL_NBA_LABEL"].value_counts().sort_index())

    print()
    print("Rows for target season:")
    print(df[df["SEASON_END_YEAR"] == TARGET_SEASON].shape)


def main() -> None:
    df = load_dataset()
    df = prepare_dataset(df)

    feature_columns = get_compact_feature_columns(df)

    print_dataset_info(
        df=df,
        feature_columns=feature_columns,
    )

    best = find_best_config(
        df=df,
        feature_columns=feature_columns,
    )

    train_final_model(
        df=df,
        feature_columns=feature_columns,
        best_config=best["config"],
    )

    save_report(
        best=best,
        feature_columns=feature_columns,
    )

    print()
    print("=" * 80)
    print("MODEL SAVED")
    print("=" * 80)
    print(f"Model:  {MODEL_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
