from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.optimize import linear_sum_assignment

from features import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_NBA_DATASET_PATH = PROCESSED_DIR / "all_nba_dataset.csv"
REPORT_PATH = MODELS_DIR / "team_reorderer_report.json"
UNIFIED_OUTPUT_DIR = MODELS_DIR / "unified_best_experiments"
HGB_FULL_REPORT_PATH = MODELS_DIR / "hgb_classifier_full_score_report.json"

MIN_TRAIN_SEASON = 2000
BACKTEST_START_SEASON = 2010
MAX_TRAIN_SEASON = 2025
RANDOM_STATE = 42

LABEL_COL = "ALL_NBA_LABEL"

BASE_SCORE_FEATURES = [
    "P_NONE",
    "P_THIRD",
    "P_SECOND",
    "P_FIRST",
    "P_ALL_NBA",
    "EXPECTED_LABEL",
    "FIRST_VALUE",
    "SECOND_VALUE",
    "THIRD_VALUE",
    "BEST_TEAM_VALUE",
    "BASE_RANK",
    "BASE_RANK_PCT",
    "GAP_PREV_EXPECTED_LABEL",
    "GAP_NEXT_EXPECTED_LABEL",
]

TARGET_MAPS = {
    "label": {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0},
    "spaced": {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0},
    "score_like": {0: 0.0, 1: 6.0, 2: 8.0, 3: 10.0},
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ALL_NBA_DATASET_PATH)

    required = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
        LABEL_COL,
    ]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def make_hgb_classifier(config: dict) -> Pipeline:
    model = HistGradientBoostingClassifier(
        max_iter=config["base_max_iter"],
        learning_rate=config["base_learning_rate"],
        max_leaf_nodes=config["base_max_leaf_nodes"],
        l2_regularization=config["base_l2_regularization"],
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def make_reorderer_classifier(config: dict) -> Pipeline:
    model = HistGradientBoostingClassifier(
        max_iter=config["reorderer_max_iter"],
        learning_rate=config["reorderer_learning_rate"],
        max_leaf_nodes=config["reorderer_max_leaf_nodes"],
        l2_regularization=config["reorderer_l2_regularization"],
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def make_reorderer_regressor(config: dict) -> Pipeline:
    model = HistGradientBoostingRegressor(
        max_iter=config["reorderer_max_iter"],
        learning_rate=config["reorderer_learning_rate"],
        max_leaf_nodes=config["reorderer_max_leaf_nodes"],
        l2_regularization=config["reorderer_l2_regularization"],
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def make_sample_weight(y: pd.Series, mode: str) -> np.ndarray:
    y_values = y.astype(int).to_numpy()

    if mode == "none":
        return np.ones(len(y_values), dtype=float)

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

    if mode == "sqrt_class_balance":
        counts = pd.Series(y_values).value_counts().to_dict()
        max_count = max(counts.values())

        weights = np.ones(len(y_values), dtype=float)
        for cls, count in counts.items():
            weights[y_values == cls] = np.sqrt(max_count / count)

        return weights

    raise ValueError(f"Unknown weight mode: {mode}")


def predict_proba_0123(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    raw_proba = model.predict_proba(X)
    classes = model.named_steps["model"].classes_

    proba = np.zeros((len(X), 4), dtype=float)

    for src_idx, cls in enumerate(classes):
        cls = int(cls)
        if 0 <= cls <= 3:
            proba[:, cls] = raw_proba[:, src_idx]

    return proba


def add_base_scores(
    df: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    df = df.copy()

    proba = predict_proba_0123(model, df[feature_columns])

    df["P_NONE"] = proba[:, 0]
    df["P_THIRD"] = proba[:, 1]
    df["P_SECOND"] = proba[:, 2]
    df["P_FIRST"] = proba[:, 3]

    df["P_ALL_NBA"] = df["P_THIRD"] + df["P_SECOND"] + df["P_FIRST"]

    df["EXPECTED_LABEL"] = (
        1.0 * df["P_THIRD"]
        + 2.0 * df["P_SECOND"]
        + 3.0 * df["P_FIRST"]
    )

    p1 = df["P_THIRD"].to_numpy()
    p2 = df["P_SECOND"].to_numpy()
    p3 = df["P_FIRST"].to_numpy()

    df["FIRST_VALUE"] = 10.0 * p3 + 8.0 * p2 + 6.0 * p1
    df["SECOND_VALUE"] = 8.0 * p3 + 10.0 * p2 + 8.0 * p1
    df["THIRD_VALUE"] = 6.0 * p3 + 8.0 * p2 + 10.0 * p1
    df["BEST_TEAM_VALUE"] = df[["FIRST_VALUE", "SECOND_VALUE", "THIRD_VALUE"]].max(axis=1)

    return df


def select_top_unique_players(df: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).copy()
    ranked = ranked.drop_duplicates(subset=["PLAYER_NAME_KEY"], keep="first")
    return ranked.head(top_n).copy()


def add_base_rank_columns(candidate_df: pd.DataFrame) -> pd.DataFrame:
    candidate_df = candidate_df.copy().reset_index(drop=True)
    candidate_df["BASE_RANK"] = np.arange(1, len(candidate_df) + 1)

    if len(candidate_df) <= 1:
        candidate_df["BASE_RANK_PCT"] = 1.0
    else:
        candidate_df["BASE_RANK_PCT"] = 1.0 - (
            (candidate_df["BASE_RANK"] - 1) / (len(candidate_df) - 1)
        )

    expected = candidate_df["EXPECTED_LABEL"].astype(float)
    candidate_df["GAP_PREV_EXPECTED_LABEL"] = expected.shift(1) - expected
    candidate_df["GAP_NEXT_EXPECTED_LABEL"] = expected - expected.shift(-1)
    candidate_df["GAP_PREV_EXPECTED_LABEL"] = candidate_df["GAP_PREV_EXPECTED_LABEL"].fillna(0.0)
    candidate_df["GAP_NEXT_EXPECTED_LABEL"] = candidate_df["GAP_NEXT_EXPECTED_LABEL"].fillna(0.0)

    return candidate_df


def fit_base_model_for_season(
    df: pd.DataFrame,
    test_season: int,
    feature_columns: list[str],
    config: dict,
) -> Pipeline:
    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] < test_season)
    ].copy()

    if train_df.empty:
        raise RuntimeError(f"Empty base training data for season {test_season}")

    y_train = train_df[LABEL_COL].astype(int)
    sample_weight = make_sample_weight(y_train, config["base_weight_mode"])

    model = make_hgb_classifier(config)
    model.fit(
        train_df[feature_columns],
        y_train,
        model__sample_weight=sample_weight,
    )

    return model


def build_top15_pool_for_season(
    df: pd.DataFrame,
    test_season: int,
    feature_columns: list[str],
    config: dict,
) -> pd.DataFrame:
    test_df = df[df["SEASON_END_YEAR"] == test_season].copy()

    if test_df.empty:
        raise RuntimeError(f"Empty test data for season {test_season}")

    base_model = fit_base_model_for_season(
        df=df,
        test_season=test_season,
        feature_columns=feature_columns,
        config=config,
    )

    scored_df = add_base_scores(
        test_df,
        model=base_model,
        feature_columns=feature_columns,
    )

    top15_df = select_top_unique_players(
        scored_df,
        score_col=config["base_selection_score_col"],
        top_n=15,
    )

    top15_df = add_base_rank_columns(top15_df)

    return top15_df


def build_top15_pools(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
    candidate_pool_start_season: int,
) -> dict[int, pd.DataFrame]:
    pools = {}

    print()
    print("=" * 80)
    print("BUILDING ROLLING BASELINE TOP-15 POOLS")
    print("=" * 80)

    for season in range(candidate_pool_start_season, MAX_TRAIN_SEASON + 1):
        top15_df = build_top15_pool_for_season(
            df=df,
            test_season=season,
            feature_columns=feature_columns,
            config=config,
        )

        pools[season] = top15_df
        true_positive_count = int((top15_df[LABEL_COL].astype(int) > 0).sum())

        print(
            f"  {season}: top15={len(top15_df):>2} | "
            f"true_all_nba_inside={true_positive_count:>2}/15",
            flush=True,
        )

    return pools


def assign_by_sorted_score(
    selected_df: pd.DataFrame,
    score_col: str,
) -> pd.DataFrame:
    pred = selected_df.sort_values(score_col, ascending=False).copy().reset_index(drop=True)

    pred["PREDICTED_LABEL"] = 0
    pred.loc[0:4, "PREDICTED_LABEL"] = 3
    pred.loc[5:9, "PREDICTED_LABEL"] = 2
    pred.loc[10:14, "PREDICTED_LABEL"] = 1

    return pred


def assign_from_team_value_matrix(
    selected_df: pd.DataFrame,
    value_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Freeze the exact 15 players and choose only team assignment.

    value_matrix columns are:
        0 -> predicted 1st team label 3
        1 -> predicted 2nd team label 2
        2 -> predicted 3rd team label 1

    We solve this as a linear assignment problem with 15 slots:
        5 first-team slots, 5 second-team slots, 5 third-team slots.
    """
    if len(selected_df) != 15:
        raise RuntimeError(f"Expected exactly 15 selected players, got {len(selected_df)}")

    value_matrix = np.asarray(value_matrix, dtype=float)
    if value_matrix.shape != (15, 3):
        raise RuntimeError(f"Expected value matrix shape (15, 3), got {value_matrix.shape}")

    slot_values = np.concatenate(
        [
            np.repeat(value_matrix[:, [0]], 5, axis=1),
            np.repeat(value_matrix[:, [1]], 5, axis=1),
            np.repeat(value_matrix[:, [2]], 5, axis=1),
        ],
        axis=1,
    )

    # linear_sum_assignment minimizes cost, so use negative value.
    row_ind, col_ind = linear_sum_assignment(-slot_values)

    pred = selected_df.copy().reset_index(drop=True)
    pred["PREDICTED_LABEL"] = 0
    pred["ASSIGNMENT_VALUE"] = 0.0

    for row, col in zip(row_ind, col_ind):
        if 0 <= col <= 4:
            label = 3
            team_col = 0
        elif 5 <= col <= 9:
            label = 2
            team_col = 1
        else:
            label = 1
            team_col = 2

        pred.loc[row, "PREDICTED_LABEL"] = label
        pred.loc[row, "ASSIGNMENT_VALUE"] = value_matrix[row, team_col]

    return pred

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


def score_prediction(prediction_df: pd.DataFrame, true_df: pd.DataFrame) -> dict:
    true_labels = dict(
        zip(
            true_df.loc[true_df[LABEL_COL] > 0, "PLAYER_NAME_KEY"],
            true_df.loc[true_df[LABEL_COL] > 0, LABEL_COL],
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

    predicted_keys = set(prediction_df["PLAYER_NAME_KEY"].tolist())
    true_keys = set(true_labels.keys())

    return {
        "score": first["total"] + second["total"] + third["total"],
        "top_hits": len(predicted_keys & true_keys),
        "first_exact": first["exact_count"],
        "second_exact": second["exact_count"],
        "third_exact": third["exact_count"],
        "first_score": first["total"],
        "second_score": second["total"],
        "third_score": third["total"],
    }


def minmax_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.ones_like(arr, dtype=float)

    return (arr - lo) / (hi - lo)


def make_reorderer_target(y: pd.Series, target_mode: str) -> pd.Series:
    mapping = TARGET_MAPS[target_mode]
    return y.astype(int).map(mapping).fillna(0.0).astype(float)


def get_reorderer_feature_columns(
    top15_df: pd.DataFrame,
    base_feature_columns: list[str],
    use_original_features: bool,
) -> list[str]:
    requested = list(BASE_SCORE_FEATURES)

    if use_original_features:
        requested += base_feature_columns

    seen = set()
    result = []

    for col in requested:
        if col in seen:
            continue
        if col not in top15_df.columns:
            continue
        if col in {LABEL_COL, "SEASON_END_YEAR", "PLAYER_NAME", "PLAYER_NAME_KEY"}:
            continue
        result.append(col)
        seen.add(col)

    if not result:
        raise RuntimeError("No reorderer features found")

    return result


def evaluate_baseline(
    df: pd.DataFrame,
    top15_pools: dict[int, pd.DataFrame],
) -> dict:
    season_results = []

    print()
    print("=" * 80)
    print("BASELINE: SAME TOP-15 + EXPECTED_LABEL ORDER")
    print("=" * 80)

    for season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        true_df = df[df["SEASON_END_YEAR"] == season].copy()
        selected_df = top15_pools[season]

        prediction_df = assign_by_sorted_score(selected_df, score_col="EXPECTED_LABEL")
        score_info = score_prediction(prediction_df, true_df)

        season_results.append({"season": season, **score_info})

        print(
            f"  base {season}: score={score_info['score']:>3}/270 | "
            f"hits={score_info['top_hits']:>2}/15 | "
            f"exact=({score_info['first_exact']}, "
            f"{score_info['second_exact']}, {score_info['third_exact']})",
            flush=True,
        )

    return summarize_results("baseline", {}, season_results)


def train_reorderer_for_season(
    top15_pools: dict[int, pd.DataFrame],
    test_season: int,
    feature_columns: list[str],
    config: dict,
) -> Pipeline:
    train_pools = [
        pool
        for season, pool in top15_pools.items()
        if season < test_season
    ]

    if not train_pools:
        raise RuntimeError(f"No reorderer training pools for season {test_season}")

    train_df = pd.concat(train_pools, ignore_index=True)
    y_label = train_df[LABEL_COL].astype(int)
    sample_weight = make_sample_weight(y_label, config["reorderer_weight_mode"])

    if config["reorderer_model_type"] == "classifier":
        model = make_reorderer_classifier(config)
        model.fit(
            train_df[feature_columns],
            y_label,
            model__sample_weight=sample_weight,
        )
        return model

    if config["reorderer_model_type"] == "regressor":
        y_target = make_reorderer_target(y_label, config["reorderer_target_mode"])
        model = make_reorderer_regressor(config)
        model.fit(
            train_df[feature_columns],
            y_target,
            model__sample_weight=sample_weight,
        )
        return model

    raise ValueError(f"Unknown reorderer_model_type: {config['reorderer_model_type']}")


def make_model_team_values(
    model: Pipeline,
    selected_df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
) -> np.ndarray:
    if config["reorderer_model_type"] == "classifier":
        proba = predict_proba_0123(model, selected_df[feature_columns])
        p_third = proba[:, 1]
        p_second = proba[:, 2]
        p_first = proba[:, 3]

        first_value = 10.0 * p_first + 8.0 * p_second + 6.0 * p_third
        second_value = 8.0 * p_first + 10.0 * p_second + 8.0 * p_third
        third_value = 6.0 * p_first + 8.0 * p_second + 10.0 * p_third

        return np.column_stack([first_value, second_value, third_value])

    if config["reorderer_model_type"] == "regressor":
        raw_score = model.predict(selected_df[feature_columns])
        raw_score = np.asarray(raw_score, dtype=float)

        # Convert a single rank-like score into soft utilities for each team slot.
        # Higher score prefers 1st, middle prefers 2nd, lower prefers 3rd.
        score_norm = minmax_array(raw_score)
        first_value = score_norm
        third_value = 1.0 - score_norm
        second_value = 1.0 - np.abs(score_norm - 0.5) * 2.0

        return np.column_stack([first_value, second_value, third_value])

    raise ValueError(f"Unknown reorderer_model_type: {config['reorderer_model_type']}")


def make_base_team_values(selected_df: pd.DataFrame) -> np.ndarray:
    return selected_df[["FIRST_VALUE", "SECOND_VALUE", "THIRD_VALUE"]].to_numpy(dtype=float)


def evaluate_reorderer_config(
    df: pd.DataFrame,
    top15_pools: dict[int, pd.DataFrame],
    feature_columns: list[str],
    config: dict,
) -> dict:
    season_results = []

    print()
    print("=" * 80)
    print(f"REORDERER CONFIG: {config}")
    print("=" * 80)

    for season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        true_df = df[df["SEASON_END_YEAR"] == season].copy()
        selected_df = top15_pools[season].copy().reset_index(drop=True)

        reorderer = train_reorderer_for_season(
            top15_pools=top15_pools,
            test_season=season,
            feature_columns=feature_columns,
            config=config,
        )

        base_values = make_base_team_values(selected_df)
        model_values = make_model_team_values(
            reorderer,
            selected_df,
            feature_columns=feature_columns,
            config=config,
        )

        base_norm = minmax_array(base_values)
        model_norm = minmax_array(model_values)

        blend = float(config["blend_weight"])
        final_values = (1.0 - blend) * base_norm + blend * model_norm

        prediction_df = assign_from_team_value_matrix(selected_df, final_values)
        score_info = score_prediction(prediction_df, true_df)
        season_results.append({"season": season, **score_info})

        print(
            f"  reorder {season}: score={score_info['score']:>3}/270 | "
            f"hits={score_info['top_hits']:>2}/15 | "
            f"exact=({score_info['first_exact']}, "
            f"{score_info['second_exact']}, {score_info['third_exact']})",
            flush=True,
        )

    return summarize_results("reorderer", config, season_results)


def summarize_results(name: str, config: dict, season_results: list[dict]) -> dict:
    avg_score = float(np.mean([row["score"] for row in season_results]))
    avg_hits = float(np.mean([row["top_hits"] for row in season_results]))
    avg_first_exact = float(np.mean([row["first_exact"] for row in season_results]))
    avg_second_exact = float(np.mean([row["second_exact"] for row in season_results]))
    avg_third_exact = float(np.mean([row["third_exact"] for row in season_results]))

    return {
        "name": name,
        "config": config,
        "avg_score": avg_score,
        "avg_hits": avg_hits,
        "avg_first_exact": avg_first_exact,
        "avg_second_exact": avg_second_exact,
        "avg_third_exact": avg_third_exact,
        "season_results": season_results,
    }


def build_reorderer_configs(args) -> list[dict]:
    configs = []

    for reorderer_model_type in args.reorderer_model_types:
        for reorderer_max_iter in args.reorderer_max_iters:
            for reorderer_learning_rate in args.reorderer_learning_rates:
                for reorderer_max_leaf_nodes in args.reorderer_max_leaf_nodes:
                    for reorderer_l2_regularization in args.reorderer_l2_regularization:
                        for reorderer_weight_mode in args.reorderer_weight_modes:
                            for reorderer_target_mode in args.reorderer_target_modes:
                                for blend_weight in args.blend_weights:
                                    if reorderer_model_type == "classifier" and reorderer_target_mode != "spaced":
                                        continue

                                    config = {
                                        "base_feature_set": args.base_feature_set,
                                        "base_weight_mode": args.base_weight_mode,
                                        "base_selection_score_col": args.base_selection_score_col,
                                        "base_max_iter": args.base_max_iter,
                                        "base_learning_rate": args.base_learning_rate,
                                        "base_max_leaf_nodes": args.base_max_leaf_nodes,
                                        "base_l2_regularization": args.base_l2_regularization,
                                        "candidate_pool_start_season": args.candidate_pool_start_season,
                                        "reorderer_model_type": reorderer_model_type,
                                        "reorderer_max_iter": reorderer_max_iter,
                                        "reorderer_learning_rate": reorderer_learning_rate,
                                        "reorderer_max_leaf_nodes": reorderer_max_leaf_nodes,
                                        "reorderer_l2_regularization": reorderer_l2_regularization,
                                        "reorderer_weight_mode": reorderer_weight_mode,
                                        "reorderer_target_mode": reorderer_target_mode,
                                        "blend_weight": blend_weight,
                                        "use_original_features": args.use_original_features,
                                    }
                                    configs.append(config)

    return configs


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
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

    parser.add_argument("--base-feature-set", default="previous_team_share_allstar")
    parser.add_argument("--base-weight-mode", default="positive_boost")
    parser.add_argument("--base-selection-score-col", default="EXPECTED_LABEL")
    parser.add_argument("--base-max-iter", type=int, default=250)
    parser.add_argument("--base-learning-rate", type=float, default=0.05)
    parser.add_argument("--base-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--base-l2-regularization", type=float, default=0.05)

    parser.add_argument("--candidate-pool-start-season", type=int, default=2005)

    parser.add_argument(
        "--reorderer-model-types",
        nargs="+",
        default=["classifier"],
        choices=["classifier", "regressor"],
    )
    parser.add_argument("--reorderer-max-iters", nargs="+", type=int, default=[150])
    parser.add_argument("--reorderer-learning-rates", nargs="+", type=float, default=[0.03])
    parser.add_argument("--reorderer-max-leaf-nodes", nargs="+", type=int, default=[15])
    parser.add_argument("--reorderer-l2-regularization", nargs="+", type=float, default=[0.05])
    parser.add_argument(
        "--reorderer-weight-modes",
        nargs="+",
        default=["team_weighted"],
        choices=["none", "positive_boost", "team_weighted", "sqrt_class_balance"],
    )
    parser.add_argument(
        "--reorderer-target-modes",
        nargs="+",
        default=["spaced"],
        choices=list(TARGET_MAPS.keys()),
    )
    parser.add_argument("--blend-weights", nargs="+", type=float, default=[0.25])
    parser.add_argument("--use-original-features", action="store_true", default=True)

    args = parser.parse_args()

    df = load_dataset()

    print()
    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"Dataset: {ALL_NBA_DATASET_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Seasons: {df['SEASON_END_YEAR'].min()} - {df['SEASON_END_YEAR'].max()}")
    print()
    print(df[LABEL_COL].value_counts().sort_index())

    base_feature_columns = get_feature_columns(
        df,
        feature_set=args.base_feature_set,
        verbose=True,
    )

    base_config = {
        "base_feature_set": args.base_feature_set,
        "base_weight_mode": args.base_weight_mode,
        "base_selection_score_col": args.base_selection_score_col,
        "base_max_iter": args.base_max_iter,
        "base_learning_rate": args.base_learning_rate,
        "base_max_leaf_nodes": args.base_max_leaf_nodes,
        "base_l2_regularization": args.base_l2_regularization,
    }

    top15_pools = build_top15_pools(
        df=df,
        feature_columns=base_feature_columns,
        config=base_config,
        candidate_pool_start_season=args.candidate_pool_start_season,
    )

    first_pool = next(iter(top15_pools.values()))
    reorderer_feature_columns = get_reorderer_feature_columns(
        first_pool,
        base_feature_columns=base_feature_columns,
        use_original_features=args.use_original_features,
    )

    print()
    print("=" * 80)
    print("REORDERER FEATURE INFO")
    print("=" * 80)
    print(f"Reorderer features: {len(reorderer_feature_columns)}")
    print(f"Use original features: {args.use_original_features}")
    print("First features:")
    for col in reorderer_feature_columns[:40]:
        print(f"  - {col}")
    if len(reorderer_feature_columns) > 40:
        print(f"  ... +{len(reorderer_feature_columns) - 40} more")

    baseline_result = evaluate_baseline(
        df=df,
        top15_pools=top15_pools,
    )

    reorderer_configs = build_reorderer_configs(args)

    print()
    print("=" * 80)
    print("REORDERER SEARCH")
    print("=" * 80)
    print(f"Configs to test: {len(reorderer_configs)}")

    results = [baseline_result]

    for idx, config in enumerate(reorderer_configs, start=1):
        print()
        print("=" * 80)
        print(f"START [{idx:03d}/{len(reorderer_configs):03d}]")
        print("=" * 80)

        result = evaluate_reorderer_config(
            df=df,
            top15_pools=top15_pools,
            feature_columns=reorderer_feature_columns,
            config=config,
        )

        results.append(result)

        print(
            f"DONE [{idx:03d}/{len(reorderer_configs):03d}] "
            f"score={result['avg_score']:.2f}/270 | "
            f"hits={result['avg_hits']:.2f}/15 | "
            f"exact=({result['avg_first_exact']:.2f}, "
            f"{result['avg_second_exact']:.2f}, "
            f"{result['avg_third_exact']:.2f}) | "
            f"{config}",
            flush=True,
        )

    results_sorted = sorted(results, key=lambda row: row["avg_score"], reverse=True)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for result in results_sorted:
        config = result["config"]
        name = result["name"]

        if name == "baseline":
            config_text = "baseline"
        else:
            config_text = (
                f"type={config['reorderer_model_type']} | "
                f"blend={config['blend_weight']} | "
                f"iter={config['reorderer_max_iter']} | "
                f"lr={config['reorderer_learning_rate']} | "
                f"leaf={config['reorderer_max_leaf_nodes']} | "
                f"l2={config['reorderer_l2_regularization']} | "
                f"target={config['reorderer_target_mode']} | "
                f"weight={config['reorderer_weight_mode']}"
            )

        print(
            f" {result['avg_score']:6.2f}/270 | "
            f"hits={result['avg_hits']:5.2f}/15 | "
            f"exact=({result['avg_first_exact']:.2f}, "
            f"{result['avg_second_exact']:.2f}, "
            f"{result['avg_third_exact']:.2f}) | "
            f"{name} | {config_text}"
        )

    best = results_sorted[0]

    print()
    print("=" * 80)
    print("BEST CONFIG")
    print("=" * 80)
    print(f"name={best['name']}")
    print(f"score={best['avg_score']:.2f}/270")
    print(f"hits={best['avg_hits']:.2f}/15")
    print(
        f"exact=({best['avg_first_exact']:.2f}, "
        f"{best['avg_second_exact']:.2f}, "
        f"{best['avg_third_exact']:.2f})"
    )
    print(best["config"])

    report = {
        "base_config": base_config,
        "candidate_pool_start_season": args.candidate_pool_start_season,
        "reorderer_feature_columns": reorderer_feature_columns,
        "results": results_sorted,
        "best": best,
    }

    REPORT_PATH.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Report saved: {REPORT_PATH}")

    from unified_best_output import write_from_all_nba_stage2_report
    write_from_all_nba_stage2_report(
        report_path=REPORT_PATH,
        hgb_report_path=HGB_FULL_REPORT_PATH,
        out_dir=UNIFIED_OUTPUT_DIR,
        experiment="team_reorderer",
    )


if __name__ == "__main__":
    main()
