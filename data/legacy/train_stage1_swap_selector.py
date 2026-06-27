from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from pathlib import Path
import argparse
import itertools
import json
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from features import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_NBA_DATASET_PATH = PROCESSED_DIR / "all_nba_dataset.csv"
REPORT_PATH = MODELS_DIR / "stage1_swap_selector_report.json"
UNIFIED_OUTPUT_DIR = MODELS_DIR / "unified_best_experiments"
HGB_FULL_REPORT_PATH = MODELS_DIR / "hgb_classifier_full_score_report.json"
PREDICTIONS_PATH = MODELS_DIR / "stage1_swap_selector_predictions.csv"
SUMMARY_PATH = MODELS_DIR / "stage1_swap_selector_summary.csv"
ALL_RESULTS_PATH = MODELS_DIR / "stage1_swap_selector_all_results.csv"

MIN_TRAIN_SEASON = 2000
BACKTEST_START_SEASON = 2010
MAX_TRAIN_SEASON = 2025
RANDOM_STATE = 42

LABEL_COL = "ALL_NBA_LABEL"

ID_COLS = {
    "SEASON_END_YEAR",
    "PLAYER_NAME",
    "PLAYER_NAME_KEY",
    LABEL_COL,
}

BASE_META_COLS = [
    "BASE_RANK",
    "BASE_RANK_PCT",
    "BASE_RANK_INV_POOL",
    "DIST_FROM_15",
    "IS_BASE_TOP5",
    "IS_BASE_TOP8",
    "IS_BASE_TOP10",
    "IS_BASE_TOP12",
    "IS_BASE_TOP15",
    "GAP_PREV_PRIMARY",
    "GAP_NEXT_PRIMARY",
    "CONSENSUS_TOP10_COUNT",
    "CONSENSUS_TOP15_COUNT",
    "CONSENSUS_TOP20_COUNT",
    "CONSENSUS_TOP25_COUNT",
    "CONSENSUS_TOP30_COUNT",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ALL_NBA_DATASET_PATH)

    required = ["SEASON_END_YEAR", "PLAYER_NAME", "PLAYER_NAME_KEY", LABEL_COL]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def make_base_classifier(config: dict) -> Pipeline:
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


def make_selector_classifier(config: dict) -> Pipeline:
    model = HistGradientBoostingClassifier(
        max_iter=config["selector_max_iter"],
        learning_rate=config["selector_learning_rate"],
        max_leaf_nodes=config["selector_max_leaf_nodes"],
        l2_regularization=config["selector_l2_regularization"],
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def make_multiclass_sample_weight(y: pd.Series, mode: str) -> np.ndarray:
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

    raise ValueError(f"Unknown base weight mode: {mode}")


def make_binary_sample_weight(y: pd.Series, mode: str) -> np.ndarray:
    y_values = y.astype(int).to_numpy()

    if mode == "none":
        return np.ones(len(y_values), dtype=float)

    if mode == "positive_2":
        weights = np.ones(len(y_values), dtype=float)
        weights[y_values == 1] = 2.0
        return weights

    if mode == "positive_3":
        weights = np.ones(len(y_values), dtype=float)
        weights[y_values == 1] = 3.0
        return weights

    if mode == "sqrt_balance":
        counts = pd.Series(y_values).value_counts().to_dict()
        max_count = max(counts.values())
        weights = np.ones(len(y_values), dtype=float)
        for cls, count in counts.items():
            weights[y_values == cls] = np.sqrt(max_count / count)
        return weights

    if mode == "balanced":
        counts = pd.Series(y_values).value_counts().to_dict()
        total = len(y_values)
        weights = np.ones(len(y_values), dtype=float)
        for cls, count in counts.items():
            weights[y_values == cls] = total / (len(counts) * count)
        return weights

    raise ValueError(f"Unknown selector weight mode: {mode}")


def predict_proba_0123(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    raw_proba = model.predict_proba(X)
    classes = model.named_steps["model"].classes_

    proba = np.zeros((len(X), 4), dtype=float)
    for src_idx, cls in enumerate(classes):
        cls = int(cls)
        if 0 <= cls <= 3:
            proba[:, cls] = raw_proba[:, src_idx]

    return proba


def predict_binary_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    raw_proba = model.predict_proba(X)
    classes = model.named_steps["model"].classes_

    if 1 in classes:
        idx = int(np.where(classes == 1)[0][0])
        return raw_proba[:, idx]

    return np.zeros(len(X), dtype=float)


def add_hgb_component_scores(
    df: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    out = df.copy()
    proba = predict_proba_0123(model, out[feature_columns])

    p_none = proba[:, 0]
    p_third = proba[:, 1]
    p_second = proba[:, 2]
    p_first = proba[:, 3]

    out[f"{prefix}__P_NONE"] = p_none
    out[f"{prefix}__P_THIRD"] = p_third
    out[f"{prefix}__P_SECOND"] = p_second
    out[f"{prefix}__P_FIRST"] = p_first
    out[f"{prefix}__P_ALL_NBA"] = p_third + p_second + p_first
    out[f"{prefix}__EXPECTED_LABEL"] = 1.0 * p_third + 2.0 * p_second + 3.0 * p_first

    out[f"{prefix}__FIRST_VALUE"] = 10.0 * p_first + 8.0 * p_second + 6.0 * p_third
    out[f"{prefix}__SECOND_VALUE"] = 8.0 * p_first + 10.0 * p_second + 8.0 * p_third
    out[f"{prefix}__THIRD_VALUE"] = 6.0 * p_first + 8.0 * p_second + 10.0 * p_third
    out[f"{prefix}__BEST_TEAM_VALUE"] = out[
        [
            f"{prefix}__FIRST_VALUE",
            f"{prefix}__SECOND_VALUE",
            f"{prefix}__THIRD_VALUE",
        ]
    ].max(axis=1)

    return out


def rank_unique_players(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).copy()
    ranked = ranked.drop_duplicates(subset=["PLAYER_NAME_KEY"], keep="first")
    ranked = ranked.reset_index(drop=True)
    ranked["BASE_RANK"] = np.arange(1, len(ranked) + 1)

    if len(ranked) <= 1:
        ranked["BASE_RANK_PCT"] = 1.0
    else:
        ranked["BASE_RANK_PCT"] = 1.0 - ((ranked["BASE_RANK"] - 1) / (len(ranked) - 1))

    ranked["DIST_FROM_15"] = ranked["BASE_RANK"] - 15
    ranked["IS_BASE_TOP5"] = (ranked["BASE_RANK"] <= 5).astype(int)
    ranked["IS_BASE_TOP8"] = (ranked["BASE_RANK"] <= 8).astype(int)
    ranked["IS_BASE_TOP10"] = (ranked["BASE_RANK"] <= 10).astype(int)
    ranked["IS_BASE_TOP12"] = (ranked["BASE_RANK"] <= 12).astype(int)
    ranked["IS_BASE_TOP15"] = (ranked["BASE_RANK"] <= 15).astype(int)

    primary_score = ranked[score_col].astype(float)
    ranked["GAP_PREV_PRIMARY"] = (primary_score.shift(1) - primary_score).fillna(0.0)
    ranked["GAP_NEXT_PRIMARY"] = (primary_score - primary_score.shift(-1)).fillna(0.0)

    return ranked


def add_component_rank_features(
    ranked: pd.DataFrame,
    component_score_cols: list[str],
) -> pd.DataFrame:
    out = ranked.copy()

    for col in component_score_cols:
        if col not in out.columns:
            continue

        tmp = out[["PLAYER_NAME_KEY", col]].sort_values(col, ascending=False).copy()
        tmp = tmp.drop_duplicates(subset=["PLAYER_NAME_KEY"], keep="first").reset_index(drop=True)
        tmp[f"RANK__{col}"] = np.arange(1, len(tmp) + 1)

        if len(tmp) <= 1:
            tmp[f"RANKPCT__{col}"] = 1.0
        else:
            tmp[f"RANKPCT__{col}"] = 1.0 - ((tmp[f"RANK__{col}"] - 1) / (len(tmp) - 1))

        rank_map = dict(zip(tmp["PLAYER_NAME_KEY"], tmp[f"RANK__{col}"]))
        pct_map = dict(zip(tmp["PLAYER_NAME_KEY"], tmp[f"RANKPCT__{col}"]))

        out[f"RANK__{col}"] = out["PLAYER_NAME_KEY"].map(rank_map)
        out[f"RANKPCT__{col}"] = out["PLAYER_NAME_KEY"].map(pct_map)

    rank_cols = [f"RANK__{col}" for col in component_score_cols if f"RANK__{col}" in out.columns]

    for k in [10, 15, 20, 25, 30]:
        if rank_cols:
            out[f"CONSENSUS_TOP{k}_COUNT"] = out[rank_cols].le(k).sum(axis=1)
        else:
            out[f"CONSENSUS_TOP{k}_COUNT"] = 0

    return out


def add_pool_rank_score(pool_df: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    out = pool_df.copy()
    if pool_size <= 1:
        out["BASE_RANK_INV_POOL"] = 1.0
    else:
        out["BASE_RANK_INV_POOL"] = 1.0 - ((out["BASE_RANK"] - 1) / (pool_size - 1))
    out["BASE_RANK_INV_POOL"] = out["BASE_RANK_INV_POOL"].clip(lower=0.0, upper=1.0)
    return out


def prepare_feature_sets(df: pd.DataFrame, feature_sets: list[str]) -> dict[str, list[str]]:
    feature_map = {}

    print()
    print("=" * 80)
    print("FEATURE SET INFO")
    print("=" * 80)

    for fs in feature_sets:
        try:
            cols = get_feature_columns(df, feature_set=fs, verbose=True)
        except Exception as exc:
            print(f"WARNING: skip feature_set={fs}: {exc}")
            continue

        cleaned = []
        for col in cols:
            if col in ID_COLS:
                continue
            if col not in df.columns:
                continue
            cleaned.append(col)

        if not cleaned:
            print(f"WARNING: skip feature_set={fs}: no usable columns")
            continue

        feature_map[fs] = cleaned

    if not feature_map:
        raise RuntimeError("No usable feature sets")

    return feature_map


def fit_base_model_for_component(
    df: pd.DataFrame,
    season: int,
    feature_columns: list[str],
    config: dict,
) -> Pipeline:
    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] < season)
    ].copy()

    if train_df.empty:
        raise RuntimeError(f"Empty base train data for season={season}")

    y = train_df[LABEL_COL].astype(int)
    weights = make_multiclass_sample_weight(y, config["base_weight_mode"])

    model = make_base_classifier(config)
    model.fit(train_df[feature_columns], y, model__sample_weight=weights)
    return model


def build_rolling_scored_seasons(
    df: pd.DataFrame,
    feature_map: dict[str, list[str]],
    config: dict,
    start_season: int,
    component_score_names: list[str],
) -> tuple[dict[int, pd.DataFrame], list[str]]:
    scored_by_season = {}
    all_component_cols = []

    primary_prefix = f"hgb__{config['primary_feature_set']}"
    primary_score_col = f"{primary_prefix}__{config['base_score_col']}"

    print()
    print("=" * 80)
    print("BUILDING ROLLING SCORED SEASONS")
    print("=" * 80)

    for season in range(start_season, MAX_TRAIN_SEASON + 1):
        test_df = df[df["SEASON_END_YEAR"] == season].copy().reset_index(drop=True)

        if test_df.empty:
            continue

        scored = test_df.copy()
        season_component_cols = []

        for feature_set, feature_columns in feature_map.items():
            model = fit_base_model_for_component(
                df=df,
                season=season,
                feature_columns=feature_columns,
                config=config,
            )

            prefix = f"hgb__{feature_set}"
            scored = add_hgb_component_scores(
                scored,
                model=model,
                feature_columns=feature_columns,
                prefix=prefix,
            )

            for score_name in component_score_names:
                col = f"{prefix}__{score_name}"
                if col in scored.columns:
                    season_component_cols.append(col)

        if primary_score_col not in scored.columns:
            raise RuntimeError(f"Missing primary score column: {primary_score_col}")

        all_component_cols = sorted(set(all_component_cols) | set(season_component_cols))

        ranked = rank_unique_players(scored, primary_score_col)
        ranked = add_component_rank_features(ranked, season_component_cols)

        true_keys = set(
            ranked.loc[ranked[LABEL_COL].astype(int) > 0, "PLAYER_NAME_KEY"].tolist()
        )
        top15_keys = set(ranked.head(15)["PLAYER_NAME_KEY"].tolist())
        top30_keys = set(ranked.head(30)["PLAYER_NAME_KEY"].tolist())

        scored_by_season[season] = ranked

        print(
            f"  {season}: top15_hits={len(top15_keys & true_keys):>2}/15 | "
            f"top30_recall={len(top30_keys & true_keys):>2}/15 | "
            f"components={len(season_component_cols)}",
            flush=True,
        )

    return scored_by_season, all_component_cols


def get_selector_feature_columns(
    sample_df: pd.DataFrame,
    component_cols: list[str],
    original_feature_cols: list[str],
    use_original_features: bool,
) -> list[str]:
    requested = []

    requested += BASE_META_COLS

    for col in component_cols:
        for candidate in [
            col,
            f"RANK__{col}",
            f"RANKPCT__{col}",
        ]:
            if candidate in sample_df.columns:
                requested.append(candidate)

    if use_original_features:
        requested += original_feature_cols

    result = []
    seen = set()
    forbidden = {
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
        LABEL_COL,
        "IS_ALL_NBA",
        "SELECTOR_TARGET",
        "SELECTOR_PROBA",
        "FINAL_SELECTOR_SCORE",
        "PREDICTED_LABEL",
    }

    for col in requested:
        if col in seen:
            continue
        if col in forbidden:
            continue
        if col not in sample_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(sample_df[col]):
            continue

        result.append(col)
        seen.add(col)

    return result


def fit_selector(
    train_pool: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
) -> Pipeline:
    y = (train_pool[LABEL_COL].astype(int) > 0).astype(int)

    if y.nunique() < 2:
        raise RuntimeError("Selector target has only one class")

    weights = make_binary_sample_weight(y, config["selector_weight_mode"])

    model = make_selector_classifier(config)
    model.fit(train_pool[feature_columns], y, model__sample_weight=weights)
    return model


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


def select_swap(
    pool_df: pd.DataFrame,
    remove_zone_start: int,
    max_swaps: int,
    margin: float,
) -> pd.DataFrame:
    pool = pool_df.sort_values("BASE_RANK").copy().reset_index(drop=True)

    selected = pool[pool["BASE_RANK"] <= 15].copy()
    outside = pool[pool["BASE_RANK"] > 15].copy()

    selected_keys = set(selected["PLAYER_NAME_KEY"].tolist())

    for _ in range(max_swaps):
        removable = selected[selected["BASE_RANK"] >= remove_zone_start].copy()
        outside_available = outside[~outside["PLAYER_NAME_KEY"].isin(selected_keys)].copy()

        if removable.empty or outside_available.empty:
            break

        remove_idx = removable["FINAL_SELECTOR_SCORE"].idxmin()
        add_idx = outside_available["FINAL_SELECTOR_SCORE"].idxmax()

        remove_score = float(selected.loc[remove_idx, "FINAL_SELECTOR_SCORE"])
        add_score = float(outside_available.loc[add_idx, "FINAL_SELECTOR_SCORE"])

        if add_score <= remove_score + margin:
            break

        remove_key = selected.loc[remove_idx, "PLAYER_NAME_KEY"]
        add_row = outside_available.loc[[add_idx]].copy()

        selected = selected[selected["PLAYER_NAME_KEY"] != remove_key].copy()
        selected = pd.concat([selected, add_row], ignore_index=True)
        selected_keys = set(selected["PLAYER_NAME_KEY"].tolist())

    return selected.sort_values("BASE_RANK").head(15).copy()


def select_locked_core(
    pool_df: pd.DataFrame,
    remove_zone_start: int,
) -> pd.DataFrame:
    pool = pool_df.copy()

    locked = pool[pool["BASE_RANK"] < remove_zone_start].copy()
    contest = pool[pool["BASE_RANK"] >= remove_zone_start].copy()

    slots_left = 15 - len(locked)
    if slots_left <= 0:
        return locked.sort_values("BASE_RANK").head(15).copy()

    chosen = contest.sort_values("FINAL_SELECTOR_SCORE", ascending=False).head(slots_left)
    selected = pd.concat([locked, chosen], ignore_index=True)
    return selected.sort_values("BASE_RANK").head(15).copy()


def apply_selection_mode(
    pool_df: pd.DataFrame,
    mode: str,
    remove_zone_start: int,
    max_swaps: int,
    margin: float,
) -> pd.DataFrame:
    if mode == "swap":
        return select_swap(
            pool_df=pool_df,
            remove_zone_start=remove_zone_start,
            max_swaps=max_swaps,
            margin=margin,
        )

    if mode == "locked_core":
        return select_locked_core(
            pool_df=pool_df,
            remove_zone_start=remove_zone_start,
        )

    raise ValueError(f"Unknown selection mode: {mode}")


def summarize_results(result_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(result_rows)

    group_cols = [
        "name",
        "selection_mode",
        "pool_size",
        "remove_zone_start",
        "max_swaps",
        "margin",
        "selector_blend",
        "selector_max_iter",
        "selector_learning_rate",
        "selector_max_leaf_nodes",
        "selector_l2_regularization",
        "selector_weight_mode",
        "assignment_score_col",
        "use_original_features",
    ]

    existing_group_cols = [c for c in group_cols if c in df.columns]

    summary = (
        df.groupby(existing_group_cols, dropna=False)
        .agg(
            score=("score", "mean"),
            hits=("top_hits", "mean"),
            first_exact=("first_exact", "mean"),
            second_exact=("second_exact", "mean"),
            third_exact=("third_exact", "mean"),
            min_hits=("top_hits", "min"),
            max_hits=("top_hits", "max"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        ["hits", "score", "first_exact", "second_exact", "third_exact"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    return summary


def print_summary_table(summary: pd.DataFrame, title: str, n: int = 30) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for _, row in summary.head(n).iterrows():
        print(
            f"{row['score']:7.2f}/270 | "
            f"hits={row['hits']:5.2f}/15 | "
            f"min={int(row['min_hits'])}/15 | "
            f"max={int(row['max_hits'])}/15 | "
            f"exact=({row['first_exact']:.2f}, {row['second_exact']:.2f}, {row['third_exact']:.2f}) | "
            f"{row['name']} | "
            f"mode={row.get('selection_mode', '-')}, "
            f"pool={row.get('pool_size', '-')}, "
            f"remove_from={row.get('remove_zone_start', '-')}, "
            f"swaps={row.get('max_swaps', '-')}, "
            f"margin={row.get('margin', '-')}, "
            f"blend={row.get('selector_blend', '-')}, "
            f"iter={row.get('selector_max_iter', '-')}, "
            f"lr={row.get('selector_learning_rate', '-')}, "
            f"leaf={row.get('selector_max_leaf_nodes', '-')}, "
            f"l2={row.get('selector_l2_regularization', '-')}, "
            f"w={row.get('selector_weight_mode', '-')}, "
            f"assign={row.get('assignment_score_col', '-')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--primary-feature-set", default="previous_team_share_allstar")
    parser.add_argument("--extra-feature-sets", nargs="*", default=[])
    parser.add_argument(
        "--component-score-cols",
        nargs="+",
        default=["EXPECTED_LABEL", "P_ALL_NBA", "BEST_TEAM_VALUE"],
        choices=["EXPECTED_LABEL", "P_ALL_NBA", "BEST_TEAM_VALUE"],
    )
    parser.add_argument(
        "--base-score-col",
        default="EXPECTED_LABEL",
        choices=["EXPECTED_LABEL", "P_ALL_NBA", "BEST_TEAM_VALUE"],
    )

    parser.add_argument("--base-weight-mode", default="positive_boost", choices=["none", "positive_boost", "team_weighted", "sqrt_class_balance"])
    parser.add_argument("--base-max-iter", type=int, default=250)
    parser.add_argument("--base-learning-rate", type=float, default=0.05)
    parser.add_argument("--base-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--base-l2-regularization", type=float, default=0.05)

    parser.add_argument("--candidate-pool-start-season", type=int, default=2005)
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[20])
    parser.add_argument("--selection-modes", nargs="+", default=["swap"], choices=["swap", "locked_core"])
    parser.add_argument("--remove-zone-starts", nargs="+", type=int, default=[15])
    parser.add_argument("--max-swaps", nargs="+", type=int, default=[1])
    parser.add_argument("--margins", nargs="+", type=float, default=[0.01])
    parser.add_argument("--selector-blends", nargs="+", type=float, default=[0.5])

    parser.add_argument("--selector-max-iters", nargs="+", type=int, default=[120])
    parser.add_argument("--selector-learning-rates", nargs="+", type=float, default=[0.03])
    parser.add_argument("--selector-max-leaf-nodes", nargs="+", type=int, default=[15])
    parser.add_argument("--selector-l2-regularization", nargs="+", type=float, default=[0.05])
    parser.add_argument("--selector-weight-modes", nargs="+", default=["positive_3"], choices=["none", "positive_2", "positive_3", "sqrt_balance", "balanced"])

    parser.add_argument("--assignment-score-cols", nargs="+", default=["base"], choices=["base", "final"])
    parser.add_argument("--use-original-features", action="store_true")
    parser.add_argument("--print-season-results", action="store_true")

    args = parser.parse_args()

    df = load_dataset()

    feature_sets = []
    for fs in [args.primary_feature_set] + args.extra_feature_sets:
        if fs not in feature_sets:
            feature_sets.append(fs)

    feature_map = prepare_feature_sets(df, feature_sets)

    if args.primary_feature_set not in feature_map:
        raise RuntimeError(f"Primary feature set not available: {args.primary_feature_set}")

    base_config = {
        "primary_feature_set": args.primary_feature_set,
        "base_score_col": args.base_score_col,
        "base_weight_mode": args.base_weight_mode,
        "base_max_iter": args.base_max_iter,
        "base_learning_rate": args.base_learning_rate,
        "base_max_leaf_nodes": args.base_max_leaf_nodes,
        "base_l2_regularization": args.base_l2_regularization,
    }

    primary_prefix = f"hgb__{args.primary_feature_set}"
    primary_score_col = f"{primary_prefix}__{args.base_score_col}"

    scored_by_season, component_cols = build_rolling_scored_seasons(
        df=df,
        feature_map=feature_map,
        config=base_config,
        start_season=args.candidate_pool_start_season,
        component_score_names=args.component_score_cols,
    )

    original_feature_cols = feature_map[args.primary_feature_set]
    result_rows = []
    prediction_rows = []

    # Baseline evaluation.
    for season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        if season not in scored_by_season:
            continue

        season_df = scored_by_season[season]
        selected = season_df.sort_values("BASE_RANK").head(15).copy()

        pred = assign_by_sorted_score(selected, primary_score_col)
        metrics = score_prediction(pred, season_df)

        result_rows.append(
            {
                "season": season,
                "name": "baseline",
                "selection_mode": "baseline",
                "pool_size": 15,
                "remove_zone_start": np.nan,
                "max_swaps": 0,
                "margin": np.nan,
                "selector_blend": np.nan,
                "selector_max_iter": np.nan,
                "selector_learning_rate": np.nan,
                "selector_max_leaf_nodes": np.nan,
                "selector_l2_regularization": np.nan,
                "selector_weight_mode": "none",
                "assignment_score_col": "base",
                "use_original_features": False,
                **metrics,
            }
        )

    selector_param_grid = list(
        itertools.product(
            args.pool_sizes,
            args.selector_max_iters,
            args.selector_learning_rates,
            args.selector_max_leaf_nodes,
            args.selector_l2_regularization,
            args.selector_weight_modes,
        )
    )

    selection_param_grid = list(
        itertools.product(
            args.selection_modes,
            args.remove_zone_starts,
            args.max_swaps,
            args.margins,
            args.selector_blends,
            args.assignment_score_cols,
        )
    )

    print()
    print("=" * 80)
    print("TRAINING SWAP SELECTOR")
    print("=" * 80)
    print(f"selector model fits per season: {len(selector_param_grid)}")
    print(f"selection configs per fitted selector: {len(selection_param_grid)}")

    for season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        if season not in scored_by_season:
            continue

        train_parts = []
        for train_season in range(args.candidate_pool_start_season, season):
            if train_season not in scored_by_season:
                continue
            train_parts.append(scored_by_season[train_season])

        if not train_parts:
            continue

        train_all = pd.concat(train_parts, ignore_index=True)
        test_all = scored_by_season[season].copy()

        season_best_hits = -1
        season_best_score = -1.0

        for (
            pool_size,
            selector_max_iter,
            selector_learning_rate,
            selector_max_leaf_nodes,
            selector_l2_regularization,
            selector_weight_mode,
        ) in selector_param_grid:
            train_pool = train_all[train_all["BASE_RANK"] <= pool_size].copy()
            test_pool = test_all[test_all["BASE_RANK"] <= pool_size].copy()

            if train_pool.empty or test_pool.empty:
                continue

            train_pool = add_pool_rank_score(train_pool, pool_size)
            test_pool = add_pool_rank_score(test_pool, pool_size)

            selector_config = {
                "selector_max_iter": selector_max_iter,
                "selector_learning_rate": selector_learning_rate,
                "selector_max_leaf_nodes": selector_max_leaf_nodes,
                "selector_l2_regularization": selector_l2_regularization,
                "selector_weight_mode": selector_weight_mode,
            }

            selector_features = get_selector_feature_columns(
                sample_df=train_pool,
                component_cols=component_cols,
                original_feature_cols=original_feature_cols,
                use_original_features=args.use_original_features,
            )

            if not selector_features:
                raise RuntimeError("No selector features were created")

            selector = fit_selector(
                train_pool=train_pool,
                feature_columns=selector_features,
                config=selector_config,
            )

            test_pool["SELECTOR_PROBA"] = predict_binary_proba(selector, test_pool[selector_features])

            for (
                selection_mode,
                remove_zone_start,
                max_swaps,
                margin,
                selector_blend,
                assignment_score_col_name,
            ) in selection_param_grid:
                # locked_core ignores max_swaps and margin, but keeping those
                # columns in the summary makes comparison easier.
                pool = test_pool.copy()
                pool["FINAL_SELECTOR_SCORE"] = (
                    (1.0 - selector_blend) * pool["BASE_RANK_INV_POOL"].astype(float)
                    + selector_blend * pool["SELECTOR_PROBA"].astype(float)
                )

                selected = apply_selection_mode(
                    pool_df=pool,
                    mode=selection_mode,
                    remove_zone_start=remove_zone_start,
                    max_swaps=max_swaps,
                    margin=margin,
                )

                if len(selected) != 15:
                    continue

                if assignment_score_col_name == "final":
                    assignment_score_col = "FINAL_SELECTOR_SCORE"
                else:
                    assignment_score_col = primary_score_col

                pred = assign_by_sorted_score(selected, assignment_score_col)
                metrics = score_prediction(pred, test_all)

                row = {
                    "season": season,
                    "name": "swap_selector",
                    "selection_mode": selection_mode,
                    "pool_size": pool_size,
                    "remove_zone_start": remove_zone_start,
                    "max_swaps": max_swaps,
                    "margin": margin,
                    "selector_blend": selector_blend,
                    "selector_max_iter": selector_max_iter,
                    "selector_learning_rate": selector_learning_rate,
                    "selector_max_leaf_nodes": selector_max_leaf_nodes,
                    "selector_l2_regularization": selector_l2_regularization,
                    "selector_weight_mode": selector_weight_mode,
                    "assignment_score_col": assignment_score_col_name,
                    "use_original_features": bool(args.use_original_features),
                    "num_selector_features": len(selector_features),
                    **metrics,
                }
                result_rows.append(row)

                if metrics["top_hits"] > season_best_hits or (
                    metrics["top_hits"] == season_best_hits and metrics["score"] > season_best_score
                ):
                    season_best_hits = metrics["top_hits"]
                    season_best_score = metrics["score"]

                # Store only a compact prediction trace for promising configs
                # to keep CSV size manageable.
                if metrics["top_hits"] >= 13:
                    for _, p_row in pred.iterrows():
                        prediction_rows.append(
                            {
                                **{k: row[k] for k in row if k not in metrics},
                                "player": p_row["PLAYER_NAME"],
                                "player_key": p_row["PLAYER_NAME_KEY"],
                                "base_rank": int(p_row["BASE_RANK"]),
                                "selector_proba": float(p_row.get("SELECTOR_PROBA", np.nan)),
                                "final_selector_score": float(p_row.get("FINAL_SELECTOR_SCORE", np.nan)),
                                "predicted_label": int(p_row["PREDICTED_LABEL"]),
                                "true_label": int(p_row[LABEL_COL]),
                                "is_correct_player": int(int(p_row[LABEL_COL]) > 0),
                            }
                        )

        if args.print_season_results:
            print(f"  {season}: best hits={season_best_hits}/15 | best score={season_best_score:.2f}")
        else:
            print(f"  {season}: done", flush=True)

    result_df = pd.DataFrame(result_rows)
    summary = summarize_results(result_rows)

    SUMMARY_PATH.write_text("", encoding="utf-8")
    summary.to_csv(SUMMARY_PATH, index=False)
    result_df.to_csv(ALL_RESULTS_PATH, index=False)

    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_PATH, index=False)

    print_summary_table(
        summary=summary.sort_values(
            ["score", "hits", "first_exact", "second_exact", "third_exact"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True),
        title="BEST BY SCORE",
        n=30,
    )

    print_summary_table(
        summary=summary.sort_values(
            ["hits", "score", "first_exact", "second_exact", "third_exact"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True),
        title="BEST BY HITS",
        n=30,
    )

    baseline = summary[summary["name"] == "baseline"].sort_values("score", ascending=False).head(1)
    best_by_score = summary.sort_values("score", ascending=False).head(1)
    best_by_hits = summary.sort_values(["hits", "score"], ascending=[False, False]).head(1)

    print()
    print("=" * 80)
    print("BEST CONFIGS")
    print("=" * 80)

    if not baseline.empty:
        b = baseline.iloc[0].to_dict()
        print(
            f"BASELINE: score={b['score']:.2f}/270 | "
            f"hits={b['hits']:.2f}/15 | "
            f"exact=({b['first_exact']:.2f}, {b['second_exact']:.2f}, {b['third_exact']:.2f})"
        )

    s = best_by_score.iloc[0].to_dict()
    print()
    print("BEST BY SCORE:")
    print(
        f"score={s['score']:.2f}/270 | hits={s['hits']:.2f}/15 | "
        f"exact=({s['first_exact']:.2f}, {s['second_exact']:.2f}, {s['third_exact']:.2f})"
    )
    print({k: v for k, v in s.items() if k not in {"score", "hits", "first_exact", "second_exact", "third_exact", "min_hits", "max_hits"}})

    h = best_by_hits.iloc[0].to_dict()
    print()
    print("BEST BY HITS:")
    print(
        f"score={h['score']:.2f}/270 | hits={h['hits']:.2f}/15 | "
        f"exact=({h['first_exact']:.2f}, {h['second_exact']:.2f}, {h['third_exact']:.2f})"
    )
    print({k: v for k, v in h.items() if k not in {"score", "hits", "first_exact", "second_exact", "third_exact", "min_hits", "max_hits"}})

    report = {
        "args": vars(args),
        "baseline": baseline.iloc[0].to_dict() if not baseline.empty else None,
        "best_by_score": best_by_score.iloc[0].to_dict(),
        "best_by_hits": best_by_hits.iloc[0].to_dict(),
        "paths": {
            "summary": str(SUMMARY_PATH),
            "predictions": str(PREDICTIONS_PATH),
            "all_results": str(ALL_RESULTS_PATH),
            "report": str(REPORT_PATH),
        },
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(f"summary:     {SUMMARY_PATH}")
    print(f"predictions: {PREDICTIONS_PATH}")
    print(f"all results: {ALL_RESULTS_PATH}")
    print(f"report:      {REPORT_PATH}")

    from unified_best_output import write_from_stage1_swap_report
    write_from_stage1_swap_report(
        report_path=REPORT_PATH,
        predictions_path=ALL_RESULTS_PATH,
        hgb_report_path=HGB_FULL_REPORT_PATH,
        out_dir=UNIFIED_OUTPUT_DIR,
        experiment="stage1_swap_selector",
    )


if __name__ == "__main__":
    main()
