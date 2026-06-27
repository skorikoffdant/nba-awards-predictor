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

from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from features import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_NBA_DATASET_PATH = PROCESSED_DIR / "all_nba_dataset.csv"
REPORT_PATH = MODELS_DIR / "stage1_ensemble_report.json"

MIN_TRAIN_SEASON = 2000
BACKTEST_START_SEASON = 2010
MAX_TRAIN_SEASON = 2025
RANDOM_STATE = 42
LABEL_COL = "ALL_NBA_LABEL"

ID_COLS = [
    "SEASON_END_YEAR",
    "PLAYER_NAME",
    "PLAYER_NAME_KEY",
    LABEL_COL,
]

HGB_SCORE_COLS = [
    "P_ALL_NBA",
    "EXPECTED_LABEL",
    "BEST_TEAM_VALUE",
]

POSITIVE_HEURISTIC_COLUMNS = [
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "W_PCT",
    "GP",
    "MIN",
    "TS_PCT",
    "USG_PCT",
    "PIE",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
    "DD2",
    "TD3",
    "PREV_ALL_NBA_LABEL",
    "PREV_ALL_NBA_VOTE_SCORE",
    "IS_ALL_STAR_THIS_SEASON",
    "PREV_ALL_STAR",
    "ALL_STAR_SELECTIONS_BEFORE_SEASON",
    "TEAM_TOTAL_PTS_SHARE",
    "TEAM_TOTAL_AST_SHARE",
    "TEAM_TOTAL_MIN_SHARE",
    "LEAGUE_TEAM_W_PCT_RANK",
]

BREF_HEURISTIC_COLUMNS = [
    "PER",
    "OWS",
    "DWS",
    "WS",
    "WS_PER_48",
    "OBPM",
    "DBPM",
    "BPM",
    "VORP",
    "PER_SEASON_RANK_PCT",
    "WS_SEASON_RANK_PCT",
    "BPM_SEASON_RANK_PCT",
    "VORP_SEASON_RANK_PCT",
]

LOWER_IS_BETTER_COLUMNS = {
    "LEAGUE_TEAM_W_PCT_RANK",
    "LEAGUE_TEAM_NET_RATING_RANK",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ALL_NBA_DATASET_PATH)

    for col in ID_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

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


def add_hgb_scores(
    df: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    scored = df.copy()
    proba = predict_proba_0123(model, scored[feature_columns])

    scored["P_NONE"] = proba[:, 0]
    scored["P_THIRD"] = proba[:, 1]
    scored["P_SECOND"] = proba[:, 2]
    scored["P_FIRST"] = proba[:, 3]
    scored["P_ALL_NBA"] = scored["P_THIRD"] + scored["P_SECOND"] + scored["P_FIRST"]
    scored["EXPECTED_LABEL"] = (
        1.0 * scored["P_THIRD"]
        + 2.0 * scored["P_SECOND"]
        + 3.0 * scored["P_FIRST"]
    )

    p1 = scored["P_THIRD"].to_numpy()
    p2 = scored["P_SECOND"].to_numpy()
    p3 = scored["P_FIRST"].to_numpy()

    scored["FIRST_VALUE"] = 10.0 * p3 + 8.0 * p2 + 6.0 * p1
    scored["SECOND_VALUE"] = 8.0 * p3 + 10.0 * p2 + 8.0 * p1
    scored["THIRD_VALUE"] = 6.0 * p3 + 8.0 * p2 + 10.0 * p1
    scored["BEST_TEAM_VALUE"] = scored[
        ["FIRST_VALUE", "SECOND_VALUE", "THIRD_VALUE"]
    ].max(axis=1)

    return scored


def rank_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median())

    if len(values) <= 1:
        return pd.Series(np.ones(len(values), dtype=float), index=series.index)

    ranks = values.rank(method="average", ascending=False)
    return 1.0 - (ranks - 1.0) / (len(values) - 1.0)


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median())

    lo = float(values.min())
    hi = float(values.max())

    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.ones(len(values), dtype=float), index=series.index)

    out = (values - lo) / (hi - lo)
    if not higher_is_better:
        out = 1.0 - out
    return out.astype(float)


def season_rank_score(df: pd.DataFrame, col: str) -> pd.Series:
    higher_is_better = col not in LOWER_IS_BETTER_COLUMNS
    return df.groupby("SEASON_END_YEAR", group_keys=False)[col].apply(
        lambda s: minmax_score(s, higher_is_better=higher_is_better)
    )


def add_heuristic_score(
    master: pd.DataFrame,
    source_df: pd.DataFrame,
    columns: list[str],
    out_col: str,
) -> pd.DataFrame:
    existing = [col for col in columns if col in source_df.columns]
    master = master.copy()

    if not existing:
        master[out_col] = np.nan
        return master

    parts = []
    for col in existing:
        parts.append(season_rank_score(source_df, col).reset_index(drop=True))

    score = pd.concat(parts, axis=1).mean(axis=1)
    master[out_col] = score.to_numpy(dtype=float)
    return master


def unique_keep_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def select_top_unique_players(df: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).copy()
    ranked = ranked.drop_duplicates(subset=["PLAYER_NAME_KEY"], keep="first")
    return ranked.head(top_n).copy()


def assign_by_sorted_score(selected_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    pred = selected_df.sort_values(score_col, ascending=False).copy().reset_index(drop=True)
    pred["PREDICTED_LABEL"] = 0
    pred.loc[0:4, "PREDICTED_LABEL"] = 3
    pred.loc[5:9, "PREDICTED_LABEL"] = 2
    pred.loc[10:14, "PREDICTED_LABEL"] = 1
    return pred


def assign_from_team_value_matrix(selected_df: pd.DataFrame, value_matrix: np.ndarray) -> pd.DataFrame:
    if len(selected_df) != 15:
        raise RuntimeError(f"Expected exactly 15 selected players, got {len(selected_df)}")

    value_matrix = np.asarray(value_matrix, dtype=float)
    if value_matrix.shape != (15, 3):
        raise RuntimeError(f"Expected value_matrix shape (15, 3), got {value_matrix.shape}")

    slot_values = np.concatenate(
        [
            np.repeat(value_matrix[:, [0]], 5, axis=1),
            np.repeat(value_matrix[:, [1]], 5, axis=1),
            np.repeat(value_matrix[:, [2]], 5, axis=1),
        ],
        axis=1,
    )

    row_ind, col_ind = linear_sum_assignment(-slot_values)

    pred = selected_df.copy().reset_index(drop=True)
    pred["PREDICTED_LABEL"] = 0

    for row, col in zip(row_ind, col_ind):
        if 0 <= col <= 4:
            label = 3
        elif 5 <= col <= 9:
            label = 2
        else:
            label = 1
        pred.loc[row, "PREDICTED_LABEL"] = label

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

    bonus_by_exact_count = {0: 0, 1: 0, 2: 5, 3: 10, 4: 20, 5: 40}
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

    first_keys = prediction_df.loc[prediction_df["PREDICTED_LABEL"] == 3, "PLAYER_NAME_KEY"].tolist()
    second_keys = prediction_df.loc[prediction_df["PREDICTED_LABEL"] == 2, "PLAYER_NAME_KEY"].tolist()
    third_keys = prediction_df.loc[prediction_df["PREDICTED_LABEL"] == 1, "PLAYER_NAME_KEY"].tolist()

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


def summarize_results(name: str, config: dict, season_results: list[dict]) -> dict:
    return {
        "name": name,
        "config": config,
        "avg_score": float(np.mean([row["score"] for row in season_results])),
        "avg_hits": float(np.mean([row["top_hits"] for row in season_results])),
        "avg_first_exact": float(np.mean([row["first_exact"] for row in season_results])),
        "avg_second_exact": float(np.mean([row["second_exact"] for row in season_results])),
        "avg_third_exact": float(np.mean([row["third_exact"] for row in season_results])),
        "season_results": season_results,
    }


def summarize_recall(name: str, config: dict, rows: list[dict]) -> dict:
    return {
        "name": name,
        "config": config,
        "avg_pool_size": float(np.mean([row["pool_size"] for row in rows])),
        "avg_hits": float(np.mean([row["hits"] for row in rows])),
        "min_hits": int(np.min([row["hits"] for row in rows])),
        "max_hits": int(np.max([row["hits"] for row in rows])),
        "season_results": rows,
    }


def make_hgb_component_name(feature_set: str, score_col: str) -> str:
    return f"hgb__{feature_set}__{score_col}"


def prepare_hgb_feature_sets(df: pd.DataFrame, requested_feature_sets: list[str]) -> dict[str, list[str]]:
    feature_map = {}
    for feature_set in requested_feature_sets:
        try:
            columns = get_feature_columns(df, feature_set=feature_set, verbose=True)
        except Exception as exc:
            print()
            print(f"WARNING: skip feature_set={feature_set}: {exc}")
            continue
        feature_map[feature_set] = columns

    if not feature_map:
        raise RuntimeError("No valid HGB feature sets available")

    return feature_map


def build_scored_season(
    df: pd.DataFrame,
    test_season: int,
    feature_map: dict[str, list[str]],
    args,
) -> tuple[pd.DataFrame, list[str]]:
    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] < test_season)
    ].copy()
    test_df = df[df["SEASON_END_YEAR"] == test_season].copy().reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise RuntimeError(f"Empty train/test split for season={test_season}")

    master = test_df[ID_COLS].copy()
    master["ROW_ID"] = np.arange(len(master))

    component_score_cols = []
    primary_values_added = False

    hgb_config = {
        "max_iter": args.hgb_max_iter,
        "learning_rate": args.hgb_learning_rate,
        "max_leaf_nodes": args.hgb_max_leaf_nodes,
        "l2_regularization": args.hgb_l2_regularization,
    }

    for feature_set, feature_columns in feature_map.items():
        y_train = train_df[LABEL_COL].astype(int)
        sample_weight = make_sample_weight(y_train, args.hgb_weight_mode)
        model = make_hgb_classifier(hgb_config)
        model.fit(train_df[feature_columns], y_train, model__sample_weight=sample_weight)

        scored = add_hgb_scores(test_df, model, feature_columns)

        for score_col in args.hgb_score_cols:
            if score_col not in scored.columns:
                continue
            out_col = make_hgb_component_name(feature_set, score_col)
            master[out_col] = scored[score_col].to_numpy(dtype=float)
            component_score_cols.append(out_col)

        # Used only for team assignment after a candidate set is chosen.
        if not primary_values_added and feature_set == args.primary_feature_set:
            master["PRIMARY_EXPECTED_LABEL"] = scored["EXPECTED_LABEL"].to_numpy(dtype=float)
            master["PRIMARY_FIRST_VALUE"] = scored["FIRST_VALUE"].to_numpy(dtype=float)
            master["PRIMARY_SECOND_VALUE"] = scored["SECOND_VALUE"].to_numpy(dtype=float)
            master["PRIMARY_THIRD_VALUE"] = scored["THIRD_VALUE"].to_numpy(dtype=float)
            primary_values_added = True

    if not primary_values_added:
        # Fallback to the first available feature set.
        first_feature_set = next(iter(feature_map.keys()))
        feature_columns = feature_map[first_feature_set]
        y_train = train_df[LABEL_COL].astype(int)
        sample_weight = make_sample_weight(y_train, args.hgb_weight_mode)
        model = make_hgb_classifier(hgb_config)
        model.fit(train_df[feature_columns], y_train, model__sample_weight=sample_weight)
        scored = add_hgb_scores(test_df, model, feature_columns)
        master["PRIMARY_EXPECTED_LABEL"] = scored["EXPECTED_LABEL"].to_numpy(dtype=float)
        master["PRIMARY_FIRST_VALUE"] = scored["FIRST_VALUE"].to_numpy(dtype=float)
        master["PRIMARY_SECOND_VALUE"] = scored["SECOND_VALUE"].to_numpy(dtype=float)
        master["PRIMARY_THIRD_VALUE"] = scored["THIRD_VALUE"].to_numpy(dtype=float)

    if args.include_stat_heuristic:
        out_col = "heuristic__stat"
        master = add_heuristic_score(master, test_df, POSITIVE_HEURISTIC_COLUMNS, out_col)
        if master[out_col].notna().any():
            component_score_cols.append(out_col)

    if args.include_bref_heuristic:
        out_col = "heuristic__bref"
        master = add_heuristic_score(master, test_df, BREF_HEURISTIC_COLUMNS, out_col)
        if master[out_col].notna().any():
            component_score_cols.append(out_col)

    component_score_cols = unique_keep_order(component_score_cols)
    return master, component_score_cols


def add_ensemble_scores(
    scored_df: pd.DataFrame,
    component_cols: list[str],
    main_col: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    df = scored_df.copy()
    valid_cols = [col for col in component_cols if col in df.columns and df[col].notna().any()]

    if not valid_cols:
        raise RuntimeError("No component score columns available for ensemble")

    rank_cols = []
    for col in valid_cols:
        rank_col = f"rank__{col}"
        df[rank_col] = rank_score(df[col])
        rank_cols.append(rank_col)

    ensemble_cols = []

    df["ens__mean_rank"] = df[rank_cols].mean(axis=1)
    ensemble_cols.append("ens__mean_rank")

    df["ens__max_rank"] = df[rank_cols].max(axis=1)
    ensemble_cols.append("ens__max_rank")

    if main_col is not None and main_col in df.columns:
        main_rank_col = f"rank__{main_col}"
        if main_rank_col in df.columns:
            other_rank_cols = [col for col in rank_cols if col != main_rank_col]
            if other_rank_cols:
                df["ens__main_050_rest_050"] = 0.50 * df[main_rank_col] + 0.50 * df[other_rank_cols].mean(axis=1)
                df["ens__main_070_rest_030"] = 0.70 * df[main_rank_col] + 0.30 * df[other_rank_cols].mean(axis=1)
                ensemble_cols += ["ens__main_050_rest_050", "ens__main_070_rest_030"]

    return df, ensemble_cols


def make_prediction_from_score(
    scored_df: pd.DataFrame,
    selection_score_col: str,
    assignment_mode: str,
) -> pd.DataFrame:
    selected = select_top_unique_players(scored_df, selection_score_col, top_n=15)

    if assignment_mode == "selection_score":
        return assign_by_sorted_score(selected, selection_score_col)

    if assignment_mode == "primary_expected_label":
        return assign_by_sorted_score(selected, "PRIMARY_EXPECTED_LABEL")

    if assignment_mode == "primary_assignment":
        values = selected[
            ["PRIMARY_FIRST_VALUE", "PRIMARY_SECOND_VALUE", "PRIMARY_THIRD_VALUE"]
        ].to_numpy(dtype=float)
        return assign_from_team_value_matrix(selected, values)

    raise ValueError(f"Unknown assignment_mode={assignment_mode}")


def evaluate_score_column(
    scored_by_season: dict[int, pd.DataFrame],
    df: pd.DataFrame,
    score_col: str,
    assignment_mode: str,
    config: dict,
) -> dict:
    season_results = []

    for season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        scored = scored_by_season[season]
        true_df = df[df["SEASON_END_YEAR"] == season].copy()
        pred = make_prediction_from_score(scored, score_col, assignment_mode)
        score_info = score_prediction(pred, true_df)
        season_results.append({"season": season, **score_info})

    return summarize_results(score_col, {**config, "assignment_mode": assignment_mode}, season_results)


def evaluate_oracle_recall(
    scored_by_season: dict[int, pd.DataFrame],
    df: pd.DataFrame,
    score_col: str,
    top_k: int,
    config: dict,
) -> dict:
    rows = []
    for season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        scored = scored_by_season[season]
        true_df = df[df["SEASON_END_YEAR"] == season].copy()
        true_keys = set(true_df.loc[true_df[LABEL_COL] > 0, "PLAYER_NAME_KEY"].tolist())
        pool = select_top_unique_players(scored, score_col, top_k)
        pool_keys = set(pool["PLAYER_NAME_KEY"].tolist())
        rows.append(
            {
                "season": season,
                "pool_size": int(len(pool_keys)),
                "hits": int(len(pool_keys & true_keys)),
            }
        )
    return summarize_recall(score_col, {**config, "top_k": top_k}, rows)


def evaluate_union_oracle(
    scored_by_season: dict[int, pd.DataFrame],
    df: pd.DataFrame,
    component_cols: list[str],
    per_component_top_k: int,
) -> dict:
    rows = []
    for season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        scored = scored_by_season[season]
        true_df = df[df["SEASON_END_YEAR"] == season].copy()
        true_keys = set(true_df.loc[true_df[LABEL_COL] > 0, "PLAYER_NAME_KEY"].tolist())

        pool_keys = set()
        for col in component_cols:
            if col not in scored.columns:
                continue
            top = select_top_unique_players(scored, col, per_component_top_k)
            pool_keys.update(top["PLAYER_NAME_KEY"].tolist())

        rows.append(
            {
                "season": season,
                "pool_size": int(len(pool_keys)),
                "hits": int(len(pool_keys & true_keys)),
            }
        )

    return summarize_recall(
        "union_components",
        {"per_component_top_k": per_component_top_k},
        rows,
    )


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


def print_stage1_summary(results: list[dict]) -> None:
    print()
    print("=" * 80)
    print("TOP-15 SELECTION + BASIC ASSIGNMENT SUMMARY")
    print("=" * 80)

    for result in sorted(results, key=lambda row: row["avg_score"], reverse=True):
        config = result["config"]
        print(
            f" {result['avg_score']:6.2f}/270 | "
            f"hits={result['avg_hits']:5.2f}/15 | "
            f"exact=({result['avg_first_exact']:.2f}, "
            f"{result['avg_second_exact']:.2f}, "
            f"{result['avg_third_exact']:.2f}) | "
            f"score_col={result['name']} | "
            f"assign={config.get('assignment_mode')}"
        )


def print_recall_summary(results: list[dict], title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for result in sorted(results, key=lambda row: (row["config"].get("top_k", 0), row["avg_hits"]), reverse=True):
        config = result["config"]
        top_k = config.get("top_k", config.get("per_component_top_k"))
        print(
            f" top_k={top_k:>2} | "
            f"hits={result['avg_hits']:5.2f}/15 | "
            f"min={result['min_hits']:>2}/15 | "
            f"max={result['max_hits']:>2}/15 | "
            f"pool_size={result['avg_pool_size']:5.2f} | "
            f"source={result['name']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hgb-feature-sets",
        nargs="+",
        default=["previous_team_share_allstar", "previous_team_share"],
    )
    parser.add_argument(
        "--hgb-score-cols",
        nargs="+",
        default=["EXPECTED_LABEL", "P_ALL_NBA", "BEST_TEAM_VALUE"],
        choices=HGB_SCORE_COLS,
    )
    parser.add_argument(
        "--primary-feature-set",
        default="previous_team_share_allstar",
        help="Model whose team values are used for assignment after ensemble selects top-15.",
    )
    parser.add_argument("--hgb-weight-mode", default="positive_boost", choices=["none", "positive_boost", "team_weighted", "sqrt_class_balance"])
    parser.add_argument("--hgb-max-iter", type=int, default=250)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2-regularization", type=float, default=0.05)

    parser.add_argument("--include-stat-heuristic", action="store_true")
    parser.add_argument("--include-bref-heuristic", action="store_true")

    parser.add_argument(
        "--assignment-modes",
        nargs="+",
        default=["primary_expected_label"],
        choices=["selection_score", "primary_expected_label", "primary_assignment"],
    )
    parser.add_argument("--top-ks", nargs="+", type=int, default=[15, 20, 25, 30, 40])
    parser.add_argument("--union-component-top-ks", nargs="+", type=int, default=[10, 15, 20])
    parser.add_argument("--max-components-in-summary", type=int, default=999)

    args = parser.parse_args()

    df = load_dataset()

    print()
    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"Dataset: {ALL_NBA_DATASET_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Seasons: {df['SEASON_END_YEAR'].min()} - {df['SEASON_END_YEAR'].max()}")
    print(df[LABEL_COL].value_counts().sort_index())

    requested_feature_sets = unique_keep_order([args.primary_feature_set] + args.hgb_feature_sets)
    feature_map = prepare_hgb_feature_sets(df, requested_feature_sets)

    scored_by_season: dict[int, pd.DataFrame] = {}
    all_component_cols: list[str] = []

    print()
    print("=" * 80)
    print("ROLLING STAGE-1 SCORING")
    print("=" * 80)

    for season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        scored, component_cols = build_scored_season(df, season, feature_map, args)
        all_component_cols = unique_keep_order(all_component_cols + component_cols)

        main_col = make_hgb_component_name(args.primary_feature_set, "EXPECTED_LABEL")
        if main_col not in scored.columns:
            main_col = component_cols[0]

        scored, ensemble_cols = add_ensemble_scores(scored, component_cols, main_col)
        scored_by_season[season] = scored

        true_hits_main = int(
            (
                select_top_unique_players(scored, main_col, 15)[LABEL_COL].astype(int) > 0
            ).sum()
        )
        true_hits_mean = int(
            (
                select_top_unique_players(scored, "ens__mean_rank", 15)[LABEL_COL].astype(int) > 0
            ).sum()
        )
        print(
            f"  {season}: components={len(component_cols):>2} | "
            f"main_top15_hits={true_hits_main:>2}/15 | "
            f"mean_ens_top15_hits={true_hits_mean:>2}/15",
            flush=True,
        )

    # Infer final available columns from the first season.
    first_scored = next(iter(scored_by_season.values()))
    component_cols = [col for col in all_component_cols if col in first_scored.columns]
    ensemble_cols = [col for col in first_scored.columns if col.startswith("ens__")]

    print()
    print("=" * 80)
    print("AVAILABLE COMPONENTS")
    print("=" * 80)
    for idx, col in enumerate(component_cols[: args.max_components_in_summary], start=1):
        print(f"  [{idx:02d}] {col}")
    if len(component_cols) > args.max_components_in_summary:
        print(f"  ... +{len(component_cols) - args.max_components_in_summary} more")

    print()
    print("Ensemble columns:")
    for col in ensemble_cols:
        print(f"  - {col}")

    selection_score_cols = unique_keep_order(component_cols + ensemble_cols)

    scoring_results = []
    for score_col, assignment_mode in itertools.product(selection_score_cols, args.assignment_modes):
        result = evaluate_score_column(
            scored_by_season=scored_by_season,
            df=df,
            score_col=score_col,
            assignment_mode=assignment_mode,
            config={"score_col": score_col},
        )
        scoring_results.append(result)

    oracle_results = []
    for score_col, top_k in itertools.product(selection_score_cols, args.top_ks):
        oracle_results.append(
            evaluate_oracle_recall(
                scored_by_season=scored_by_season,
                df=df,
                score_col=score_col,
                top_k=top_k,
                config={"score_col": score_col},
            )
        )

    union_results = []
    for per_component_top_k in args.union_component_top_ks:
        union_results.append(
            evaluate_union_oracle(
                scored_by_season=scored_by_season,
                df=df,
                component_cols=component_cols,
                per_component_top_k=per_component_top_k,
            )
        )

    print_stage1_summary(scoring_results)
    print_recall_summary(oracle_results, "ORACLE RECALL BY SCORE COLUMN")
    print_recall_summary(union_results, "UNION ORACLE RECALL")

    best_scoring = sorted(scoring_results, key=lambda row: row["avg_score"], reverse=True)[0]
    best_hits_top15 = sorted(
        [row for row in oracle_results if row["config"].get("top_k") == 15],
        key=lambda row: row["avg_hits"],
        reverse=True,
    )[0]

    print()
    print("=" * 80)
    print("BEST STAGE-1 SCORE")
    print("=" * 80)
    print(
        f"score={best_scoring['avg_score']:.2f}/270 | "
        f"hits={best_scoring['avg_hits']:.2f}/15 | "
        f"exact=({best_scoring['avg_first_exact']:.2f}, "
        f"{best_scoring['avg_second_exact']:.2f}, "
        f"{best_scoring['avg_third_exact']:.2f})"
    )
    print(best_scoring["config"])

    print()
    print("=" * 80)
    print("BEST TOP-15 ORACLE HITS")
    print("=" * 80)
    print(
        f"hits={best_hits_top15['avg_hits']:.2f}/15 | "
        f"min={best_hits_top15['min_hits']}/15 | "
        f"max={best_hits_top15['max_hits']}/15 | "
        f"source={best_hits_top15['name']}"
    )

    report = {
        "args": vars(args),
        "component_cols": component_cols,
        "ensemble_cols": ensemble_cols,
        "scoring_results": sorted(scoring_results, key=lambda row: row["avg_score"], reverse=True),
        "oracle_results": sorted(oracle_results, key=lambda row: (row["config"].get("top_k", 0), row["avg_hits"]), reverse=True),
        "union_results": sorted(union_results, key=lambda row: row["avg_hits"], reverse=True),
        "best_scoring": best_scoring,
        "best_hits_top15": best_hits_top15,
    }

    REPORT_PATH.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
