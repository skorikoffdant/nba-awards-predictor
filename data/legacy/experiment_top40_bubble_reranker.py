from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from legacy_experiment_common import (
    add_classifier_scores,
    apply_all_nba_eligibility,
    assign_by_rank,
    evaluate_base_hgb_for_season,
    fit_base_classifier,
    get_feature_columns,
    load_award_dataset,
    make_hgb_classifier,
    parse_legacy_args,
    prediction_to_rows,
    print_experiment_header,
    print_season_line,
    score_prediction,
    selected_awards,
    summarize_results,
    write_outputs,
)

EXPERIMENT = "top40_bubble_reranker"
DESCRIPTION = (
    "Guarded bubble reranker. For each award, the baseline HGB model creates a wider "
    "candidate pool, then a binary model may make one conservative swap near the top-N boundary."
)

CONFIG = {
    "pool_size_all_nba": 40,
    "pool_size_all_rookie": 25,
    "max_swaps": 1,
    "min_margin": 0.10,
    "bubble_max_iter": 160,
    "bubble_learning_rate": 0.04,
    "bubble_max_leaf_nodes": 15,
    "bubble_l2_regularization": 0.05,
}

CONTEXT_COLS = [
    "POOL_RANK", "POOL_RANK_PCT", "BASE_EXPECTED_LABEL", "BASE_P_AWARD",
    "GAP_TO_RANK_TOP_N", "GAP_TO_RANK_NEXT", "GAP_TO_PREV", "GAP_TO_NEXT_SCORE",
]


def pool_size(spec) -> int:
    return int(CONFIG["pool_size_all_nba"] if spec.name == "all_nba" else CONFIG["pool_size_all_rookie"])


def add_pool_context(pool: pd.DataFrame, spec) -> pd.DataFrame:
    parts = []
    for _, group in pool.groupby("SEASON_END_YEAR", sort=True):
        g = group.sort_values(["BASE_EXPECTED_LABEL", "PLAYER_NAME"], ascending=[False, True]).copy()
        g = g.reset_index(drop=True)
        n = len(g)
        g["POOL_RANK"] = np.arange(1, n + 1)
        g["POOL_RANK_PCT"] = g["POOL_RANK"] / max(n, 1)
        score_by_rank = dict(zip(g["POOL_RANK"], g["BASE_EXPECTED_LABEL"].astype(float)))
        s_top = score_by_rank.get(min(spec.top_n, n), np.nan)
        s_next = score_by_rank.get(min(spec.top_n + 1, n), np.nan)
        g["GAP_TO_RANK_TOP_N"] = g["BASE_EXPECTED_LABEL"].astype(float) - s_top
        g["GAP_TO_RANK_NEXT"] = g["BASE_EXPECTED_LABEL"].astype(float) - s_next
        scores = g["BASE_EXPECTED_LABEL"].astype(float).to_numpy()
        g["GAP_TO_PREV"] = np.r_[0.0, scores[1:] - scores[:-1]] if len(scores) else 0.0
        g["GAP_TO_NEXT_SCORE"] = np.r_[scores[:-1] - scores[1:], 0.0] if len(scores) else 0.0
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pool.copy()


def make_top_pool(scored: pd.DataFrame, spec) -> pd.DataFrame:
    pool = scored.sort_values(["BASE_EXPECTED_LABEL", "PLAYER_NAME"], ascending=[False, True])
    pool = pool.drop_duplicates("PLAYER_NAME_KEY").head(pool_size(spec)).copy()
    return add_pool_context(pool, spec)


def build_training_pool(train_df: pd.DataFrame, feature_cols: list[str], spec, args) -> pd.DataFrame:
    base_model = fit_base_classifier(train_df, feature_cols, spec, args)
    scored = add_classifier_scores(train_df, base_model, feature_cols, spec, prefix="BASE")
    pools = []
    for _, group in scored.groupby("SEASON_END_YEAR", sort=True):
        pools.append(make_top_pool(group, spec))
    if not pools:
        return pd.DataFrame()
    pool = pd.concat(pools, ignore_index=True)
    pool["IS_TRUE_AWARD"] = (pool[spec.label_col].fillna(0).astype(int) > 0).astype(int)
    return pool


def apply_one_guarded_swap(pool: pd.DataFrame, spec, proba_col: str) -> pd.DataFrame:
    pool = pool.sort_values("POOL_RANK").copy()
    selected = pool[pool["POOL_RANK"] <= spec.top_n].copy()

    remove_start = max(1, spec.top_n - 3)
    add_start = spec.top_n + 1
    add_end = min(pool_size(spec), spec.top_n + 10)
    remove_df = selected[selected["POOL_RANK"] >= remove_start].copy()
    add_df = pool[(pool["POOL_RANK"] >= add_start) & (pool["POOL_RANK"] <= add_end)].copy()

    selected_keys = set(selected["PLAYER_NAME_KEY"])
    best = None
    for _, remove_row in remove_df.iterrows():
        for _, add_row in add_df.iterrows():
            if add_row["PLAYER_NAME_KEY"] in selected_keys:
                continue
            margin = float(add_row[proba_col] - remove_row[proba_col])
            if margin < CONFIG["min_margin"]:
                continue
            penalty = max(0.0, float(remove_row["BASE_EXPECTED_LABEL"] - add_row["BASE_EXPECTED_LABEL"])) * 0.02
            decision_score = margin - penalty
            candidate = (decision_score, remove_row["PLAYER_NAME_KEY"], add_row["PLAYER_NAME_KEY"])
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return selected
    _, remove_key, add_key = best
    out = pool[((pool["POOL_RANK"] <= spec.top_n) & (pool["PLAYER_NAME_KEY"] != remove_key)) | (pool["PLAYER_NAME_KEY"] == add_key)].copy()
    return out


def evaluate_award(spec, args):
    df = load_award_dataset(spec, feature_set=args.feature_set)
    feature_cols = get_feature_columns(df, args.feature_set)
    bubble_cols = feature_cols + CONTEXT_COLS

    season_rows = []
    prediction_rows = [] if args.save_predictions else None

    for season in range(args.backtest_start, args.backtest_end + 1):
        train_df = df[(df["SEASON_END_YEAR"] >= args.min_train_season) & (df["SEASON_END_YEAR"] < season)].copy()
        test_df = df[df["SEASON_END_YEAR"] == season].copy()
        if train_df.empty or test_df.empty:
            continue
        if spec.name == "all_nba":
            test_df = apply_all_nba_eligibility(test_df)
        _, base_info = evaluate_base_hgb_for_season(df, feature_cols, spec, args, season)

        train_pool = build_training_pool(train_df, feature_cols, spec, args)
        if train_pool.empty or train_pool["IS_TRUE_AWARD"].nunique() < 2:
            continue
        for col in CONTEXT_COLS:
            if col not in train_pool.columns:
                train_pool[col] = 0.0

        bubble_model = make_hgb_classifier(CONFIG["bubble_max_iter"], CONFIG["bubble_learning_rate"], CONFIG["bubble_max_leaf_nodes"], CONFIG["bubble_l2_regularization"])
        bubble_model.fit(train_pool[bubble_cols], train_pool["IS_TRUE_AWARD"].astype(int))

        base_model = fit_base_classifier(train_df, feature_cols, spec, args)
        scored = add_classifier_scores(test_df, base_model, feature_cols, spec, prefix="BASE")
        pool = make_top_pool(scored, spec)
        for col in bubble_cols:
            if col not in pool.columns:
                pool[col] = 0.0

        proba = bubble_model.predict_proba(pool[bubble_cols])
        classes = list(bubble_model.named_steps["model"].classes_)
        pool["BUBBLE_PROBA"] = proba[:, classes.index(1)] if 1 in classes else 0.0

        selected = apply_one_guarded_swap(pool, spec, "BUBBLE_PROBA")
        prediction = assign_by_rank(selected, spec, order_col="BASE_EXPECTED_LABEL")
        info = score_prediction(prediction, test_df, spec)
        info.update({"experiment": args.experiment_name, "award": spec.name, "season": season, "base_score": base_info["score"]})
        season_rows.append(info)
        print_season_line(args, args.experiment_name, spec.name, season, info, base_info, spec)

        if prediction_rows is not None:
            prediction_rows.extend(prediction_to_rows(prediction, season, args.experiment_name, spec.name, ["BASE_EXPECTED_LABEL", "BUBBLE_PROBA"]))

    season_df = pd.DataFrame(season_rows)
    summary_df = summarize_results(season_df, args.experiment_name, spec.name, spec, CONFIG, len(bubble_cols))
    return summary_df, season_df, prediction_rows


def main():
    parser = parse_legacy_args(DESCRIPTION, EXPERIMENT)
    args = parser.parse_args()
    print_experiment_header(args.experiment_name, DESCRIPTION, args, CONFIG)

    summary_parts, season_parts = [], []
    all_prediction_rows = [] if args.save_predictions else None
    for spec in selected_awards(args.award):
        summary_df, season_df, prediction_rows = evaluate_award(spec, args)
        summary_parts.append(summary_df)
        season_parts.append(season_df)
        if all_prediction_rows is not None and prediction_rows is not None:
            all_prediction_rows.extend(prediction_rows)

    write_outputs(args.experiment_name, args.output_dir, summary_parts, season_parts, all_prediction_rows, args.quiet)


if __name__ == "__main__":
    main()
