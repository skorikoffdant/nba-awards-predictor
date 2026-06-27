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
    label_vote_proxy,
    load_award_dataset,
    make_hgb_regressor,
    parse_legacy_args,
    prediction_to_rows,
    print_experiment_header,
    print_season_line,
    score_prediction,
    select_top_candidates,
    selected_awards,
    summarize_results,
    write_outputs,
)

EXPERIMENT = "candidate_pool_stage2"
DESCRIPTION = (
    "Candidate-pool Stage 2. For each award, a baseline HGB model creates a wider "
    "candidate pool, then a second regressor selects the final top-N list."
)

CONFIG = {
    "candidate_top_k_all_nba": 30,
    "candidate_top_k_all_rookie": 20,
    "stage2_target": "label_vote_proxy",
    "stage2_max_iter": 180,
    "stage2_learning_rate": 0.035,
    "stage2_max_leaf_nodes": 15,
    "stage2_l2_regularization": 0.05,
}

BASE_CONTEXT_COLS = [
    "POOL_RANK",
    "POOL_RANK_PCT",
    "BASE_EXPECTED_LABEL",
    "BASE_P_AWARD",
    "GAP_TO_RANK_1",
    "GAP_TO_RANK_5",
    "GAP_TO_RANK_10",
    "GAP_TO_PREV",
    "GAP_TO_NEXT",
]


def candidate_top_k(spec) -> int:
    return int(CONFIG["candidate_top_k_all_nba"] if spec.name == "all_nba" else CONFIG["candidate_top_k_all_rookie"])


def context_cols_for_spec(spec) -> list[str]:
    cols = list(BASE_CONTEXT_COLS)
    cols.append(f"GAP_TO_RANK_{spec.top_n}")
    cols.append(f"GAP_TO_RANK_{candidate_top_k(spec)}")
    for label in range(1, spec.max_label + 1):
        cols.append(f"BASE_P_LABEL_{label}")
        cols.append(f"BASE_TEAM_VALUE_{label}")
    seen = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def add_candidate_context(pool: pd.DataFrame, spec) -> pd.DataFrame:
    parts = []
    gap_ranks = sorted({1, 5, 10, spec.top_n, candidate_top_k(spec)})
    for _, group in pool.groupby("SEASON_END_YEAR", sort=True):
        g = group.sort_values(["BASE_EXPECTED_LABEL", "PLAYER_NAME"], ascending=[False, True]).copy()
        g = g.reset_index(drop=True)
        n = len(g)
        g["POOL_RANK"] = np.arange(1, n + 1)
        g["POOL_RANK_PCT"] = g["POOL_RANK"] / max(n, 1)
        score_by_rank = dict(zip(g["POOL_RANK"], g["BASE_EXPECTED_LABEL"].astype(float)))
        for rank in gap_ranks:
            anchor = score_by_rank.get(min(rank, n), np.nan)
            g[f"GAP_TO_RANK_{rank}"] = g["BASE_EXPECTED_LABEL"].astype(float) - anchor
        scores = g["BASE_EXPECTED_LABEL"].astype(float).to_numpy()
        g["GAP_TO_PREV"] = np.r_[0.0, scores[1:] - scores[:-1]] if len(scores) else 0.0
        g["GAP_TO_NEXT"] = np.r_[scores[:-1] - scores[1:], 0.0] if len(scores) else 0.0
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pool.copy()


def build_stage2_training_pool(train_df: pd.DataFrame, feature_cols: list[str], spec, args) -> pd.DataFrame:
    base_model = fit_base_classifier(train_df, feature_cols, spec, args)
    scored = add_classifier_scores(train_df, base_model, feature_cols, spec, prefix="BASE")
    pools = []
    for _, group in scored.groupby("SEASON_END_YEAR", sort=True):
        pool = group.sort_values(["BASE_EXPECTED_LABEL", "PLAYER_NAME"], ascending=[False, True])
        pool = pool.drop_duplicates("PLAYER_NAME_KEY").head(candidate_top_k(spec)).copy()
        pools.append(pool)
    if not pools:
        return pd.DataFrame()
    return add_candidate_context(pd.concat(pools, ignore_index=True), spec)


def evaluate_award(spec, args):
    df = load_award_dataset(spec, feature_set=args.feature_set)
    feature_cols = get_feature_columns(df, args.feature_set)
    context_cols = context_cols_for_spec(spec)
    stage2_cols = feature_cols + context_cols

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

        train_pool = build_stage2_training_pool(train_df, feature_cols, spec, args)
        if train_pool.empty:
            continue

        for col in context_cols:
            if col not in train_pool.columns:
                train_pool[col] = 0.0

        y_stage2 = label_vote_proxy(train_pool[spec.label_col], spec)
        stage2_model = make_hgb_regressor(
            CONFIG["stage2_max_iter"], CONFIG["stage2_learning_rate"], CONFIG["stage2_max_leaf_nodes"], CONFIG["stage2_l2_regularization"]
        )
        stage2_model.fit(train_pool[stage2_cols], y_stage2)

        base_model = fit_base_classifier(train_df, feature_cols, spec, args)
        scored = add_classifier_scores(test_df, base_model, feature_cols, spec, prefix="BASE")
        pool = scored.sort_values(["BASE_EXPECTED_LABEL", "PLAYER_NAME"], ascending=[False, True])
        pool = pool.drop_duplicates("PLAYER_NAME_KEY").head(candidate_top_k(spec)).copy()
        pool = add_candidate_context(pool, spec)
        for col in stage2_cols:
            if col not in pool.columns:
                pool[col] = 0.0

        pool["STAGE2_SCORE"] = stage2_model.predict(pool[stage2_cols])
        selected = select_top_candidates(pool, spec, score_col="STAGE2_SCORE")
        prediction = assign_by_rank(selected, spec, order_col="STAGE2_SCORE")
        info = score_prediction(prediction, test_df, spec)
        info.update({"experiment": args.experiment_name, "award": spec.name, "season": season, "base_score": base_info["score"]})
        season_rows.append(info)
        print_season_line(args, args.experiment_name, spec.name, season, info, base_info, spec)

        if prediction_rows is not None:
            prediction_rows.extend(prediction_to_rows(prediction, season, args.experiment_name, spec.name, ["BASE_EXPECTED_LABEL", "STAGE2_SCORE"]))

    season_df = pd.DataFrame(season_rows)
    summary_df = summarize_results(season_df, args.experiment_name, spec.name, spec, CONFIG, len(stage2_cols))
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
