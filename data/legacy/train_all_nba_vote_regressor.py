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
    apply_all_nba_eligibility,
    assign_by_rank,
    evaluate_base_hgb_for_season,
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

EXPERIMENT = "vote_regressor"
DESCRIPTION = (
    "Vote-score/vote-share regressor. For each award, the model tries to predict "
    "voting support directly instead of predicting the discrete team label."
)

CONFIG = {
    "all_nba_dataset_with_votes": "data/processed/all_nba_dataset_with_votes.csv",
    "all_rookie_dataset_with_votes": "data/processed/all_rookie_dataset_with_votes.csv",
    "regressor_max_iter": 180,
    "regressor_learning_rate": 0.035,
    "regressor_max_leaf_nodes": 15,
    "regressor_l2_regularization": 0.05,
}


def vote_dataset_path(spec) -> Path:
    key = "all_nba_dataset_with_votes" if spec.name == "all_nba" else "all_rookie_dataset_with_votes"
    return Path(CONFIG[key])


def load_vote_dataset(spec, feature_set):
    path = vote_dataset_path(spec)
    if path.exists():
        return load_award_dataset(spec, feature_set=feature_set, dataset_path=path), True
    return load_award_dataset(spec, feature_set=feature_set), False


def make_target(df, spec, votes_available):
    if votes_available:
        prefix = "ALL_NBA" if spec.name == "all_nba" else "ALL_ROOKIE"
        for col in [f"{prefix}_VOTE_SHARE", f"{prefix}_VOTE_SCORE_SEASON_MAX_NORM", f"{prefix}_VOTE_SCORE"]:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0.0), col
    return label_vote_proxy(df[spec.label_col], spec), "label_vote_proxy"


def evaluate_award(spec, args):
    df, votes_available = load_vote_dataset(spec, args.feature_set)
    feature_cols = get_feature_columns(df, args.feature_set)
    target, target_name = make_target(df, spec, votes_available)
    df = df.copy()
    df["__VOTE_TARGET__"] = target
    config = dict(CONFIG)
    config.update({"votes_available": votes_available, "target": target_name})

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

        model = make_hgb_regressor(CONFIG["regressor_max_iter"], CONFIG["regressor_learning_rate"], CONFIG["regressor_max_leaf_nodes"], CONFIG["regressor_l2_regularization"])
        model.fit(train_df[feature_cols], train_df["__VOTE_TARGET__"].astype(float))

        scored = test_df.copy()
        scored["VOTE_REG_SCORE"] = model.predict(scored[feature_cols])
        selected = select_top_candidates(scored, spec, score_col="VOTE_REG_SCORE")
        prediction = assign_by_rank(selected, spec, order_col="VOTE_REG_SCORE")
        info = score_prediction(prediction, test_df, spec)
        info.update({"experiment": args.experiment_name, "award": spec.name, "season": season, "base_score": base_info["score"]})
        season_rows.append(info)
        print_season_line(args, args.experiment_name, spec.name, season, info, base_info, spec)

        if prediction_rows is not None:
            prediction_rows.extend(prediction_to_rows(prediction, season, args.experiment_name, spec.name, ["VOTE_REG_SCORE"]))

    season_df = pd.DataFrame(season_rows)
    summary_df = summarize_results(season_df, args.experiment_name, spec.name, spec, config, len(feature_cols))
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
