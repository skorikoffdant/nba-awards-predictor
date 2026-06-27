from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from pathlib import Path

import numpy as np
import pandas as pd

from legacy_experiment_common import (
    AWARDS,
    apply_all_nba_eligibility,
    add_classifier_scores,
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
    safe_minmax,
    score_prediction,
    select_top_candidates,
    selected_awards,
    summarize_results,
    write_outputs,
)


EXPERIMENT = "top40_vote_support_reranker"
DESCRIPTION = (
    "Top-k reranker with a vote-support regressor. The baseline HGB ranking creates "
    "a candidate pool, then a regressor estimates voting support and the final score "
    "is a blend of the base score and the vote-support score."
)

CONFIG = {
    "pool_size_all_nba": 40,
    "pool_size_all_rookie": 25,
    "blend": 0.35,
    "vote_max_iter": 180,
    "vote_learning_rate": 0.035,
    "vote_max_leaf_nodes": 15,
    "vote_l2_regularization": 0.05,
}


def vote_dataset_path(spec):
    if spec.name == "all_nba":
        return Path("data/processed/all_nba_dataset_with_votes.csv")
    return Path("data/processed/all_rookie_dataset_with_votes.csv")


def load_dataset_with_optional_votes(spec, feature_set):
    candidate = vote_dataset_path(spec)
    if candidate.exists():
        return load_award_dataset(spec, feature_set=feature_set, dataset_path=candidate), True
    return load_award_dataset(spec, feature_set=feature_set), False


def make_vote_target(df, spec, votes_available):
    if votes_available:
        prefix = "ALL_NBA" if spec.name == "all_nba" else "ALL_ROOKIE"
        for col in [
            f"{prefix}_VOTE_SHARE",
            f"{prefix}_VOTE_SCORE_SEASON_MAX_NORM",
            f"{prefix}_VOTE_SCORE",
        ]:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0.0), col

    return label_vote_proxy(df[spec.label_col], spec), "label_vote_proxy"


def evaluate_award(spec, args):
    df, votes_available = load_dataset_with_optional_votes(spec, args.feature_set)
    feature_cols = get_feature_columns(df, args.feature_set)
    target, target_name = make_vote_target(df, spec, votes_available)
    df = df.copy()
    df["__VOTE_TARGET__"] = target

    pool_size = CONFIG["pool_size_all_nba"] if spec.name == "all_nba" else CONFIG["pool_size_all_rookie"]
    season_rows = []
    prediction_rows = [] if args.save_predictions else None

    for season in range(args.backtest_start, args.backtest_end + 1):
        train_df = df[(df["SEASON_END_YEAR"] >= args.min_train_season) & (df["SEASON_END_YEAR"] < season)].copy()
        test_df = df[df["SEASON_END_YEAR"] == season].copy()

        if train_df.empty or test_df.empty:
            continue

        if spec.name == "all_nba":
            test_df = apply_all_nba_eligibility(test_df)

        base_model = fit_base_classifier(train_df, feature_cols, spec, args)
        vote_model = make_hgb_regressor(
            CONFIG["vote_max_iter"],
            CONFIG["vote_learning_rate"],
            CONFIG["vote_max_leaf_nodes"],
            CONFIG["vote_l2_regularization"],
        )
        vote_model.fit(train_df[feature_cols], train_df["__VOTE_TARGET__"].astype(float))

        scored = add_classifier_scores(test_df, base_model, feature_cols, spec, prefix="BASE")
        scored["VOTE_SUPPORT_SCORE"] = vote_model.predict(scored[feature_cols])

        base_selected = select_top_candidates(scored, spec, score_col="BASE_EXPECTED_LABEL")
        base_prediction = assign_by_rank(base_selected, spec, order_col="BASE_EXPECTED_LABEL")
        base_info = score_prediction(base_prediction, test_df, spec)

        pool = scored.sort_values(["BASE_EXPECTED_LABEL", "PLAYER_NAME"], ascending=[False, True])
        pool = pool.drop_duplicates("PLAYER_NAME_KEY").head(pool_size).copy()
        pool["BASE_NORM"] = safe_minmax(pool["BASE_EXPECTED_LABEL"])
        pool["VOTE_SUPPORT_NORM"] = safe_minmax(pool["VOTE_SUPPORT_SCORE"])
        pool["FINAL_SCORE"] = (
            (1.0 - CONFIG["blend"]) * pool["BASE_NORM"]
            + CONFIG["blend"] * pool["VOTE_SUPPORT_NORM"]
        )

        selected = select_top_candidates(pool, spec, score_col="FINAL_SCORE")
        prediction = assign_by_rank(selected, spec, order_col="BASE_EXPECTED_LABEL")
        info = score_prediction(prediction, test_df, spec)
        info.update({
            "experiment": args.experiment_name,
            "award": spec.name,
            "season": season,
            "base_score": base_info["score"],
        })
        season_rows.append(info)
        print_season_line(args, args.experiment_name, spec.name, season, info, base_info, spec)

        if prediction_rows is not None:
            prediction_rows.extend(
                prediction_to_rows(
                    prediction,
                    season,
                    args.experiment_name,
                    spec.name,
                    ["BASE_EXPECTED_LABEL", "VOTE_SUPPORT_SCORE", "FINAL_SCORE"],
                )
            )

    season_df = pd.DataFrame(season_rows)
    summary_df = summarize_results(
        season_df,
        args.experiment_name,
        spec.name,
        spec,
        {**CONFIG, "vote_target": target_name, "votes_available": votes_available},
        len(feature_cols),
    )
    return summary_df, season_df, prediction_rows


def main():
    parser = parse_legacy_args(DESCRIPTION, EXPERIMENT)
    args = parser.parse_args()
    print_experiment_header(args.experiment_name, DESCRIPTION, args, CONFIG)

    summary_parts = []
    season_parts = []
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
