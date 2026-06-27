from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import pandas as pd

from unified_experiment_utils import (
    add_classifier_scores,
    assign_prediction,
    get_feature_cols,
    load_award_dataset,
    make_hgb_classifier,
    make_sample_weight,
    parse_common_args,
    print_award_table,
    print_season_line,
    score_prediction,
    select_top_candidates,
    selected_awards,
    selection_column,
    summarize_seasons,
    unique_by_score,
    write_experiment_outputs,
)


def evaluate_award(spec, args):
    df = load_award_dataset(spec, feature_set=args.feature_set)
    feature_columns = get_feature_cols(df, args.feature_set)
    season_rows = []
    diagnostic_rows = []

    config = {
        "model": "HGB classifier diagnostic",
        "feature_set": args.feature_set,
        "weight_mode": args.weight_mode,
        "selection_score": args.selection_score,
        "diagnostic_pools": args.pool_ks,
    }

    for season in range(args.backtest_start, args.backtest_end + 1):
        train_df = df[(df["SEASON_END_YEAR"] >= args.min_train_season) & (df["SEASON_END_YEAR"] < season)].copy()
        test_df = df[df["SEASON_END_YEAR"] == season].copy()
        if train_df.empty or test_df.empty:
            continue

        model = make_hgb_classifier(args.max_iter, args.learning_rate, args.max_leaf_nodes, args.l2_regularization)
        y_train = train_df[spec.label_col].astype(int)
        weights = make_sample_weight(y_train, args.weight_mode, spec.max_label)
        model.fit(train_df[feature_columns], y_train, model__sample_weight=weights)

        scored = add_classifier_scores(test_df, model, feature_columns, spec, prefix="MODEL")
        score_col = selection_column(args.selection_score, prefix="MODEL")
        ranked = unique_by_score(scored, score_col=score_col, ascending=False)
        selected = ranked.head(spec.top_n).copy()
        prediction = assign_prediction(selected, spec=spec, assignment="sort", sort_col="MODEL_EXPECTED_LABEL")

        info = score_prediction(prediction, test_df, spec)
        true_keys = set(test_df.loc[test_df[spec.label_col] > 0, "PLAYER_NAME_KEY"])
        pool_hits = {}
        for k in args.pool_ks:
            pool_keys = set(ranked.head(k)["PLAYER_NAME_KEY"])
            pool_hits[f"top{k}_hits"] = len(pool_keys & true_keys)

        info.update({"experiment": args.experiment_name, "award": spec.name, "season": season, **pool_hits})
        season_rows.append(info)
        diagnostic_rows.append({"experiment": args.experiment_name, "award": spec.name, "season": season, **pool_hits})
        print_season_line(args.quiet, args.experiment_name, spec.name, season, info)

    season_df = pd.DataFrame(season_rows)
    diag_df = pd.DataFrame(diagnostic_rows)
    summary_df = summarize_seasons(season_df, args.experiment_name, spec.name, spec, config, len(feature_columns), args.output_dir)

    for k in args.pool_ks:
        col = f"top{k}_hits"
        if col in season_df.columns:
            summary_df[f"{col}_mean"] = float(season_df[col].mean())
            summary_df[f"{col}_min"] = int(season_df[col].min())
            summary_df[f"{col}_max"] = int(season_df[col].max())

    return summary_df, season_df, diag_df


def main() -> None:
    parser = parse_common_args(
        description="Unified missed-candidate and top-k recall diagnostic for All-NBA and All-Rookie.",
        default_experiment="missed_candidates_diagnostic",
    )
    parser.add_argument("--weight-mode", choices=["none", "positive_boost", "team_weighted", "sqrt_class_balance"], default="positive_boost")
    parser.add_argument("--selection-score", choices=["expected_label", "p_award", "best_team_value"], default="expected_label")
    parser.add_argument("--pool-ks", nargs="+", type=int, default=[20, 25, 30, 40])
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    args = parser.parse_args()

    summary_parts, season_parts, diag_parts = [], [], []
    for spec in selected_awards(args.award):
        summary_df, season_df, diag_df = evaluate_award(spec, args)
        summary_parts.append(summary_df)
        season_parts.append(season_df)
        diag_parts.append(diag_df)

    summary_path, season_path, _ = write_experiment_outputs(args.experiment_name, args.output_dir, summary_parts, season_parts, None)
    diag_path = args.output_dir / f"{args.experiment_name}_topk_recall.csv"
    pd.concat(diag_parts, ignore_index=True).to_csv(diag_path, index=False)

    print_award_table(pd.concat(summary_parts, ignore_index=True), quiet=args.quiet)
    if not args.quiet:
        print(f"\nSaved summary: {summary_path}")
        print(f"Saved seasons: {season_path}")
        print(f"Saved top-k recall: {diag_path}")


if __name__ == "__main__":
    main()
