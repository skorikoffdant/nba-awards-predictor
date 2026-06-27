from __future__ import annotations

from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import numpy as np
import pandas as pd

from legacy_experiment_common import (
    AWARDS,
    add_classifier_scores,
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
    select_top_candidates,
    summarize_results,
    write_outputs,
)


EXPERIMENT = "all_rookie_reorderer"
DESCRIPTION = (
    "All-Rookie fixed top-10 reorderer. The baseline HGB classifier chooses ten "
    "rookies, then a second model tries to reorder the same players between "
    "First and Second Rookie Team."
)

CONFIG = {
    "blend": 0.20,
    "reorderer_max_iter": 120,
    "reorderer_learning_rate": 0.04,
    "reorderer_max_leaf_nodes": 7,
    "reorderer_l2_regularization": 0.05,
}


META_COLS = [
    "BASE_RANK",
    "BASE_RANK_PCT",
    "BASE_EXPECTED_LABEL",
    "BASE_P_AWARD",
    "BASE_P_LABEL_1",
    "BASE_P_LABEL_2",
    "BASE_TEAM_VALUE_1",
    "BASE_TEAM_VALUE_2",
    "BASE_GAP_TO_RANK_5",
    "BASE_GAP_TO_RANK_10",
    "BASE_GAP_PREV",
    "BASE_GAP_NEXT",
]


def add_reorder_context(selected: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in selected.groupby("SEASON_END_YEAR", sort=True):
        g = group.sort_values(["BASE_EXPECTED_LABEL", "PLAYER_NAME"], ascending=[False, True]).copy()
        g = g.reset_index(drop=True)
        n = len(g)
        g["BASE_RANK"] = np.arange(1, n + 1)
        g["BASE_RANK_PCT"] = g["BASE_RANK"] / max(n, 1)

        score_by_rank = dict(zip(g["BASE_RANK"], g["BASE_EXPECTED_LABEL"].astype(float)))
        s5 = score_by_rank.get(min(5, n), np.nan)
        s10 = score_by_rank.get(min(10, n), np.nan)
        g["BASE_GAP_TO_RANK_5"] = g["BASE_EXPECTED_LABEL"].astype(float) - s5
        g["BASE_GAP_TO_RANK_10"] = g["BASE_EXPECTED_LABEL"].astype(float) - s10

        scores = g["BASE_EXPECTED_LABEL"].astype(float).to_numpy()
        g["BASE_GAP_PREV"] = np.r_[0.0, scores[1:] - scores[:-1]] if len(scores) else 0.0
        g["BASE_GAP_NEXT"] = np.r_[scores[:-1] - scores[1:], 0.0] if len(scores) else 0.0
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else selected.copy()


def build_reorderer_training_rows(train_df: pd.DataFrame, feature_cols: list[str], spec, args) -> pd.DataFrame:
    base_model = fit_base_classifier(train_df, feature_cols, spec, args)
    scored = add_classifier_scores(train_df, base_model, feature_cols, spec, prefix="BASE")
    selected_parts = []
    for _, group in scored.groupby("SEASON_END_YEAR", sort=True):
        selected = select_top_candidates(group, spec, score_col="BASE_EXPECTED_LABEL")
        selected_parts.append(selected)
    if not selected_parts:
        return pd.DataFrame()
    return add_reorder_context(pd.concat(selected_parts, ignore_index=True))


def reorder_score_columns(pool: pd.DataFrame, reorderer_model, reorderer_cols: list[str]) -> pd.DataFrame:
    out = pool.copy()
    raw = reorderer_model.predict_proba(out[reorderer_cols])
    classes = list(reorderer_model.named_steps["model"].classes_)

    p1 = np.zeros(len(out), dtype=float)
    p2 = np.zeros(len(out), dtype=float)

    for idx, cls in enumerate(classes):
        if int(cls) == 1:
            p1 = raw[:, idx]
        elif int(cls) == 2:
            p2 = raw[:, idx]

    out["REORDERER_EXPECTED_LABEL"] = 1.0 * p1 + 2.0 * p2
    out["FINAL_REORDER_SCORE"] = (
        (1.0 - CONFIG["blend"]) * out["BASE_EXPECTED_LABEL"].astype(float)
        + CONFIG["blend"] * out["REORDERER_EXPECTED_LABEL"].astype(float)
    )
    return out


def evaluate(args):
    spec = AWARDS["all_rookie"]
    df = load_award_dataset(spec, feature_set=args.feature_set)
    feature_cols = get_feature_columns(df, args.feature_set)
    reorderer_cols = feature_cols + META_COLS

    season_rows = []
    prediction_rows = [] if args.save_predictions else None

    for season in range(args.backtest_start, args.backtest_end + 1):
        train_df = df[(df["SEASON_END_YEAR"] >= args.min_train_season) & (df["SEASON_END_YEAR"] < season)].copy()
        test_df = df[df["SEASON_END_YEAR"] == season].copy()
        if train_df.empty or test_df.empty:
            continue

        _, base_info = evaluate_base_hgb_for_season(df, feature_cols, spec, args, season)
        train_pool = build_reorderer_training_rows(train_df, feature_cols, spec, args)
        if train_pool.empty:
            continue

        for col in META_COLS:
            if col not in train_pool.columns:
                train_pool[col] = 0.0

        reorderer = make_hgb_classifier(
            CONFIG["reorderer_max_iter"],
            CONFIG["reorderer_learning_rate"],
            CONFIG["reorderer_max_leaf_nodes"],
            CONFIG["reorderer_l2_regularization"],
        )
        y = train_pool[spec.label_col].clip(lower=1).astype(int)
        reorderer.fit(train_pool[reorderer_cols], y)

        base_model = fit_base_classifier(train_df, feature_cols, spec, args)
        scored = add_classifier_scores(test_df, base_model, feature_cols, spec, prefix="BASE")
        selected = select_top_candidates(scored, spec, score_col="BASE_EXPECTED_LABEL")
        selected = add_reorder_context(selected)

        for col in reorderer_cols:
            if col not in selected.columns:
                selected[col] = 0.0

        selected = reorder_score_columns(selected, reorderer, reorderer_cols)
        prediction = assign_by_rank(selected, spec, order_col="FINAL_REORDER_SCORE")
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
                prediction_to_rows(prediction, season, args.experiment_name, spec.name, ["BASE_EXPECTED_LABEL", "REORDERER_EXPECTED_LABEL", "FINAL_REORDER_SCORE"])
            )

    season_df = pd.DataFrame(season_rows)
    summary_df = summarize_results(
        season_df,
        args.experiment_name,
        spec.name,
        spec,
        CONFIG,
        len(reorderer_cols),
    )
    return summary_df, season_df, prediction_rows


def prediction_to_json_block(prediction: pd.DataFrame, spec) -> dict:
    """Convert prediction rows with labels into JSON-style teams."""
    result = {name: [] for name in spec.team_names}
    label_to_name = {
        spec.max_label: "first",
        spec.max_label - 1: "second",
    }
    if spec.max_label == 3:
        label_to_name[1] = "third"

    for _, row in prediction.iterrows():
        label = int(row.get("PREDICTED_LABEL", 0))
        team = label_to_name.get(label)
        if team is not None:
            result[team].append(str(row["PLAYER_NAME"]))
    return result


def predict_current(args) -> pd.DataFrame:
    """Train on historical seasons and print/save current-season All-Rookie reorderer prediction."""
    spec = AWARDS["all_rookie"]
    season = int(args.predict_season)
    df = load_award_dataset(spec, feature_set=args.feature_set)
    feature_cols = get_feature_columns(df, args.feature_set)
    reorderer_cols = feature_cols + META_COLS

    train_df = df[(df["SEASON_END_YEAR"] >= args.min_train_season) & (df["SEASON_END_YEAR"] < season)].copy()
    current_df = df[df["SEASON_END_YEAR"] == season].copy()

    if train_df.empty:
        raise RuntimeError(f"No training data before season {season}.")
    if current_df.empty:
        raise RuntimeError(f"No current data for season {season}.")

    train_pool = build_reorderer_training_rows(train_df, feature_cols, spec, args)
    if train_pool.empty:
        raise RuntimeError("Could not build reorderer training pool.")

    for col in META_COLS:
        if col not in train_pool.columns:
            train_pool[col] = 0.0

    reorderer = make_hgb_classifier(
        CONFIG["reorderer_max_iter"],
        CONFIG["reorderer_learning_rate"],
        CONFIG["reorderer_max_leaf_nodes"],
        CONFIG["reorderer_l2_regularization"],
    )
    y = train_pool[spec.label_col].clip(lower=1).astype(int)
    reorderer.fit(train_pool[reorderer_cols], y)

    base_model = fit_base_classifier(train_df, feature_cols, spec, args)
    scored = add_classifier_scores(current_df, base_model, feature_cols, spec, prefix="BASE")
    selected = select_top_candidates(scored, spec, score_col="BASE_EXPECTED_LABEL")
    selected = add_reorder_context(selected)

    for col in reorderer_cols:
        if col not in selected.columns:
            selected[col] = 0.0

    selected = reorder_score_columns(selected, reorderer, reorderer_cols)
    prediction = assign_by_rank(selected, spec, order_col="FINAL_REORDER_SCORE")
    prediction = prediction.reset_index(drop=True)
    prediction["RANK"] = range(1, len(prediction) + 1)

    if not args.quiet:
        print()
        print("=" * 96)
        print(f"ALL-ROOKIE REORDERER CURRENT PREDICTION {season}")
        print("=" * 96)
        for team_name, players in prediction_to_json_block(prediction, spec).items():
            print(f"{team_name}: " + ", ".join(players))
        print()
        cols = [
            "RANK", "PLAYER_NAME", "PREDICTED_LABEL",
            "BASE_EXPECTED_LABEL", "REORDERER_EXPECTED_LABEL", "FINAL_REORDER_SCORE",
        ]
        existing = [c for c in cols if c in prediction.columns]
        print(prediction[existing].to_string(index=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.experiment_name}_{season}_prediction.csv"
    json_path = Path(args.current_output) if args.current_output else output_dir / f"{args.experiment_name}_{season}_teams.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    prediction.to_csv(csv_path, index=False)
    payload = {
        "season": season,
        "all_rookie": prediction_to_json_block(prediction, spec),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"\n[saved prediction csv] {csv_path}")
        print(f"[saved prediction json] {json_path}")

    return prediction


def main():
    parser = parse_legacy_args(DESCRIPTION, EXPERIMENT, awards=["all_rookie"])
    parser.add_argument("--apply-current", action="store_true", help="Train on historical seasons and predict current All-Rookie teams.")
    parser.add_argument("--predict-season", type=int, default=2026, help="Season to predict when --apply-current is used.")
    parser.add_argument("--current-output", type=Path, default=None, help="Optional JSON path for current-season prediction.")
    args = parser.parse_args()
    print_experiment_header(args.experiment_name, DESCRIPTION, args, CONFIG)

    if args.apply_current:
        predict_current(args)
        return

    summary_df, season_df, prediction_rows = evaluate(args)
    write_outputs(args.experiment_name, args.output_dir, [summary_df], [season_df], prediction_rows, args.quiet)


if __name__ == "__main__":
    main()
