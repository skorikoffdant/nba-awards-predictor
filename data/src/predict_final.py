import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import add_season_rank_features

import train_hgb_classifier_full_score as hgb
import train_team_reorderer as reorder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEASON = 2026
MODELS_DIR = PROJECT_ROOT / "models" / "final"

ALL_NBA_MODEL_PATH = MODELS_DIR / "final_all_nba_hgb_reorderer.joblib"
ALL_ROOKIE_MODEL_PATH = MODELS_DIR / "final_all_rookie_hgb.joblib"

NBA_TEAM_NAMES = {
    3: "first",
    2: "second",
    1: "third",
}

ROOKIE_TEAM_NAMES = {
    2: "first",
    1: "second",
}


def apply_all_nba_eligibility_filter(df):
    df = df.copy()

    if "IS_ALL_NBA_ELIGIBLE" not in df.columns:
        return df

    eligible = df["IS_ALL_NBA_ELIGIBLE"].fillna(1).astype(int) == 1
    removed = df[~eligible].copy()

    if not removed.empty:
        cols = [
            "PLAYER_NAME",
            "GP",
            "MIN",
            "PTS",
            "ALL_NBA_ELIGIBILITY_REASON",
        ]
        cols = [col for col in cols if col in removed.columns]

        print()
        print("ALL-NBA eligibility filter removed:")
        print(removed[cols].to_string(index=False))

    return df[eligible].copy()


def predict_all_nba(artifact):
    df = reorder.load_dataset()
    test_df = df[df["SEASON_END_YEAR"] == SEASON].copy()
    test_df = apply_all_nba_eligibility_filter(test_df)

    if test_df.empty:
        raise RuntimeError(f"No eligible All-NBA rows found for season {SEASON}")

    config = artifact["config"]
    base_model = artifact["base_model"]
    reorderer_model = artifact["reorderer_model"]
    base_feature_columns = artifact["base_feature_columns"]
    reorderer_feature_columns = artifact["reorderer_feature_columns"]

    scored_df = reorder.add_base_scores(
        test_df,
        model=base_model,
        feature_columns=base_feature_columns,
    )

    selected_df = reorder.select_top_unique_players(
        scored_df,
        score_col=config["base_selection_score_col"],
        top_n=15,
    )
    selected_df = reorder.add_base_rank_columns(selected_df)
    selected_df = selected_df.reset_index(drop=True)

    missing_features = []
    for col in reorderer_feature_columns:
        if col not in selected_df.columns:
            missing_features.append(col)

    if missing_features:
        raise RuntimeError(
            "Missing reorderer features during prediction: "
            + ", ".join(missing_features[:20])
        )

    base_values = reorder.make_base_team_values(selected_df)
    model_values = reorder.make_model_team_values(
        reorderer_model,
        selected_df,
        feature_columns=reorderer_feature_columns,
        config=config,
    )

    base_norm = reorder.minmax_array(base_values)
    model_norm = reorder.minmax_array(model_values)
    blend = float(config["blend_weight"])
    final_values = (1.0 - blend) * base_norm + blend * model_norm

    prediction_df = reorder.assign_from_team_value_matrix(
        selected_df,
        value_matrix=final_values,
    )

    prediction_df["AWARD"] = "all_nba"
    prediction_df["PREDICTED_TEAM"] = prediction_df["PREDICTED_LABEL"].map(NBA_TEAM_NAMES)
    prediction_df = prediction_df.sort_values(
        ["PREDICTED_LABEL", "EXPECTED_LABEL", "PLAYER_NAME"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    prediction_df["FINAL_RANK"] = np.arange(1, len(prediction_df) + 1)

    return prediction_df


def predict_all_rookie(artifact):
    award_cfg = hgb.AWARD_CONFIGS["all_rookie"]
    df = hgb.load_dataset(award_cfg["dataset_path"], award_cfg["label_col"])
    df = add_season_rank_features(df)

    test_df = df[df["SEASON_END_YEAR"] == SEASON].copy()
    if test_df.empty:
        raise RuntimeError(f"No All-Rookie rows found for season {SEASON}")

    prediction_df = hgb.make_prediction(
        test_df=test_df,
        model=artifact["model"],
        feature_columns=artifact["feature_columns"],
        config=artifact["config"],
        award_cfg=award_cfg,
    )

    prediction_df["AWARD"] = "all_rookie"
    prediction_df["PREDICTED_TEAM"] = prediction_df["PREDICTED_LABEL"].map(ROOKIE_TEAM_NAMES)
    prediction_df = prediction_df.sort_values(
        ["PREDICTED_LABEL", "EXPECTED_LABEL", "PLAYER_NAME"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    prediction_df["FINAL_RANK"] = np.arange(1, len(prediction_df) + 1)

    return prediction_df


def teams_from_prediction(prediction_df, team_names):
    teams = {}

    for label, team_name in sorted(team_names.items(), reverse=True):
        players = prediction_df[prediction_df["PREDICTED_LABEL"] == label]
        players = players.sort_values(["FINAL_RANK", "PLAYER_NAME"])
        teams[team_name] = players["PLAYER_NAME"].astype(str).tolist()

    return teams


def write_predictions_csv(all_nba, all_rookie, output_json):
    csv_path = output_json.with_suffix(".csv")
    combined = pd.concat([all_nba, all_rookie], ignore_index=True)

    cols = [
        "AWARD",
        "FINAL_RANK",
        "PREDICTED_TEAM",
        "PREDICTED_LABEL",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
        "TEAM_ABBREVIATION",
        "SEASON_END_YEAR",
        "EXPECTED_LABEL",
        "P_AWARD",
        "IS_ALL_NBA_ELIGIBLE",
        "ALL_NBA_ELIGIBILITY_REASON",
    ]
    cols = [col for col in cols if col in combined.columns]

    combined[cols].to_csv(csv_path, index=False)
    return csv_path


def load_models():
    if not ALL_NBA_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing All-NBA model: {ALL_NBA_MODEL_PATH}")
    if not ALL_ROOKIE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing All-Rookie model: {ALL_ROOKIE_MODEL_PATH}")

    all_nba_artifact = joblib.load(ALL_NBA_MODEL_PATH)
    all_rookie_artifact = joblib.load(ALL_ROOKIE_MODEL_PATH)

    return all_nba_artifact, all_rookie_artifact


def save_json(output_json, all_nba_pred, all_rookie_pred):
    output = {
        "season": SEASON,
        "all_nba": teams_from_prediction(all_nba_pred, NBA_TEAM_NAMES),
        "all_rookie": teams_from_prediction(all_rookie_pred, ROOKIE_TEAM_NAMES),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output


def print_prediction(output):
    print()
    print("=" * 80)
    print("FINAL PREDICTIONS")
    print("=" * 80)
    print(f"season={SEASON}")
    print()

    print("All-NBA")
    for team, players in output["all_nba"].items():
        print(f"  {team}: {', '.join(players)}")

    print()
    print("All-Rookie")
    for team, players in output["all_rookie"].items():
        print(f"  {team}: {', '.join(players)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    all_nba_artifact, all_rookie_artifact = load_models()

    all_nba_pred = predict_all_nba(all_nba_artifact)
    all_rookie_pred = predict_all_rookie(all_rookie_artifact)

    output = save_json(args.output, all_nba_pred, all_rookie_pred)
    csv_path = write_predictions_csv(all_nba_pred, all_rookie_pred, args.output)

    print_prediction(output)
    print()
    print(f"[saved] {args.output}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()