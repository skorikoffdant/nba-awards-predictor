from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

from features import (
    FEATURE_SETS,
    get_feature_columns,
    missing_columns,
    unique_keep_order,
    add_season_rank_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_DATASETS = {
    "all_nba": PROCESSED_DIR / "all_nba_dataset.csv",
    "all_rookie": PROCESSED_DIR / "all_rookie_dataset.csv",
    "labeled": PROCESSED_DIR / "player_seasons_labeled.csv",
}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def print_dataset_info(df: pd.DataFrame, path: Path) -> None:
    print()
    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"Path:  {path}")
    print(f"Shape: {df.shape}")

    if "SEASON_END_YEAR" in df.columns:
        print(
            "Seasons:",
            int(df["SEASON_END_YEAR"].min()),
            "-",
            int(df["SEASON_END_YEAR"].max()),
        )

    if "ALL_NBA_LABEL" in df.columns:
        print()
        print("ALL_NBA_LABEL counts:")
        print(df["ALL_NBA_LABEL"].value_counts().sort_index())

    if "ALL_ROOKIE_LABEL" in df.columns:
        print()
        print("ALL_ROOKIE_LABEL counts:")
        print(df["ALL_ROOKIE_LABEL"].value_counts().sort_index())


def check_required_base_columns(df: pd.DataFrame) -> list[str]:
    required = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
    ]

    missing = [col for col in required if col not in df.columns]

    print()
    print("=" * 80)
    print("BASE COLUMNS CHECK")
    print("=" * 80)

    if missing:
        print("Missing base columns:")
        for col in missing:
            print(f"  - {col}")
    else:
        print("OK: all base columns are present.")

    return missing


def check_feature_set(
    df: pd.DataFrame,
    feature_set: str,
    fail_on_missing: bool,
) -> dict:
    requested = unique_keep_order(FEATURE_SETS[feature_set])
    existing = get_feature_columns(
        df,
        feature_set=feature_set,
        verbose=False,
    )
    missing = missing_columns(df, requested)

    print()
    print("=" * 80)
    print(f"FEATURE SET CHECK: {feature_set}")
    print("=" * 80)
    print(f"Requested: {len(requested)}")
    print(f"Existing:  {len(existing)}")
    print(f"Missing:   {len(missing)}")

    if missing:
        print()
        print("Missing columns:")
        for col in missing:
            print(f"  - {col}")
    else:
        print()
        print("OK: all requested features are present.")

    if existing:
        numeric_existing = [
            col for col in existing
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        non_numeric = [
            col for col in existing
            if col not in numeric_existing
        ]

        print()
        print(f"Numeric features:     {len(numeric_existing)}")
        print(f"Non-numeric features: {len(non_numeric)}")

        if non_numeric:
            print()
            print("Non-numeric feature columns:")
            for col in non_numeric:
                print(f"  - {col}: {df[col].dtype}")

        nan_report = (
            df[existing]
            .isna()
            .mean()
            .sort_values(ascending=False)
        )

        high_nan = nan_report[nan_report > 0.30]

        print()
        print(f"Features with NaN ratio > 30%: {len(high_nan)}")

        if not high_nan.empty:
            print()
            print(high_nan.to_string())

    status_ok = len(missing) == 0 or not fail_on_missing

    return {
        "feature_set": feature_set,
        "requested": len(requested),
        "existing": len(existing),
        "missing": missing,
        "ok": status_ok,
    }


def check_specific_new_features(df: pd.DataFrame) -> list[str]:
    important_new_features = [
        "TOTAL_MIN",
        "TOTAL_PTS",
        "TOTAL_REB",
        "TOTAL_AST",
        "PTS_REB_AST",
        "STOCKS",
        "AST_TOV_SIMPLE",

        "PREV_target_all_nba",
        "PREV_PTS",
        "PREV_REB",
        "PREV_AST",
        "PREV_STL",
        "PREV_BLK",
        "PREV_GP",
        "PREV_MIN",
        "PREV_W_PCT",
        "PREV_TS_PCT",
        "PREV_USG_PCT",
        "PREV_PIE",

        "IS_MULTI_TEAM_PLAYER",
        "TEAM_PLAYER_COUNT",
        "TEAM_TOTAL_PTS_RANK",
        "TEAM_TOTAL_AST_RANK",
        "TEAM_TOTAL_MIN_RANK",
        "TEAM_TOTAL_PTS_SHARE",
        "TEAM_TOTAL_AST_SHARE",
        "TEAM_TOTAL_MIN_SHARE",
        "IS_TEAM_TOTAL_PTS_LEADER",
        "IS_TEAM_TOTAL_MIN_LEADER",
        "TEAM_TOP3_TOTAL_PTS_FLAG",
        "TEAM_TOP3_TOTAL_MIN_FLAG",
        "LEAGUE_TEAM_W_PCT_RANK",
        "LEAGUE_TEAM_NET_RATING_RANK",
    ]

    missing = [col for col in important_new_features if col not in df.columns]

    print()
    print("=" * 80)
    print("IMPORTANT NEW FEATURES CHECK")
    print("=" * 80)

    if missing:
        print(f"Missing important new features: {len(missing)}")
        for col in missing:
            print(f"  - {col}")
    else:
        print("OK: all important new features are present.")

    return missing


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["all_nba", "all_rookie", "labeled"],
        default="all_nba",
    )

    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Optional custom csv path.",
    )

    parser.add_argument(
        "--feature-set",
        type=str,
        default="previous_team_share",
        choices=list(FEATURE_SETS.keys()) + ["all"],
    )

    parser.add_argument(
        "--no-add-rank-features",
        action="store_true",
        help="Do not generate *_SEASON_RANK_PCT features before checking.",
    )

    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Do not exit with error code if features are missing.",
    )

    args = parser.parse_args()

    if args.path is not None:
        dataset_path = Path(args.path)
    else:
        dataset_path = DEFAULT_DATASETS[args.dataset]

    df = load_dataset(dataset_path)

    if not args.no_add_rank_features:
        df = add_season_rank_features(df)

    print_dataset_info(df, dataset_path)

    base_missing = check_required_base_columns(df)

    important_missing = check_specific_new_features(df)

    feature_sets_to_check = (
        list(FEATURE_SETS.keys())
        if args.feature_set == "all"
        else [args.feature_set]
    )

    results = []

    for feature_set in feature_sets_to_check:
        result = check_feature_set(
            df=df,
            feature_set=feature_set,
            fail_on_missing=not args.warn_only,
        )
        results.append(result)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    has_errors = False

    if base_missing:
        has_errors = True
        print(f"Base columns missing: {len(base_missing)}")

    if important_missing:
        has_errors = True
        print(f"Important new features missing: {len(important_missing)}")

    for result in results:
        missing_count = len(result["missing"])
        print(
            f"{result['feature_set']}: "
            f"{result['existing']}/{result['requested']} existing, "
            f"{missing_count} missing"
        )

        if missing_count > 0:
            has_errors = True

    if has_errors and not args.warn_only:
        print()
        print("RESULT: FAIL")
        sys.exit(1)

    print()
    print("RESULT: OK")


if __name__ == "__main__":
    main()