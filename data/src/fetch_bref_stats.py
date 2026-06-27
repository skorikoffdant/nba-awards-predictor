from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_SEASON = 2000
MAX_SEASON = 2026

OUTPUT_PATH = PROCESSED_DIR / "player_seasons_raw.csv"

KEY_COLS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "SEASON_END_YEAR",
    "SEASON",
]

MEASURE_TYPES = [
    "Base",
    "Advanced",
    "Usage",
]


def season_end_to_nba_format(season_end_year: int) -> str:
    start_year = season_end_year - 1
    end_short = str(season_end_year)[-2:]
    return f"{start_year}-{end_short}"


def fetch_player_stats(season_end_year: int, measure_type: str) -> pd.DataFrame:
    season = season_end_to_nba_format(season_end_year)
    print(f"Downloading {measure_type} stats for season {season}")

    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense=measure_type,
        timeout=60,
    )

    df = endpoint.get_data_frames()[0]
    df["SEASON_END_YEAR"] = season_end_year
    df["SEASON"] = season

    time.sleep(1.5)
    return df


def save_raw_measure(
    df: pd.DataFrame,
    season_end_year: int,
    measure_type: str,
) -> None:
    path = RAW_DIR / f"nba_api_{season_end_year}_{measure_type.lower()}.csv"
    df.to_csv(path, index=False)


def merge_measure_stats(base: pd.DataFrame, advanced: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    missing_key_cols = [col for col in KEY_COLS if col not in base.columns]

    if missing_key_cols:
        season_end_year = base["SEASON_END_YEAR"].iloc[0] if "SEASON_END_YEAR" in base.columns else "unknown"
        raise RuntimeError(
            f"Missing key columns in base stats for {season_end_year}: {missing_key_cols}"
        )

    advanced_extra_cols = [col for col in advanced.columns if col not in base.columns]
    usage_extra_cols = [
        col for col in usage.columns
        if col not in base.columns and col not in advanced_extra_cols
    ]

    merged = base.merge(
        advanced[KEY_COLS + advanced_extra_cols],
        on=KEY_COLS,
        how="left",
    )

    merged = merged.merge(
        usage[KEY_COLS + usage_extra_cols],
        on=KEY_COLS,
        how="left",
    )

    return merged


def fetch_one_season(season_end_year: int) -> pd.DataFrame:
    stats = {
        measure_type: fetch_player_stats(season_end_year, measure_type)
        for measure_type in MEASURE_TYPES
    }

    for measure_type, df in stats.items():
        save_raw_measure(df, season_end_year, measure_type)

    return merge_measure_stats(
        base=stats["Base"],
        advanced=stats["Advanced"],
        usage=stats["Usage"],
    )


def print_season_header(season_end_year: int) -> None:
    print()
    print("=" * 80)
    print(f"Season end year: {season_end_year}")
    print("=" * 80)


def print_output_info(df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("PLAYER STATS SAVED")
    print("=" * 80)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("Seasons:", df["SEASON_END_YEAR"].min(), "-", df["SEASON_END_YEAR"].max())

    rows_by_season = df.groupby("SEASON_END_YEAR").size()

    print()
    print("Rows by first seasons:")
    print(rows_by_season.head(10))

    print()
    print("Rows by last seasons:")
    print(rows_by_season.tail(10))


def main() -> None:
    all_seasons = []

    for season_end_year in range(MIN_SEASON, MAX_SEASON + 1):
        print_season_header(season_end_year)

        try:
            df = fetch_one_season(season_end_year)
        except Exception as exc:
            print(f"ERROR for {season_end_year}: {exc}")
            continue

        if df.empty:
            print(f"WARNING: empty dataframe for {season_end_year}")
            continue

        all_seasons.append(df)

    if not all_seasons:
        raise RuntimeError("No data downloaded.")

    combined = pd.concat(all_seasons, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print_output_info(combined)


if __name__ == "__main__":
    main()