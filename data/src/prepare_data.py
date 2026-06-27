import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STATIC_DIR = PROJECT_ROOT / "data" / "static"


STEPS = [
    ("fetch nba_api player stats", ["data/src/fetch_bref_stats.py"]),
    ("fetch award labels", ["data/src/fetch_award_labels.py"]),
    ("fetch Basketball-Reference advanced stats", ["data/src/fetch_bref_advanced.py"]),
    ("build final datasets", ["data/src/build_dataset.py"]),
    ("apply static official overrides", ["data/src/apply_static_overrides.py"]),
    (
        "check All-NBA dataset features",
        [
            "data/src/check_features.py",
            "--dataset",
            "all_nba",
            "--feature-set",
            "previous_team_share_allstar",
        ],
    ),
    (
        "check All-Rookie dataset features",
        [
            "data/src/check_features.py",
            "--dataset",
            "all_rookie",
            "--feature-set",
            "previous_team_share_allstar",
        ],
    ),
]


SUMMARY_FILES = [
    PROCESSED_DIR / "player_seasons_raw.csv",
    RAW_DIR / "all_nba_labels.csv",
    RAW_DIR / "all_rookie_labels.csv",
    RAW_DIR / "bref_advanced_stats.csv",
    RAW_DIR / "all_star_players.csv",
    STATIC_DIR / "award_eligibility_overrides.csv",
    STATIC_DIR / "official_all_star_rosters.csv",
    PROCESSED_DIR / "player_seasons_labeled.csv",
    PROCESSED_DIR / "all_nba_dataset.csv",
    PROCESSED_DIR / "all_rookie_dataset.csv",
]


def run_step(title, command):
    print(f"[RUN] {title}", flush=True)
    subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=True)
    print(f"[OK]  {title}", flush=True)


def summarize_file(path):
    relative_path = path.relative_to(PROJECT_ROOT)

    if not path.exists():
        return str(relative_path), "missing", "missing", "missing"

    df = pd.read_csv(path)
    seasons = "-"

    if "SEASON_END_YEAR" in df.columns and not df.empty:
        first_season = int(df["SEASON_END_YEAR"].min())
        last_season = int(df["SEASON_END_YEAR"].max())
        seasons = f"{first_season}-{last_season}"

    return str(relative_path), len(df), len(df.columns), seasons


def print_summary():
    print()
    print("=" * 80)
    print("DATA FILES SUMMARY")
    print("=" * 80)
    print(f"{'file':<62} {'rows':>8} {'cols':>6} {'seasons':>12}")
    print("-" * 92)

    for path in SUMMARY_FILES:
        file_name, rows, cols, seasons = summarize_file(path)
        print(f"{file_name:<62} {str(rows):>8} {str(cols):>6} {str(seasons):>12}")


def main():
    for title, command in STEPS:
        run_step(title, command)

    print_summary()


if __name__ == "__main__":
    main()