from __future__ import annotations

from io import StringIO
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

MIN_SEASON = 2000
MAX_SEASON = 2026
SLEEP_SEC = 3.0

OUTPUT_PATH = RAW_DIR / "bref_advanced_stats.csv"

NAME_ALIASES = {
    "jimmy butler": "jimmy butler iii",
    "ron artest": "metta world peace",
    "donovan clinglan": "donovan clingan",
    "gg jackson ii": "gg jackson",
    "terence davis ii": "terence davis",
    "pj washington jr": "pj washington",
    "walter hermann": "walter herrmann",
    "nene hilario": "nene",
    "gordon giricek": "gordan giricek",
}

BREF_FEATURES = [
    "PER",
    "OWS",
    "DWS",
    "WS",
    "WS_PER_48",
    "OBPM",
    "DBPM",
    "BPM",
    "VORP",
]

RAW_TO_CANONICAL = {
    "PER": "PER",
    "OWS": "OWS",
    "DWS": "DWS",
    "WS": "WS",
    "WS/48": "WS_PER_48",
    "OBPM": "OBPM",
    "DBPM": "DBPM",
    "BPM": "BPM",
    "VORP": "VORP",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def clean_player_name(name: str) -> str:
    name = str(name)
    name = name.replace("\xa0", " ")
    name = name.replace("*", "")
    name = name.replace("^", "")
    name = name.replace("†", "")
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def normalize_name_key(name: str) -> str:
    name = clean_player_name(name)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def apply_name_alias(name_key: str) -> str:
    return NAME_ALIASES.get(name_key, name_key)


def make_player_name_key(name: str) -> str:
    name_key = normalize_name_key(name)
    name_key = apply_name_alias(name_key)

    return name_key


def season_end_to_bref_url(season_end_year: int) -> str:
    return (
        "https://www.basketball-reference.com/leagues/"
        f"NBA_{int(season_end_year)}_advanced.html"
    )


def read_html_tables_including_comments(html: str) -> list[pd.DataFrame]:
    tables = []

    try:
        tables.extend(pd.read_html(StringIO(html)))
    except ValueError:
        pass

    comment_blocks = re.findall(r"<!--(.*?)-->", html, flags=re.DOTALL)

    for block in comment_blocks:
        if "<table" not in block:
            continue

        try:
            tables.extend(pd.read_html(StringIO(block)))
        except ValueError:
            continue

    return tables


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            str(parts[-1]).strip()
            for parts in df.columns.to_flat_index()
        ]
    else:
        df.columns = [
            str(col).strip()
            for col in df.columns
        ]

    return df


def find_advanced_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    required_cols = {"Player", "PER", "WS", "BPM", "VORP"}

    for table in tables:
        table = flatten_columns(table)

        if required_cols.issubset(set(table.columns)):
            return table

    raise RuntimeError("Could not find Basketball Reference advanced stats table")


def clean_bref_advanced_table(
    raw_df: pd.DataFrame,
    season_end_year: int,
) -> pd.DataFrame:
    df = flatten_columns(raw_df)

    if "Player" not in df.columns:
        raise RuntimeError("B-Ref advanced table does not contain Player column")

    df = df[df["Player"].astype(str) != "Player"].copy()
    df = df[df["Player"].notna()].copy()

    df["PLAYER_NAME"] = df["Player"].apply(clean_player_name)
    df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].apply(make_player_name_key)
    df["SEASON_END_YEAR"] = int(season_end_year)

    if "Tm" in df.columns:
        df["BREF_TEAM"] = df["Tm"].astype(str)
    else:
        df["BREF_TEAM"] = ""

    if "G" in df.columns:
        df["BREF_G"] = pd.to_numeric(df["G"], errors="coerce")
    else:
        df["BREF_G"] = np.nan

    for raw_col, canonical_col in RAW_TO_CANONICAL.items():
        if raw_col in df.columns:
            df[canonical_col] = pd.to_numeric(df[raw_col], errors="coerce")
        elif canonical_col not in df.columns:
            df[canonical_col] = np.nan

    df["_IS_TOT"] = (df["BREF_TEAM"] == "TOT").astype(int)
    df["_SORT_G"] = df["BREF_G"].fillna(-1)

    df = df.sort_values(
        ["SEASON_END_YEAR", "PLAYER_NAME_KEY", "_IS_TOT", "_SORT_G"],
        ascending=[True, True, False, False],
    )

    keep_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
        "BREF_TEAM",
        "BREF_G",
    ] + BREF_FEATURES

    df = df[keep_cols]
    df = df.drop_duplicates(
        subset=["SEASON_END_YEAR", "PLAYER_NAME_KEY"],
        keep="first",
    )

    return df.reset_index(drop=True)


def fetch_bref_advanced_for_season(season_end_year: int) -> pd.DataFrame:
    url = season_end_to_bref_url(season_end_year)

    print(f"Fetching B-Ref advanced stats: {url}", flush=True)

    response = requests.get(url, headers=HEADERS, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"status={response.status_code}")

    tables = read_html_tables_including_comments(response.text)
    raw_table = find_advanced_table(tables)
    cleaned = clean_bref_advanced_table(raw_table, season_end_year)

    print(f"  rows: {len(cleaned)}", flush=True)

    return cleaned


def fetch_bref_advanced_stats(
    min_season: int = MIN_SEASON,
    max_season: int = MAX_SEASON,
    sleep_sec: float = SLEEP_SEC,
) -> pd.DataFrame:
    parts = []

    for season_end_year in range(int(min_season), int(max_season) + 1):
        print()
        print("=" * 80)
        print(f"B-Ref season end year: {season_end_year}")
        print("=" * 80)

        try:
            season_df = fetch_bref_advanced_for_season(season_end_year)
        except Exception as exc:
            print(f"WARNING: failed for {season_end_year}: {exc}", flush=True)
            time.sleep(sleep_sec)
            continue

        if not season_df.empty:
            parts.append(season_df)

        time.sleep(sleep_sec)

    if not parts:
        raise RuntimeError("No Basketball Reference advanced data downloaded")

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["SEASON_END_YEAR", "PLAYER_NAME_KEY"])
    result = result.reset_index(drop=True)

    return result


def main() -> None:
    df = fetch_bref_advanced_stats(
        min_season=MIN_SEASON,
        max_season=MAX_SEASON,
        sleep_sec=SLEEP_SEC,
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 80)
    print("B-REF ADVANCED STATS SAVED")
    print("=" * 80)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("Seasons:", df["SEASON_END_YEAR"].min(), "-", df["SEASON_END_YEAR"].max())
    print()
    print("Columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()