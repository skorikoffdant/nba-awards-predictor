from __future__ import annotations

import argparse
import re
import time
import unicodedata
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests as std_requests
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.basketball-reference.com/",
}

AWARDS = ["all_nba", "all_rookie"]

AWARD_CONFIG = {
    "all_nba": {
        "prefix": "ALL_NBA",
        "table_ids": {"all_league", "all_nba"},
        "votes_path": RAW_DIR / "all_nba_voting.csv",
        "dataset_path": PROCESSED_DIR / "all_nba_dataset.csv",
        "output_path": PROCESSED_DIR / "all_nba_dataset_with_votes.csv",
    },
    "all_rookie": {
        "prefix": "ALL_ROOKIE",
        "table_ids": {"all_rookie"},
        "votes_path": RAW_DIR / "all_rookie_voting.csv",
        "dataset_path": PROCESSED_DIR / "all_rookie_dataset.csv",
        "output_path": PROCESSED_DIR / "all_rookie_dataset_with_votes.csv",
    },
}


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""

    name = str(name)
    name = name.replace("*", "")
    name = name.replace("’", "'").replace("‘", "'")
    name = name.replace("`", "'").replace("´", "'")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower()
    name = re.sub(r"[^a-z0-9' .-]", "", name)
    name = " ".join(name.split())

    return name


def clean_player_name(name: str) -> str:
    if pd.isna(name):
        return ""

    name = str(name)
    name = name.replace("*", "")
    name = re.sub(r"\s+", " ", name).strip()

    return name


def compact_col_name(col: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def normalize_col_name(col: object) -> str:
    if not isinstance(col, tuple):
        return re.sub(r"\s+", " ", str(col)).strip()

    parts = []

    for value in col:
        text = str(value)

        if text.lower() == "nan":
            continue

        if text.startswith("Unnamed:"):
            continue

        parts.append(text)

    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(col) for col in df.columns]

    return df


def get_col(
    df: pd.DataFrame,
    candidates: list[str],
    exclude: list[str] | None = None,
) -> str | None:
    exclude = exclude or []

    for exact_match in [True, False]:
        for col in df.columns:
            col_key = compact_col_name(col)

            for candidate in candidates:
                candidate_key = compact_col_name(candidate)

                if exact_match and col_key != candidate_key:
                    continue

                if not exact_match and candidate_key not in col_key:
                    continue

                if any(compact_col_name(value) in col_key for value in exclude):
                    continue

                return col

    return None


def safe_get(url: str, timeout: int = 30) -> Any | None:
    if curl_requests is not None:
        try:
            response = curl_requests.get(
                url,
                headers=HEADERS,
                timeout=timeout,
                impersonate="chrome120",
            )

            if response.status_code == 200:
                return response

            print(f"[WARN] curl_cffi GET failed: {response.status_code} {url}")
        except Exception as exc:
            print(f"[WARN] curl_cffi GET exception: {exc} {url}")

    try:
        response = std_requests.get(url, headers=HEADERS, timeout=timeout)

        if response.status_code == 200:
            return response

        print(f"[WARN] requests GET failed: {response.status_code} {url}")
    except Exception as exc:
        print(f"[WARN] requests GET exception: {exc} {url}")

    return None


def bref_awards_url(season_end_year: int) -> str:
    return f"https://www.basketball-reference.com/awards/awards_{season_end_year}.html"


def get_award_config(award: str) -> dict:
    if award not in AWARD_CONFIG:
        raise ValueError(f"Unknown award: {award}")

    return AWARD_CONFIG[award]


def clean_vote_dataframe(df: pd.DataFrame, award: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    required_cols = [
        "SEASON_END_YEAR",
        "AWARD",
        "PLAYER_NAME",
        "TEAM_TEXT",
        "FIRST_TEAM_VOTES",
        "SECOND_TEAM_VOTES",
        "THIRD_TEAM_VOTES",
        "VOTE_SCORE",
        "VOTE_MAX",
        "VOTE_SHARE",
        "player_key",
        "SOURCE",
        "SOURCE_URL",
    ]

    for col in required_cols:
        if col in df.columns:
            continue

        if col in {"TEAM_TEXT", "SOURCE", "SOURCE_URL"}:
            df[col] = ""
        elif col == "AWARD":
            df[col] = award
        else:
            df[col] = 0.0

    df["SEASON_END_YEAR"] = pd.to_numeric(df["SEASON_END_YEAR"], errors="coerce")
    df = df[df["SEASON_END_YEAR"].notna()].copy()
    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)

    df["AWARD"] = award
    df["PLAYER_NAME"] = df["PLAYER_NAME"].map(clean_player_name)
    df["player_key"] = df["PLAYER_NAME"].map(normalize_name)
    df = df[df["player_key"].str.len() > 0].copy()

    numeric_cols = [
        "FIRST_TEAM_VOTES",
        "SECOND_TEAM_VOTES",
        "THIRD_TEAM_VOTES",
        "VOTE_SCORE",
        "VOTE_MAX",
        "VOTE_SHARE",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df[df["VOTE_SCORE"] > 0].copy()

    if df.empty:
        return df

    if (df["VOTE_SHARE"] <= 0).all() and (df["VOTE_MAX"] > 0).any():
        df["VOTE_SHARE"] = df["VOTE_SCORE"] / df["VOTE_MAX"].replace(0, pd.NA)
        df["VOTE_SHARE"] = df["VOTE_SHARE"].fillna(0.0)

    season_max = df.groupby("SEASON_END_YEAR")["VOTE_SCORE"].transform("max")
    df["VOTE_SCORE_SEASON_MAX_NORM"] = (df["VOTE_SCORE"] / season_max).fillna(0.0)

    missing_share = df["VOTE_SHARE"].isna() | (df["VOTE_SHARE"] <= 0)
    df.loc[missing_share, "VOTE_SHARE"] = df.loc[
        missing_share,
        "VOTE_SCORE_SEASON_MAX_NORM",
    ]

    duplicate_mask = df.duplicated(["SEASON_END_YEAR", "player_key"], keep=False)

    if duplicate_mask.any():
        print("[WARN] duplicate vote rows, keeping highest VOTE_SCORE:")
        print(
            df.loc[
                duplicate_mask,
                [
                    "SEASON_END_YEAR",
                    "PLAYER_NAME",
                    "player_key",
                    "VOTE_SCORE",
                    "VOTE_SHARE",
                ],
            ]
            .sort_values(["SEASON_END_YEAR", "player_key", "VOTE_SCORE"])
            .to_string(index=False)
        )

    df = (
        df.sort_values(
            ["SEASON_END_YEAR", "player_key", "VOTE_SCORE"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["SEASON_END_YEAR", "player_key"], keep="first")
        .copy()
    )

    return df[
        [
            "SEASON_END_YEAR",
            "AWARD",
            "PLAYER_NAME",
            "TEAM_TEXT",
            "FIRST_TEAM_VOTES",
            "SECOND_TEAM_VOTES",
            "THIRD_TEAM_VOTES",
            "VOTE_SCORE",
            "VOTE_MAX",
            "VOTE_SHARE",
            "VOTE_SCORE_SEASON_MAX_NORM",
            "player_key",
            "SOURCE",
            "SOURCE_URL",
        ]
    ].copy()


def parse_bref_award_table(
    table_html: str,
    season_end_year: int,
    award: str,
    source_url: str,
) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(table_html))
    except Exception as exc:
        print(f"[WARN] pd.read_html failed for {season_end_year} {award}: {exc}")
        return pd.DataFrame()

    rows = []

    for table in tables:
        df = flatten_columns(table)

        player_col = get_col(df, ["Player"])
        team_col = get_col(df, ["Tm", "Team"])
        pts_col = get_col(
            df,
            ["Pts Won", "Voting Pts Won", "Points Won", "Total Points"],
            exclude=["max"],
        )
        max_col = get_col(
            df,
            ["Pts Max", "Voting Pts Max", "Max"],
            exclude=["won"],
        )
        share_col = get_col(df, ["Share", "Voting Share"])
        first_col = get_col(df, ["1st", "Voting 1st", "First"])
        second_col = get_col(df, ["2nd", "Voting 2nd", "Second"])
        third_col = get_col(df, ["3rd", "Voting 3rd", "Third"])

        if player_col is None or pts_col is None:
            continue

        out = pd.DataFrame(index=df.index)
        out["SEASON_END_YEAR"] = int(season_end_year)
        out["AWARD"] = award
        out["PLAYER_NAME"] = df[player_col].map(clean_player_name)
        out["TEAM_TEXT"] = df[team_col].astype(str).fillna("") if team_col is not None else ""

        vote_cols = {
            "FIRST_TEAM_VOTES": first_col,
            "SECOND_TEAM_VOTES": second_col,
            "THIRD_TEAM_VOTES": third_col,
        }

        for new_col, old_col in vote_cols.items():
            if old_col is None:
                out[new_col] = 0.0
            else:
                out[new_col] = pd.to_numeric(df[old_col], errors="coerce").fillna(0.0)

        out["VOTE_SCORE"] = pd.to_numeric(df[pts_col], errors="coerce").fillna(0.0)
        out["VOTE_MAX"] = (
            pd.to_numeric(df[max_col], errors="coerce").fillna(0.0)
            if max_col is not None
            else 0.0
        )
        out["VOTE_SHARE"] = (
            pd.to_numeric(df[share_col], errors="coerce").fillna(0.0)
            if share_col is not None
            else 0.0
        )
        out["player_key"] = out["PLAYER_NAME"].map(normalize_name)
        out["SOURCE"] = "basketball_reference"
        out["SOURCE_URL"] = source_url

        out = out[out["PLAYER_NAME"].str.lower() != "player"].copy()
        out = out[out["player_key"].str.len() > 0].copy()
        out = out[out["VOTE_SCORE"] > 0].copy()

        if not out.empty:
            rows.append(out)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)

    return clean_vote_dataframe(result, award=award)


def extract_bref_table_html(html: str, award: str) -> str | None:
    html = html.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(html, "html.parser")
    config = get_award_config(award)

    for table_id in config["table_ids"]:
        table = soup.find("table", id=table_id)

        if table is not None:
            return str(table)

    for table in soup.find_all("table"):
        table_id = (table.get("id") or "").lower()
        caption = table.find("caption")
        caption_text = caption.get_text(" ", strip=True).lower() if caption else ""

        if award == "all_nba":
            if "all-nba" in caption_text or "all nba" in caption_text or table_id == "all_league":
                return str(table)

        if award == "all_rookie":
            if "all-rookie" in caption_text or "all rookie" in caption_text or table_id == "all_rookie":
                return str(table)

    return None


def fetch_bref_votes_for_year(season_end_year: int, award: str) -> pd.DataFrame:
    url = bref_awards_url(season_end_year)
    response = safe_get(url)

    if response is None:
        return pd.DataFrame()

    table_html = extract_bref_table_html(response.text, award)

    if table_html is None:
        print(f"[MISS] no BRef table found: {award} {season_end_year}")
        return pd.DataFrame()

    return parse_bref_award_table(
        table_html=table_html,
        season_end_year=season_end_year,
        award=award,
        source_url=url,
    )


def fetch_bref_votes(
    start_year: int,
    end_year: int,
    award: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    rows = []

    for year in range(start_year, end_year + 1):
        print(f"[fetch] {award} {year}")

        df = fetch_bref_votes_for_year(year, award)

        if df.empty:
            print(f"[MISS] {award} {year}")
        else:
            print(f"[OK] {award} {year}: {len(df)} rows from Basketball-Reference")
            rows.append(df)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)

    return clean_vote_dataframe(result, award=award)


def find_name_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "PLAYER_NAME",
        "PLAYER",
        "player_name",
        "player",
        "NAME",
        "name",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if compact_col_name(col) in {"playername", "player", "name"}:
            return col

    return None


def prepare_dataset_keys(df: pd.DataFrame, dataset_path: Path) -> pd.DataFrame:
    df = df.copy()

    if "player_key" in df.columns:
        df["player_key"] = df["player_key"].map(normalize_name)
        return df

    name_col = find_name_column(df)

    if name_col is None:
        raise ValueError(
            f"{dataset_path} has no player_key and no recognizable player name column"
        )

    df["player_key"] = df[name_col].map(normalize_name)

    return df


def prepare_votes_for_merge(votes: pd.DataFrame, prefix: str, award: str) -> pd.DataFrame:
    votes = votes.copy()

    if "player_key" not in votes.columns:
        votes["player_key"] = votes["PLAYER_NAME"].map(normalize_name)

    votes = clean_vote_dataframe(votes, award=award)

    votes = votes[
        [
            "SEASON_END_YEAR",
            "player_key",
            "PLAYER_NAME",
            "TEAM_TEXT",
            "FIRST_TEAM_VOTES",
            "SECOND_TEAM_VOTES",
            "THIRD_TEAM_VOTES",
            "VOTE_SCORE",
            "VOTE_MAX",
            "VOTE_SHARE",
            "VOTE_SCORE_SEASON_MAX_NORM",
            "SOURCE",
            "SOURCE_URL",
        ]
    ].copy()

    return votes.rename(
        columns={
            "PLAYER_NAME": f"{prefix}_VOTE_PLAYER_NAME",
            "TEAM_TEXT": f"{prefix}_VOTE_TEAM_TEXT",
            "FIRST_TEAM_VOTES": f"{prefix}_FIRST_TEAM_VOTES",
            "SECOND_TEAM_VOTES": f"{prefix}_SECOND_TEAM_VOTES",
            "THIRD_TEAM_VOTES": f"{prefix}_THIRD_TEAM_VOTES",
            "VOTE_SCORE": f"{prefix}_VOTE_SCORE",
            "VOTE_MAX": f"{prefix}_VOTE_MAX",
            "VOTE_SHARE": f"{prefix}_VOTE_SHARE",
            "VOTE_SCORE_SEASON_MAX_NORM": f"{prefix}_VOTE_SCORE_SEASON_MAX_NORM",
            "SOURCE": f"{prefix}_VOTE_SOURCE",
            "SOURCE_URL": f"{prefix}_VOTE_SOURCE_URL",
        }
    )


def drop_old_vote_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    old_vote_cols = [
        col
        for col in df.columns
        if col.startswith(f"{prefix}_")
        and (
            "VOTE" in col
            or col.endswith("FIRST_TEAM_VOTES")
            or col.endswith("SECOND_TEAM_VOTES")
            or col.endswith("THIRD_TEAM_VOTES")
            or col.endswith("RECEIVED_VOTES")
        )
    ]

    if old_vote_cols:
        return df.drop(columns=old_vote_cols)

    return df


def fill_vote_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        f"{prefix}_FIRST_TEAM_VOTES",
        f"{prefix}_SECOND_TEAM_VOTES",
        f"{prefix}_THIRD_TEAM_VOTES",
        f"{prefix}_VOTE_SCORE",
        f"{prefix}_VOTE_MAX",
        f"{prefix}_VOTE_SHARE",
        f"{prefix}_VOTE_SCORE_SEASON_MAX_NORM",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    text_cols = [
        f"{prefix}_VOTE_PLAYER_NAME",
        f"{prefix}_VOTE_TEAM_TEXT",
        f"{prefix}_VOTE_SOURCE",
        f"{prefix}_VOTE_SOURCE_URL",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df[f"{prefix}_RECEIVED_VOTES"] = (df[f"{prefix}_VOTE_SCORE"] > 0).astype(int)

    return df


def print_vote_merge_check(
    merged: pd.DataFrame,
    votes: pd.DataFrame,
    output_path: Path,
    prefix: str,
) -> None:
    print()
    print(f"[saved] {output_path}")
    print(f"shape: {merged.shape}")
    print("vote coverage by season:")
    print(
        merged.groupby("SEASON_END_YEAR")[f"{prefix}_RECEIVED_VOTES"]
        .sum()
        .astype(int)
        .to_string()
    )

    votes_not_merged = votes.merge(
        merged[["SEASON_END_YEAR", "player_key"]].drop_duplicates(),
        on=["SEASON_END_YEAR", "player_key"],
        how="left",
        indicator=True,
    )

    votes_not_merged = votes_not_merged[votes_not_merged["_merge"] == "left_only"]

    if votes_not_merged.empty:
        return

    print()
    print(f"[WARN] positive vote rows not matched to dataset for {prefix}:")
    print(
        votes_not_merged[
            [
                "SEASON_END_YEAR",
                f"{prefix}_VOTE_PLAYER_NAME",
                "player_key",
                f"{prefix}_VOTE_SCORE",
            ]
        ]
        .sort_values(
            ["SEASON_END_YEAR", f"{prefix}_VOTE_SCORE"],
            ascending=[True, False],
        )
        .head(80)
        .to_string(index=False)
    )


def add_vote_targets_to_dataset(
    dataset_path: Path,
    votes_path: Path,
    output_path: Path,
    prefix: str,
    award: str,
) -> None:
    df = pd.read_csv(dataset_path)
    votes = pd.read_csv(votes_path)

    df = prepare_dataset_keys(df, dataset_path)
    votes = prepare_votes_for_merge(votes, prefix, award)
    df = drop_old_vote_cols(df, prefix)

    merged = df.merge(
        votes,
        on=["SEASON_END_YEAR", "player_key"],
        how="left",
        validate="many_to_one",
    )

    merged = fill_vote_columns(merged, prefix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print_vote_merge_check(merged, votes, output_path, prefix)


def run_award(args: argparse.Namespace, award: str) -> None:
    config = get_award_config(award)
    votes_path = config["votes_path"]
    dataset_path = config["dataset_path"]
    output_path = config["output_path"]
    prefix = config["prefix"]

    if not args.skip_fetch:
        votes = fetch_bref_votes(
            start_year=args.start_year,
            end_year=args.end_year,
            award=award,
            sleep_seconds=args.sleep,
        )

        if votes.empty:
            print(f"[WARN] no BRef voting rows fetched for {award}")
        else:
            votes.to_csv(votes_path, index=False)
            print(f"[saved] {votes_path}")

    if args.skip_merge:
        return

    if not dataset_path.exists():
        print(f"[skip] merge for {award}: missing {dataset_path}")
        return

    if not votes_path.exists():
        print(f"[skip] merge for {award}: missing {votes_path}")
        return

    add_vote_targets_to_dataset(
        dataset_path=dataset_path,
        votes_path=votes_path,
        output_path=output_path,
        prefix=prefix,
        award=award,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument(
        "--award",
        choices=["all_nba", "all_rookie", "all"],
        default="all_nba",
    )
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    awards = AWARDS if args.award == "all" else [args.award]

    for award in awards:
        run_award(args, award)


if __name__ == "__main__":
    main()