from __future__ import annotations

from pathlib import Path
import re
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = DATA_DIR / "static"
OFFICIAL_ALL_STAR_ROSTERS_PATH = STATIC_DIR / "official_all_star_rosters.csv"

ALL_STAR_COLUMNS = [
    "SEASON_END_YEAR",
    "PLAYER_NAME",
    "PLAYER_NAME_KEY",
    "IS_ALL_STAR_THIS_SEASON",
]

MIN_REASONABLE_ALL_STAR_PLAYERS = 20

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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
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
    return apply_name_alias(normalize_name_key(name))


def _html_with_commented_tables(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    comment_parts = []
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        text = str(comment)
        if "<table" in text or "/players/" in text:
            comment_parts.append(text)
    if comment_parts:
        return html + "\n" + "\n".join(comment_parts)
    return html


def _extract_player_names_from_bref_html(html: str) -> list[str]:
    html = _html_with_commented_tables(html)
    soup = BeautifulSoup(html, "html.parser")
    names = set()

    for a in soup.find_all("a"):
        href = a.get("href", "")
        text = clean_player_name(a.get_text(" ", strip=True))
        if not text:
            continue
        if not re.match(r"^/players/[a-z]/[a-z0-9]+\.html$", href):
            continue
        if text in {"Player", "Starters", "Reserves", "Team Totals", "Did Not Play"}:
            continue
        names.add(text)

    return sorted(names)


def _extract_player_names_from_nba_allstar_roster_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    names: list[str] = []
    seen = set()

    pattern = re.compile(
        r"^#\d+\s*\|\s*(?:Guard|Frontcourt)\s+(.+?)\s+PTS\s+[-0-9.]",
        flags=re.IGNORECASE,
    )

    for line in lines:
        match = pattern.search(line)
        if not match:
            continue

        name = match.group(1)
        name = re.sub(r"\s+Injury replacement for .*$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+Injured, will not play.*$", "", name, flags=re.IGNORECASE)
        name = clean_player_name(name)
        key = make_player_name_key(name)

        if not name or key in seen:
            continue
        seen.add(key)
        names.append(name)

    return names


def _rows_from_names(season_end_year: int, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        rows.append(
            {
                "SEASON_END_YEAR": int(season_end_year),
                "PLAYER_NAME": clean_player_name(name),
                "PLAYER_NAME_KEY": make_player_name_key(name),
                "IS_ALL_STAR_THIS_SEASON": 1,
            }
        )
    return pd.DataFrame(rows, columns=ALL_STAR_COLUMNS)


def load_static_all_star_rosters(path: Path = OFFICIAL_ALL_STAR_ROSTERS_PATH) -> pd.DataFrame:
    """Load official roster snapshots used as reproducibility/manual data.

    The file is intentionally stored in data/static instead of data/raw, because it
    is a small official snapshot/override and should survive deleting downloaded
    raw caches. It is not a target label.
    """
    if not path.exists():
        return pd.DataFrame(columns=ALL_STAR_COLUMNS)

    df = pd.read_csv(path)
    required = ["SEASON_END_YEAR", "PLAYER_NAME"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")

    df = df[required].copy()
    df["SEASON_END_YEAR"] = pd.to_numeric(df["SEASON_END_YEAR"], errors="coerce")
    df = df[df["SEASON_END_YEAR"].notna()].copy()
    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(clean_player_name)
    df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].apply(make_player_name_key)
    df["IS_ALL_STAR_THIS_SEASON"] = 1
    df = df.drop_duplicates(["SEASON_END_YEAR", "PLAYER_NAME_KEY"], keep="last")
    return df[ALL_STAR_COLUMNS].sort_values(["SEASON_END_YEAR", "PLAYER_NAME_KEY"]).reset_index(drop=True)


def fetch_all_star_players_for_year_bref(season_end_year: int, timeout: int = 30) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/allstar/NBA_{season_end_year}.html"
    print(f"Fetching All-Star players from B-Ref: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
    except Exception as exc:
        print(f"  ERROR request failed for {season_end_year}: {exc}")
        return pd.DataFrame(columns=ALL_STAR_COLUMNS)

    if response.status_code != 200:
        print(f"  WARNING status={response.status_code} for {season_end_year}; skipping B-Ref")
        return pd.DataFrame(columns=ALL_STAR_COLUMNS)

    names = _extract_player_names_from_bref_html(response.text)
    df = _rows_from_names(season_end_year, names)
    print(f"  B-Ref players found: {len(df)}")
    return df


def fetch_all_star_players_for_year_nba_com(season_end_year: int, timeout: int = 30) -> pd.DataFrame:
    url = f"https://www.nba.com/allstar/{season_end_year}/roster"
    print(f"Fetching All-Star players from NBA.com: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
    except Exception as exc:
        print(f"  ERROR NBA.com request failed for {season_end_year}: {exc}")
        return pd.DataFrame(columns=ALL_STAR_COLUMNS)

    if response.status_code != 200:
        print(f"  WARNING NBA.com status={response.status_code} for {season_end_year}")
        return pd.DataFrame(columns=ALL_STAR_COLUMNS)

    names = _extract_player_names_from_nba_allstar_roster_html(response.text)
    df = _rows_from_names(season_end_year, names)
    print(f"  NBA.com players found: {len(df)}")
    return df


def static_roster_for_year(season_end_year: int) -> pd.DataFrame:
    static = load_static_all_star_rosters()
    if static.empty:
        return pd.DataFrame(columns=ALL_STAR_COLUMNS)
    return static[static["SEASON_END_YEAR"].astype(int) == int(season_end_year)].copy()


def ensure_static_roster_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    static = load_static_all_star_rosters()
    parts = [df]

    if not static.empty:
        for year, group in static.groupby("SEASON_END_YEAR"):
            current_count = 0
            if not df.empty and "SEASON_END_YEAR" in df.columns:
                current_count = int((df["SEASON_END_YEAR"].astype(int) == int(year)).sum())
            if current_count < len(group):
                print(
                    f"Applying static official All-Star roster snapshot for {int(year)}: "
                    f"current={current_count}, static={len(group)}"
                )
        parts.append(static)

    if parts:
        df = pd.concat(parts, ignore_index=True)

    if df.empty:
        return pd.DataFrame(columns=ALL_STAR_COLUMNS)

    for col in ALL_STAR_COLUMNS:
        if col not in df.columns:
            df[col] = 1 if col == "IS_ALL_STAR_THIS_SEASON" else ""

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(clean_player_name)
    df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].apply(make_player_name_key)
    df["IS_ALL_STAR_THIS_SEASON"] = 1

    df = df.drop_duplicates(
        subset=["SEASON_END_YEAR", "PLAYER_NAME_KEY"],
        keep="last",
    )

    return df[ALL_STAR_COLUMNS].sort_values(
        ["SEASON_END_YEAR", "PLAYER_NAME_KEY"]
    ).reset_index(drop=True)


def fetch_all_star_players(
    min_year: int,
    max_year: int,
    sleep_sec: float = 3.0,
) -> pd.DataFrame:
    all_rows = []

    for season_end_year in range(int(min_year), int(max_year) + 1):
        season_df = fetch_all_star_players_for_year_bref(season_end_year)

        if len(season_df) < MIN_REASONABLE_ALL_STAR_PLAYERS:
            nba_df = fetch_all_star_players_for_year_nba_com(season_end_year)
            if len(nba_df) > len(season_df):
                season_df = nba_df

        if len(season_df) < MIN_REASONABLE_ALL_STAR_PLAYERS:
            static_df = static_roster_for_year(season_end_year)
            if len(static_df) > len(season_df):
                print(f"  using static official roster snapshot for {season_end_year}: {len(static_df)} players")
                season_df = static_df

        if not season_df.empty:
            all_rows.append(season_df)

        time.sleep(sleep_sec)

    if not all_rows:
        df = pd.DataFrame(columns=ALL_STAR_COLUMNS)
    else:
        df = pd.concat(all_rows, ignore_index=True)

    return ensure_static_roster_rows(df)


def load_or_fetch_all_star_players(
    raw_dir: Path,
    min_year: int,
    max_year: int,
    filename: str = "all_star_players.csv",
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / filename

    if path.exists():
        cached = pd.read_csv(path)
        cached = ensure_static_roster_rows(cached)
        cached.to_csv(path, index=False)

        if not cached.empty:
            print(f"Loaded All-Star players: {path} shape={cached.shape}")
            return cached

        print(f"Existing All-Star file is empty, refetching: {path}")

    df = fetch_all_star_players(min_year=min_year, max_year=max_year)
    df = ensure_static_roster_rows(df)
    df.to_csv(path, index=False)
    print(f"Saved All-Star players: {path} shape={df.shape}")
    return df


def add_all_star_features(
    df: pd.DataFrame,
    all_star_players: pd.DataFrame | None,
) -> pd.DataFrame:
    df = df.copy()

    if "PLAYER_NAME_KEY" not in df.columns:
        df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].apply(make_player_name_key)

    key_col = "PLAYER_NAME_KEY"

    if all_star_players is None or all_star_players.empty:
        all_star_players = load_static_all_star_rosters()

    if all_star_players is None or all_star_players.empty:
        df["IS_ALL_STAR_THIS_SEASON"] = 0
        df["PREV_ALL_STAR"] = 0
        df["ALL_STAR_SELECTIONS_BEFORE_SEASON"] = 0
        return df

    all_star_players = ensure_static_roster_rows(all_star_players)

    marks = all_star_players[
        ["SEASON_END_YEAR", "PLAYER_NAME_KEY", "IS_ALL_STAR_THIS_SEASON"]
    ].drop_duplicates(
        subset=["SEASON_END_YEAR", "PLAYER_NAME_KEY"],
        keep="first",
    )

    df = df.drop(
        columns=[
            col
            for col in [
                "IS_ALL_STAR_THIS_SEASON",
                "PREV_ALL_STAR",
                "ALL_STAR_SELECTIONS_BEFORE_SEASON",
            ]
            if col in df.columns
        ],
        errors="ignore",
    )

    df = df.merge(marks, on=["SEASON_END_YEAR", key_col], how="left")

    df["IS_ALL_STAR_THIS_SEASON"] = (
        df["IS_ALL_STAR_THIS_SEASON"].fillna(0).astype(int)
    )

    player_season = (
        df[[key_col, "SEASON_END_YEAR", "IS_ALL_STAR_THIS_SEASON"]]
        .drop_duplicates(subset=[key_col, "SEASON_END_YEAR"])
        .sort_values([key_col, "SEASON_END_YEAR"])
        .reset_index(drop=True)
    )

    player_season["PREV_ALL_STAR"] = (
        player_season.groupby(key_col)["IS_ALL_STAR_THIS_SEASON"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    player_season["ALL_STAR_SELECTIONS_BEFORE_SEASON"] = (
        player_season.groupby(key_col)["IS_ALL_STAR_THIS_SEASON"].cumsum()
        - player_season["IS_ALL_STAR_THIS_SEASON"]
    ).astype(int)

    df = df.merge(
        player_season[
            [
                key_col,
                "SEASON_END_YEAR",
                "PREV_ALL_STAR",
                "ALL_STAR_SELECTIONS_BEFORE_SEASON",
            ]
        ],
        on=[key_col, "SEASON_END_YEAR"],
        how="left",
    )

    df["PREV_ALL_STAR"] = df["PREV_ALL_STAR"].fillna(0).astype(int)
    df["ALL_STAR_SELECTIONS_BEFORE_SEASON"] = (
        df["ALL_STAR_SELECTIONS_BEFORE_SEASON"].fillna(0).astype(int)
    )

    return df


if __name__ == "__main__":
    RAW_DIR = PROJECT_ROOT / "data" / "raw"

    df = load_or_fetch_all_star_players(
        raw_dir=RAW_DIR,
        min_year=2000,
        max_year=2026,
    )

    print()
    print(df.tail(40).to_string(index=False))

    if not df.empty:
        print()
        print(df.groupby("SEASON_END_YEAR").size().tail(30))
