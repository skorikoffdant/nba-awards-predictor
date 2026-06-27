from pathlib import Path
import re
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEBUG_DIR = RAW_DIR / "debug_awards"

RAW_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

MIN_SEASON = 2000
MAX_TRAIN_SEASON = 2025

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

ALL_NBA_URL = "https://www.nba.com/news/history-all-nba-teams"
ALL_ROOKIE_URL = "https://www.nba.com/news/history-all-rookie-teams"

ALL_NBA_LABEL_MAP = {
    "FIRST TEAM": 3,
    "SECOND TEAM": 2,
    "THIRD TEAM": 1,
}

ALL_ROOKIE_LABEL_MAP = {
    "FIRST TEAM": 2,
    "SECOND TEAM": 1,
}

NOISE_PATTERNS = [
    "official release",
    "voting totals",
    "year-by-year",
    "take a look",
    "nba history",
    "related",
    "latest",
    "stats & records",
    "awards",
    "legends profiles",
    "updated on",
    "note:",
    "history:",
    "image",
    "complete season recaps",
    "list of nba champions",
    "team-by-team championships",
    "all-star",
    "hall of fame",
    "league pass",
    "privacy policy",
    "terms of use",
]


def download_page_lines(url: str, debug_name: str) -> list[str]:
    print(f"Downloading: {url}")

    response = requests.get(url, headers=HEADERS, timeout=40)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download {url}. Status code: {response.status_code}"
        )

    time.sleep(2)

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")

    raw_debug_path = DEBUG_DIR / f"{debug_name}_raw_text.txt"
    raw_debug_path.write_text(text, encoding="utf-8")

    lines = []

    for line in text.splitlines():
        line = line.replace("\xa0", " ")
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            lines.append(line)

    clean_debug_path = DEBUG_DIR / f"{debug_name}_clean_lines.txt"
    clean_debug_path.write_text("\n".join(lines), encoding="utf-8")

    return lines


def season_to_end_year(season_text: str) -> int | None:
    text = str(season_text).strip()
    text = text.lstrip(">").strip()

    match = re.search(r"(\d{4})-(\d{2})", text)

    if not match:
        return None

    start_year = int(match.group(1))
    end_short = int(match.group(2))

    century = start_year // 100 * 100
    end_year = century + end_short

    if end_year <= start_year:
        end_year += 100

    return end_year


def is_season_line(line: str) -> bool:
    text = str(line).strip()
    text = text.lstrip(">").strip()

    return re.fullmatch(r"\d{4}-\d{2}", text) is not None


def normalize_team_header(line: str) -> str:
    text = str(line).strip().upper()
    text = text.replace(":", "")
    text = re.sub(r"\s+", " ", text)

    return text


def get_team_label(line: str, team_label_map: dict[str, int]) -> int | None:
    header = normalize_team_header(line)

    return team_label_map.get(header)


def clean_player_name(text: str) -> str:
    text = str(text)

    text = text.replace("\xa0", " ")
    text = text.replace("•", " ")
    text = text.replace("·", " ")

    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"^\s*[FGC]\s*:\s*", "", text, flags=re.IGNORECASE)

    text = text.split(",")[0]

    text = re.sub(r"\(tie\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*\)", "", text)

    text = text.replace("*", "")
    text = text.replace("^", "")
    text = text.replace("†", "")

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name_key(name: str) -> str:
    name = clean_player_name(name)

    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def is_noise_line(line: str) -> bool:
    low = str(line).lower()

    return any(pattern in low for pattern in NOISE_PATTERNS)


def looks_like_player_line(line: str) -> bool:
    line = str(line).strip()

    if not line:
        return False

    if is_noise_line(line):
        return False

    if is_season_line(line):
        return False

    if get_team_label(line, ALL_NBA_LABEL_MAP) is not None:
        return False

    if "," not in line:
        return False

    player_name = clean_player_name(line)

    if not player_name:
        return False

    if len(player_name.split()) < 2:
        return False

    bad_words = [
        "official",
        "release",
        "voting",
        "history",
        "team",
        "season",
        "awards",
    ]

    low_name = player_name.lower()

    return not any(word in low_name for word in bad_words)


def parse_award_page(
    url: str,
    debug_name: str,
    team_label_map: dict[str, int],
    output_label_col: str,
) -> pd.DataFrame:
    lines = download_page_lines(url, debug_name)

    rows = []
    current_season = None
    current_label = None

    for line in lines:
        if is_season_line(line):
            current_season = season_to_end_year(line)
            current_label = None
            continue

        label = get_team_label(line, team_label_map)

        if label is not None:
            current_label = label
            continue

        if current_season is None:
            continue

        if current_label is None:
            continue

        if not (MIN_SEASON <= current_season <= MAX_TRAIN_SEASON):
            continue

        if not looks_like_player_line(line):
            continue

        player_name = clean_player_name(line)

        rows.append(
            {
                "SEASON_END_YEAR": current_season,
                "PLAYER_NAME": player_name,
                output_label_col: current_label,
                "PLAYER_NAME_KEY": normalize_name_key(player_name),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        debug_path = DEBUG_DIR / f"{debug_name}_clean_lines.txt"
        raise RuntimeError(
            f"No labels parsed from {url}. "
            f"Check debug file: {debug_path}"
        )

    return df.drop_duplicates()


def add_known_label_fixes(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    fixes = []

    if label_col == "ALL_NBA_LABEL":
        fixes.append(
            {
                "SEASON_END_YEAR": 1999,
                "PLAYER_NAME": "Gary Payton",
                "ALL_NBA_LABEL": 2,
                "PLAYER_NAME_KEY": normalize_name_key("Gary Payton"),
            }
        )

    if not fixes:
        return df

    fixes_df = pd.DataFrame(fixes)

    df = pd.concat([df, fixes_df], ignore_index=True)
    df = df.drop_duplicates(
        subset=["SEASON_END_YEAR", "PLAYER_NAME_KEY", label_col],
        keep="first",
    )

    return df


def fetch_all_nba_labels() -> pd.DataFrame:
    return parse_award_page(
        url=ALL_NBA_URL,
        debug_name="all_nba",
        team_label_map=ALL_NBA_LABEL_MAP,
        output_label_col="ALL_NBA_LABEL",
    )


def fetch_all_rookie_labels() -> pd.DataFrame:
    return parse_award_page(
        url=ALL_ROOKIE_URL,
        debug_name="all_rookie",
        team_label_map=ALL_ROOKIE_LABEL_MAP,
        output_label_col="ALL_ROOKIE_LABEL",
    )


def print_count_check(
    df: pd.DataFrame,
    label_col: str,
    title: str,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print()
    print("Rows:", len(df))
    print("Seasons:", df["SEASON_END_YEAR"].min(), "-", df["SEASON_END_YEAR"].max())

    print()
    print("Counts by season and label:")

    counts = (
        df.groupby(["SEASON_END_YEAR", label_col])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    print(counts.tail(35))

    print()
    print("Total players per season:")

    totals = df.groupby("SEASON_END_YEAR").size().sort_index()
    print(totals.tail(35))

    suspicious = totals[(totals < 5) | (totals > 20)]

    if not suspicious.empty:
        print()
        print("WARNING: suspicious season totals:")
        print(suspicious)


def main() -> None:
    all_nba = fetch_all_nba_labels()
    all_rookie = fetch_all_rookie_labels()

    all_nba = add_known_label_fixes(all_nba, "ALL_NBA_LABEL")
    all_rookie = add_known_label_fixes(all_rookie, "ALL_ROOKIE_LABEL")

    all_nba_path = RAW_DIR / "all_nba_labels.csv"
    all_rookie_path = RAW_DIR / "all_rookie_labels.csv"

    all_nba.to_csv(all_nba_path, index=False)
    all_rookie.to_csv(all_rookie_path, index=False)

    print()
    print(f"Saved: {all_nba_path}")
    print(f"Saved: {all_rookie_path}")

    print_count_check(
        all_nba,
        label_col="ALL_NBA_LABEL",
        title="ALL-NBA LABELS CHECK",
    )

    print_count_check(
        all_rookie,
        label_col="ALL_ROOKIE_LABEL",
        title="ALL-ROOKIE LABELS CHECK",
    )


if __name__ == "__main__":
    main()