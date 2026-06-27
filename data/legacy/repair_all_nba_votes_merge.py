from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

try:
    import ftfy
except ImportError:
    ftfy = None


RAW_VOTES_PATH = Path("data/raw/all_nba_voting.csv")
DATASET_PATH = Path("data/processed/all_nba_dataset.csv")

FIXED_VOTES_PATH = Path("data/raw/all_nba_voting_fixed.csv")
OUTPUT_PATH = Path("data/processed/all_nba_dataset_with_votes.csv")


def fix_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if ftfy is not None:
        text = ftfy.fix_text(text)

    text = text.replace("*", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""

    name = fix_text(name)
    name = name.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower()
    name = re.sub(r"[^a-z0-9' .-]", "", name)
    name = " ".join(name.split())

    return name


def to_number(value) -> float:
    if pd.isna(value):
        return 0.0

    text = str(value).strip()
    text = text.replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return 0.0

    return float(match.group(0))


def hard_alias_key(player_name: str, key: str) -> str:
    name = normalize_name(player_name)

    alias = {
        "peja stojakovia": "peja stojakovic",
        "peja stojakovi": "peja stojakovic",
        "manu gina3bili": "manu ginobili",
        "manu ginbili": "manu ginobili",
        "nenad krstia": "nenad krstic",
        "nenad krsti": "nenad krstic",
        "nenaa": "nene",
        "goran dragia": "goran dragic",
        "goran dragi": "goran dragic",
        "nikola jokia": "nikola jokic",
        "nikola joki": "nikola jokic",
        "kristaps porziaais": "kristaps porzingis",
        "kristaps porziis": "kristaps porzingis",
        "luka donaia": "luka doncic",
        "luka donai": "luka doncic",
        "nikola vuaevia": "nikola vucevic",
        "nikola vuevi": "nikola vucevic",
    }

    if key in alias:
        return alias[key]

    if "peja" in name or "stojak" in name:
        return "peja stojakovic"

    if "manu" in name or "gin" in name:
        return "manu ginobili"

    if "nenad" in name or "krsti" in name:
        return "nenad krstic"

    if name in {"nene", "nene hilario"} or "nen" in name:
        return "nene"

    if "goran" in name or "dragi" in name:
        return "goran dragic"

    if "nikola" in name and ("jok" in name or "joki" in name):
        return "nikola jokic"

    if "kristaps" in name or "porzi" in name:
        return "kristaps porzingis"

    if "luka" in name or "don" in name:
        return "luka doncic"

    if "nikola" in name and ("vu" in name or "vua" in name):
        return "nikola vucevic"

    if "hedo" in name or "turko" in name or "turk" in name:
        return "hedo turkoglu"

    if "alperen" in name or "seng" in name:
        return "alperen sengun"

    return key


def best_dataset_key(year: int, player_name: str, current_key: str, keys_by_year: dict[int, set[str]]) -> str:
    keys = keys_by_year.get(int(year), set())

    if current_key in keys:
        return current_key

    if current_key == "jimmy butler" and "jimmy butler iii" in keys:
        return "jimmy butler iii"

    aliased = hard_alias_key(player_name, current_key)

    if aliased in keys:
        return aliased

    if aliased == "jimmy butler" and "jimmy butler iii" in keys:
        return "jimmy butler iii"

    matches = difflib.get_close_matches(aliased, list(keys), n=1, cutoff=0.84)
    if matches:
        return matches[0]

    matches = difflib.get_close_matches(current_key, list(keys), n=1, cutoff=0.84)
    if matches:
        return matches[0]

    return aliased


def find_name_column(df: pd.DataFrame) -> str:
    candidates = ["PLAYER_NAME", "PLAYER", "player_name", "player", "NAME", "name"]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        key = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        if key in {"playername", "player", "name"}:
            return col

    raise ValueError("No player name column found")


def prepare_votes(votes: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    votes = votes.copy()

    for col in ["PLAYER_NAME", "TEAM_TEXT", "SOURCE", "SOURCE_URL"]:
        if col in votes.columns:
            votes[col] = votes[col].map(fix_text)
        else:
            votes[col] = ""

    if "VOTE_MAX" not in votes.columns:
        votes["VOTE_MAX"] = 0.0

    for col in [
        "FIRST_TEAM_VOTES",
        "SECOND_TEAM_VOTES",
        "THIRD_TEAM_VOTES",
        "VOTE_SCORE",
        "VOTE_MAX",
        "VOTE_SHARE",
        "VOTE_SCORE_SEASON_MAX_NORM",
    ]:
        if col not in votes.columns:
            votes[col] = 0.0
        votes[col] = votes[col].map(to_number)

    votes["SEASON_END_YEAR"] = pd.to_numeric(votes["SEASON_END_YEAR"], errors="coerce")
    votes = votes[votes["SEASON_END_YEAR"].notna()].copy()
    votes["SEASON_END_YEAR"] = votes["SEASON_END_YEAR"].astype(int)

    votes["AWARD"] = "all_nba"
    votes["PLAYER_NAME"] = votes["PLAYER_NAME"].map(fix_text)
    votes["player_key"] = votes["PLAYER_NAME"].map(normalize_name)

    if "player_key" not in dataset.columns:
        name_col = find_name_column(dataset)
        dataset["player_key"] = dataset[name_col].map(normalize_name)
    else:
        dataset["player_key"] = dataset["player_key"].map(normalize_name)

    keys_by_year = (
        dataset.groupby("SEASON_END_YEAR")["player_key"]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )

    votes["player_key_original"] = votes["player_key"]

    votes["player_key"] = votes.apply(
        lambda row: best_dataset_key(
            year=int(row["SEASON_END_YEAR"]),
            player_name=row["PLAYER_NAME"],
            current_key=row["player_key"],
            keys_by_year=keys_by_year,
        ),
        axis=1,
    )

    season_max = votes.groupby("SEASON_END_YEAR")["VOTE_SCORE"].transform("max")
    votes["VOTE_SCORE_SEASON_MAX_NORM"] = (votes["VOTE_SCORE"] / season_max).fillna(0.0)

    missing_share = votes["VOTE_SHARE"].isna() | (votes["VOTE_SHARE"] <= 0)
    votes.loc[missing_share, "VOTE_SHARE"] = votes.loc[missing_share, "VOTE_SCORE_SEASON_MAX_NORM"]

    votes = votes[votes["VOTE_SCORE"] > 0].copy()

    votes = (
        votes.sort_values(
            ["SEASON_END_YEAR", "player_key", "VOTE_SCORE"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["SEASON_END_YEAR", "player_key"], keep="first")
        .copy()
    )

    return votes[
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
            "player_key_original",
            "SOURCE",
            "SOURCE_URL",
        ]
    ].copy()


def merge_votes(dataset: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()

    if "player_key" not in dataset.columns:
        name_col = find_name_column(dataset)
        dataset["player_key"] = dataset[name_col].map(normalize_name)
    else:
        dataset["player_key"] = dataset["player_key"].map(normalize_name)

    prefix = "ALL_NBA"

    old_vote_cols = [
        col
        for col in dataset.columns
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
        dataset = dataset.drop(columns=old_vote_cols)

    votes_for_merge = votes[
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

    votes_for_merge = votes_for_merge.rename(
        columns={
            "PLAYER_NAME": "ALL_NBA_VOTE_PLAYER_NAME",
            "TEAM_TEXT": "ALL_NBA_VOTE_TEAM_TEXT",
            "FIRST_TEAM_VOTES": "ALL_NBA_FIRST_TEAM_VOTES",
            "SECOND_TEAM_VOTES": "ALL_NBA_SECOND_TEAM_VOTES",
            "THIRD_TEAM_VOTES": "ALL_NBA_THIRD_TEAM_VOTES",
            "VOTE_SCORE": "ALL_NBA_VOTE_SCORE",
            "VOTE_MAX": "ALL_NBA_VOTE_MAX",
            "VOTE_SHARE": "ALL_NBA_VOTE_SHARE",
            "VOTE_SCORE_SEASON_MAX_NORM": "ALL_NBA_VOTE_SCORE_SEASON_MAX_NORM",
            "SOURCE": "ALL_NBA_VOTE_SOURCE",
            "SOURCE_URL": "ALL_NBA_VOTE_SOURCE_URL",
        }
    )

    merged = dataset.merge(
        votes_for_merge,
        on=["SEASON_END_YEAR", "player_key"],
        how="left",
        validate="many_to_one",
    )

    numeric_cols = [
        "ALL_NBA_FIRST_TEAM_VOTES",
        "ALL_NBA_SECOND_TEAM_VOTES",
        "ALL_NBA_THIRD_TEAM_VOTES",
        "ALL_NBA_VOTE_SCORE",
        "ALL_NBA_VOTE_MAX",
        "ALL_NBA_VOTE_SHARE",
        "ALL_NBA_VOTE_SCORE_SEASON_MAX_NORM",
    ]

    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    text_cols = [
        "ALL_NBA_VOTE_PLAYER_NAME",
        "ALL_NBA_VOTE_TEAM_TEXT",
        "ALL_NBA_VOTE_SOURCE",
        "ALL_NBA_VOTE_SOURCE_URL",
    ]

    for col in text_cols:
        merged[col] = merged[col].fillna("")

    merged["ALL_NBA_RECEIVED_VOTES"] = (merged["ALL_NBA_VOTE_SCORE"] > 0).astype(int)

    return merged


def main() -> None:
    if not RAW_VOTES_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_VOTES_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}")

    votes = pd.read_csv(RAW_VOTES_PATH)
    dataset = pd.read_csv(DATASET_PATH)

    fixed_votes = prepare_votes(votes, dataset)
    fixed_votes.to_csv(FIXED_VOTES_PATH, index=False)

    merged = merge_votes(dataset, fixed_votes)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"[saved] {FIXED_VOTES_PATH}")
    print(f"[saved] {OUTPUT_PATH}")
    print(f"shape: {merged.shape}")

    print("\ncoverage:")
    print(
        merged.groupby("SEASON_END_YEAR")["ALL_NBA_RECEIVED_VOTES"]
        .sum()
        .astype(int)
        .to_string()
    )

    dataset_keys = (
        dataset.assign(player_key=dataset["player_key"].map(normalize_name))
        .groupby("SEASON_END_YEAR")["player_key"]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )

    unmatched = fixed_votes[
        ~fixed_votes.apply(
            lambda row: row["player_key"] in dataset_keys.get(int(row["SEASON_END_YEAR"]), set()),
            axis=1,
        )
    ].copy()

    if not unmatched.empty:
        print("\nunmatched after repair:")
        print(
            unmatched[
                [
                    "SEASON_END_YEAR",
                    "PLAYER_NAME",
                    "player_key_original",
                    "player_key",
                    "VOTE_SCORE",
                ]
            ]
            .sort_values(["SEASON_END_YEAR", "VOTE_SCORE"], ascending=[True, False])
            .to_string(index=False)
        )
    else:
        print("\nunmatched after repair: 0")


if __name__ == "__main__":
    main()
