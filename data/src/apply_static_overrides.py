from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = DATA_DIR / "static"
PROCESSED_DIR = DATA_DIR / "processed"

ELIGIBILITY_PATH = STATIC_DIR / "award_eligibility_overrides.csv"
ALL_STAR_STATIC_PATH = STATIC_DIR / "official_all_star_rosters.csv"

PROCESSED_FILES = [
    PROCESSED_DIR / "player_seasons_labeled.csv",
    PROCESSED_DIR / "all_nba_dataset.csv",
    PROCESSED_DIR / "all_rookie_dataset.csv",
]


def clean_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name)
    text = text.replace("\xa0", " ")
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("*", "").replace("^", "").replace("†", "")
    text = re.sub(r"\([^)]*\)", "", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_player_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "PLAYER_NAME_KEY" not in df.columns:
        if "PLAYER_NAME" not in df.columns:
            raise RuntimeError("Dataset has neither PLAYER_NAME_KEY nor PLAYER_NAME")
        df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].map(clean_name)
    else:
        # Normalize again into a helper because older files may use slightly different key conventions.
        if "PLAYER_NAME" in df.columns:
            df["__STATIC_PLAYER_KEY__"] = df["PLAYER_NAME"].map(clean_name)
        else:
            df["__STATIC_PLAYER_KEY__"] = df["PLAYER_NAME_KEY"].map(clean_name)
    if "__STATIC_PLAYER_KEY__" not in df.columns:
        df["__STATIC_PLAYER_KEY__"] = df["PLAYER_NAME_KEY"].map(clean_name)
    return df


def load_eligibility_overrides() -> pd.DataFrame:
    if not ELIGIBILITY_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(ELIGIBILITY_PATH)
    required = ["SEASON_END_YEAR", "PLAYER_NAME", "AWARD_TYPE", "IS_ELIGIBLE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {ELIGIBILITY_PATH}: {missing}")

    df = df.copy()
    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df["AWARD_TYPE"] = df["AWARD_TYPE"].astype(str).str.lower().str.strip()
    df["IS_ELIGIBLE"] = pd.to_numeric(df["IS_ELIGIBLE"], errors="coerce").fillna(1).astype(int)
    df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].map(clean_name)
    if "REASON" not in df.columns:
        df["REASON"] = ""
    if "SOURCE" not in df.columns:
        df["SOURCE"] = ""
    return df


def apply_eligibility_to_df(df: pd.DataFrame, path: Path, overrides: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty or overrides.empty:
        return df, 0

    # Only All-NBA eligibility is currently used by the final solution.
    nba_overrides = overrides[overrides["AWARD_TYPE"] == "all_nba"].copy()
    if nba_overrides.empty:
        return df, 0

    df = ensure_player_key(df)

    if "IS_ALL_NBA_ELIGIBLE" not in df.columns:
        df["IS_ALL_NBA_ELIGIBLE"] = 1
    df["IS_ALL_NBA_ELIGIBLE"] = pd.to_numeric(df["IS_ALL_NBA_ELIGIBLE"], errors="coerce").fillna(1).astype(int)

    if "ALL_NBA_ELIGIBILITY_REASON" not in df.columns:
        df["ALL_NBA_ELIGIBILITY_REASON"] = ""
    df["ALL_NBA_ELIGIBILITY_REASON"] = df["ALL_NBA_ELIGIBILITY_REASON"].fillna("").astype(str)

    if "ALL_NBA_ELIGIBILITY_SOURCE" not in df.columns:
        df["ALL_NBA_ELIGIBILITY_SOURCE"] = ""
    df["ALL_NBA_ELIGIBILITY_SOURCE"] = df["ALL_NBA_ELIGIBILITY_SOURCE"].fillna("").astype(str)

    changes = 0
    for row in nba_overrides.itertuples(index=False):
        mask = (
            (df["SEASON_END_YEAR"].astype(int) == int(row.SEASON_END_YEAR))
            & (df["__STATIC_PLAYER_KEY__"] == row.PLAYER_NAME_KEY)
        )
        if not mask.any():
            continue
        df.loc[mask, "IS_ALL_NBA_ELIGIBLE"] = int(row.IS_ELIGIBLE)
        df.loc[mask, "ALL_NBA_ELIGIBILITY_REASON"] = str(row.REASON)
        df.loc[mask, "ALL_NBA_ELIGIBILITY_SOURCE"] = str(row.SOURCE)
        changes += int(mask.sum())

    df = df.drop(columns=["__STATIC_PLAYER_KEY__"], errors="ignore")
    return df, changes


def load_static_all_star_roster() -> pd.DataFrame:
    if not ALL_STAR_STATIC_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(ALL_STAR_STATIC_PATH)
    required = ["SEASON_END_YEAR", "PLAYER_NAME"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {ALL_STAR_STATIC_PATH}: {missing}")

    df = df.copy()
    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].map(clean_name)
    return df.drop_duplicates(["SEASON_END_YEAR", "PLAYER_NAME_KEY"])


def apply_static_all_stars_to_df(df: pd.DataFrame, static_all_stars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty or static_all_stars.empty or "PLAYER_NAME" not in df.columns:
        return df, 0

    df = ensure_player_key(df)
    if "IS_ALL_STAR_THIS_SEASON" not in df.columns:
        df["IS_ALL_STAR_THIS_SEASON"] = 0
    df["IS_ALL_STAR_THIS_SEASON"] = pd.to_numeric(df["IS_ALL_STAR_THIS_SEASON"], errors="coerce").fillna(0).astype(int)

    changes = 0
    for row in static_all_stars.itertuples(index=False):
        mask = (
            (df["SEASON_END_YEAR"].astype(int) == int(row.SEASON_END_YEAR))
            & (df["__STATIC_PLAYER_KEY__"] == row.PLAYER_NAME_KEY)
        )
        if not mask.any():
            continue
        before = df.loc[mask, "IS_ALL_STAR_THIS_SEASON"].copy()
        df.loc[mask, "IS_ALL_STAR_THIS_SEASON"] = 1
        changes += int((before != 1).sum())

    df = df.drop(columns=["__STATIC_PLAYER_KEY__"], errors="ignore")
    return df, changes


def apply_to_file(path: Path, eligibility: pd.DataFrame, static_all_stars: pd.DataFrame) -> None:
    if not path.exists():
        print(f"[skip] missing {path}")
        return

    df = pd.read_csv(path)
    original_shape = df.shape

    df, elig_changes = apply_eligibility_to_df(df, path, eligibility)
    df, star_changes = apply_static_all_stars_to_df(df, static_all_stars)

    df.to_csv(path, index=False)
    print(
        f"[static] {path} shape={original_shape} "
        f"eligibility_rows={elig_changes} allstar_rows={star_changes}"
    )


def main() -> None:
    eligibility = load_eligibility_overrides()
    static_all_stars = load_static_all_star_roster()

    print("=" * 80)
    print("APPLY STATIC MANUAL/OFFICIAL OVERRIDES")
    print("=" * 80)
    print(f"eligibility_overrides: {len(eligibility)} rows from {ELIGIBILITY_PATH if ELIGIBILITY_PATH.exists() else 'missing'}")
    print(f"static_all_stars:      {len(static_all_stars)} rows from {ALL_STAR_STATIC_PATH if ALL_STAR_STATIC_PATH.exists() else 'missing'}")

    for path in PROCESSED_FILES:
        apply_to_file(path, eligibility, static_all_stars)


if __name__ == "__main__":
    main()
