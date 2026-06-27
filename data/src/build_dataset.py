from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from all_star_features import (
    add_all_star_features,
    load_or_fetch_all_star_players,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = DATA_DIR / "static"

PLAYER_STATS_PATH = PROCESSED_DIR / "player_seasons_raw.csv"
ALL_NBA_LABELS_PATH = RAW_DIR / "all_nba_labels.csv"
ALL_ROOKIE_LABELS_PATH = RAW_DIR / "all_rookie_labels.csv"
ALL_STAR_PLAYERS_PATH = RAW_DIR / "all_star_players.csv"
BREF_ADVANCED_STATS_PATH = RAW_DIR / "bref_advanced_stats.csv"
AWARD_ELIGIBILITY_OVERRIDES_PATH = STATIC_DIR / "award_eligibility_overrides.csv"

LABELED_OUTPUT_PATH = PROCESSED_DIR / "player_seasons_labeled.csv"
ALL_NBA_OUTPUT_PATH = PROCESSED_DIR / "all_nba_dataset.csv"
ALL_ROOKIE_OUTPUT_PATH = PROCESSED_DIR / "all_rookie_dataset.csv"

MIN_DATA_SEASON = 2000
TARGET_SEASON = 2026

BREF_ADVANCED_FEATURES = [
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

def safe_divide(numerator, denominator):
    numerator = pd.Series(numerator)
    denominator = pd.Series(denominator)

    result = numerator / denominator.replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return result.to_numpy()

def load_player_stats() -> pd.DataFrame:
    if not PLAYER_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PLAYER_STATS_PATH}")

    df = pd.read_csv(PLAYER_STATS_PATH)

    required_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(
                f"Column {col} not found in player stats. "
                f"Available columns: {df.columns.tolist()}"
            )

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df[df["SEASON_END_YEAR"] >= MIN_DATA_SEASON].copy()

    df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(clean_player_name)

    player_name_key = df["PLAYER_NAME"].apply(make_player_name_key)

    df = pd.concat(
        [
            df,
            player_name_key.rename("PLAYER_NAME_KEY"),
        ],
        axis=1,
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def load_bref_advanced_stats(path: Path = BREF_ADVANCED_STATS_PATH) -> pd.DataFrame:
    expected_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
    ] + BREF_ADVANCED_FEATURES

    if not path.exists():
        print()
        print("=" * 80)
        print("WARNING: B-REF ADVANCED STATS NOT FOUND")
        print("=" * 80)
        print(f"Missing file: {path}")
        print("Run: python data/src/fetch_bref_advanced.py")
        print("B-Ref features will be filled with 0 for now.")
        return pd.DataFrame(columns=expected_cols)

    df = pd.read_csv(path)

    if "SEASON_END_YEAR" not in df.columns:
        raise RuntimeError(
            f"Missing SEASON_END_YEAR in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    if "PLAYER_NAME_KEY" not in df.columns:
        if "PLAYER_NAME" not in df.columns:
            raise RuntimeError(
                f"Missing PLAYER_NAME_KEY and PLAYER_NAME in {path}. "
                f"Available columns: {df.columns.tolist()}"
            )

        df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(clean_player_name)
        df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].apply(make_player_name_key)

    if "PLAYER_NAME" not in df.columns:
        df["PLAYER_NAME"] = df["PLAYER_NAME_KEY"]

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df[df["SEASON_END_YEAR"] >= MIN_DATA_SEASON].copy()

    for col in BREF_ADVANCED_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
    ] + BREF_ADVANCED_FEATURES

    df = df[keep_cols]
    df = (
        df.groupby(["SEASON_END_YEAR", "PLAYER_NAME_KEY"], as_index=False)
        .agg({
            "PLAYER_NAME": "first",
            **{col: "max" for col in BREF_ADVANCED_FEATURES},
        })
    )

    return df

def add_bref_advanced_features(
    df: pd.DataFrame,
    bref_advanced_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()

    for col in BREF_ADVANCED_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])

    if bref_advanced_stats is None or bref_advanced_stats.empty:
        for col in BREF_ADVANCED_FEATURES:
            df[col] = 0.0
        return df

    bref = bref_advanced_stats.copy()

    if "PLAYER_NAME_KEY" not in bref.columns:
        bref["PLAYER_NAME_KEY"] = bref["PLAYER_NAME"].apply(make_player_name_key)

    for col in BREF_ADVANCED_FEATURES:
        if col not in bref.columns:
            bref[col] = np.nan
        bref[col] = pd.to_numeric(bref[col], errors="coerce")

    keep_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME_KEY",
    ] + BREF_ADVANCED_FEATURES

    bref = bref[keep_cols]
    bref = (
        bref.groupby(["SEASON_END_YEAR", "PLAYER_NAME_KEY"], as_index=False)
        .max(numeric_only=True)
    )

    df = df.merge(
        bref,
        on=["SEASON_END_YEAR", "PLAYER_NAME_KEY"],
        how="left",
    )

    for col in BREF_ADVANCED_FEATURES:
        df[col] = df[col].fillna(0.0)

    return df

def load_labels(path: Path, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        label_col,
    ]

    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(
                f"Column {col} not found in {path}. "
                f"Available columns: {df.columns.tolist()}"
            )

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df[df["SEASON_END_YEAR"] >= MIN_DATA_SEASON].copy()

    df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(clean_player_name)
    df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].apply(make_player_name_key)
    df[label_col] = df[label_col].astype(int)

    df = df[
        [
            "SEASON_END_YEAR",
            "PLAYER_NAME_KEY",
            label_col,
        ]
    ].drop_duplicates()

    return df

def check_unmatched_labels(
    player_stats: pd.DataFrame,
    labels: pd.DataFrame,
    label_col: str,
) -> None:
    stats_keys = player_stats[
        [
            "SEASON_END_YEAR",
            "PLAYER_NAME_KEY",
        ]
    ].drop_duplicates()

    merged = labels.merge(
        stats_keys,
        on=["SEASON_END_YEAR", "PLAYER_NAME_KEY"],
        how="left",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] == "left_only"].copy()

    print()
    print("=" * 80)
    print(f"UNMATCHED LABELS CHECK: {label_col}")
    print("=" * 80)

    if unmatched.empty:
        print("All labels matched player stats.")
        return

    print(f"Unmatched labels: {len(unmatched)}")
    print(
        unmatched[
            [
                "SEASON_END_YEAR",
                "PLAYER_NAME_KEY",
                label_col,
            ]
        ].head(100)
    )

def add_total_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"MIN", "GP"}.issubset(df.columns):
        df["TOTAL_MIN"] = df["MIN"] * df["GP"]
    else:
        df["TOTAL_MIN"] = 0.0

    if {"PTS", "GP"}.issubset(df.columns):
        df["TOTAL_PTS"] = df["PTS"] * df["GP"]
    else:
        df["TOTAL_PTS"] = 0.0

    if {"REB", "GP"}.issubset(df.columns):
        df["TOTAL_REB"] = df["REB"] * df["GP"]
    else:
        df["TOTAL_REB"] = 0.0

    if {"AST", "GP"}.issubset(df.columns):
        df["TOTAL_AST"] = df["AST"] * df["GP"]
    else:
        df["TOTAL_AST"] = 0.0

    if {"PTS", "REB", "AST"}.issubset(df.columns):
        df["PTS_REB_AST"] = df["PTS"] + df["REB"] + df["AST"]
    else:
        df["PTS_REB_AST"] = 0.0

    if {"STL", "BLK"}.issubset(df.columns):
        df["STOCKS"] = df["STL"] + df["BLK"]
    else:
        df["STOCKS"] = 0.0

    if {"AST", "TOV"}.issubset(df.columns):
        df["AST_TOV_SIMPLE"] = safe_divide(df["AST"], df["TOV"])
    else:
        df["AST_TOV_SIMPLE"] = 0.0

    return df

def add_previous_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    player_key_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PLAYER_NAME_KEY"

    prev_source_cols = [
        "ALL_NBA_LABEL",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "GP",
        "MIN",
        "W_PCT",
        "TS_PCT",
        "USG_PCT",
        "PIE",
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

    prev_source_cols = [
        col for col in prev_source_cols
        if col in df.columns
    ]

    player_season = (
        df[[player_key_col, "SEASON_END_YEAR"] + prev_source_cols]
        .groupby([player_key_col, "SEASON_END_YEAR"], as_index=False)
        .max()
    )

    prev_df = player_season.copy()
    prev_df["SEASON_END_YEAR"] = prev_df["SEASON_END_YEAR"] + 1

    rename_map = {}

    if "ALL_NBA_LABEL" in prev_df.columns:
        rename_map["ALL_NBA_LABEL"] = "PREV_ALL_NBA_LABEL"

    for col in prev_source_cols:
        if col != "ALL_NBA_LABEL":
            rename_map[col] = f"PREV_{col}"

    prev_df = prev_df.rename(columns=rename_map)

    df = df.merge(
        prev_df,
        on=[player_key_col, "SEASON_END_YEAR"],
        how="left",
    )

    if "PREV_ALL_NBA_LABEL" not in df.columns:
        df["PREV_ALL_NBA_LABEL"] = 0.0

    df["PREV_ALL_NBA_LABEL"] = df["PREV_ALL_NBA_LABEL"].fillna(0.0)

    df["PREV_target_all_nba"] = df["PREV_ALL_NBA_LABEL"]

    previous_stat_cols = [
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
        "PREV_PER",
        "PREV_OWS",
        "PREV_DWS",
        "PREV_WS",
        "PREV_WS_PER_48",
        "PREV_OBPM",
        "PREV_DBPM",
        "PREV_BPM",
        "PREV_VORP",
    ]

    existing_prev_stat_cols = [
        col for col in previous_stat_cols
        if col in df.columns
    ]

    if existing_prev_stat_cols:
        df["PLAYED_PREVIOUS_SEASON"] = (
            df[existing_prev_stat_cols]
            .notna()
            .any(axis=1)
            .astype(int)
        )
    else:
        df["PLAYED_PREVIOUS_SEASON"] = 0

    for col in previous_stat_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    df["PREV_WAS_ALL_NBA"] = (df["PREV_ALL_NBA_LABEL"] > 0).astype(float)

    vote_map = {
        0.0: 0.0,
        1.0: 1.0,
        2.0: 3.0,
        3.0: 5.0,
    }

    df["PREV_ALL_NBA_VOTE_SCORE"] = (
        df["PREV_ALL_NBA_LABEL"]
        .map(vote_map)
        .fillna(0.0)
        .astype(float)
    )

    return df

def add_rookie_candidate_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    player_key_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PLAYER_NAME_KEY"

    first_known_season = df.groupby(player_key_col)["SEASON_END_YEAR"].transform("min")

    if "AGE" in df.columns:
        age_ok = df["AGE"] <= 27
    else:
        age_ok = True

    df["IS_ROOKIE_CANDIDATE"] = (
        (df["SEASON_END_YEAR"] == first_known_season)
        & (df["SEASON_END_YEAR"] > MIN_DATA_SEASON)
        & age_ok
    ).astype(int)

    df.loc[df["ALL_ROOKIE_LABEL"] > 0, "IS_ROOKIE_CANDIDATE"] = 1

    return df

def add_rookie_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "IS_ROOKIE_CANDIDATE" in df.columns:
        df["IS_ROOKIE"] = df["IS_ROOKIE_CANDIDATE"].astype(int)
    else:
        player_key_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PLAYER_NAME_KEY"
        first_known_season = df.groupby(player_key_col)["SEASON_END_YEAR"].transform("min")

        if "AGE" in df.columns:
            age_ok = df["AGE"] <= 27
        else:
            age_ok = True

        df["IS_ROOKIE"] = (
            (df["SEASON_END_YEAR"] == first_known_season)
            & (df["SEASON_END_YEAR"] > MIN_DATA_SEASON)
            & age_ok
        ).astype(int)

    if "ALL_ROOKIE_LABEL" in df.columns:
        df.loc[df["ALL_ROOKIE_LABEL"] > 0, "IS_ROOKIE"] = 1

    return df

def add_team_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    player_key_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PLAYER_NAME_KEY"

    if "TEAM_ID" in df.columns:
        team_col = "TEAM_ID"
    elif "TEAM_ABBREVIATION" in df.columns:
        team_col = "TEAM_ABBREVIATION"
    else:
        print("WARNING: no TEAM_ID / TEAM_ABBREVIATION, adding zero team-relative features")

        zero_cols = [
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

        for col in zero_cols:
            df[col] = 0.0

        return df

    for col in ["TOTAL_PTS", "TOTAL_AST", "TOTAL_MIN"]:
        if col not in df.columns:
            df[col] = 0.0

    team_keys = ["SEASON_END_YEAR", team_col]

    player_team_count = (
        df.groupby(["SEASON_END_YEAR", player_key_col])[team_col]
        .transform("nunique")
    )

    df["IS_MULTI_TEAM_PLAYER"] = (player_team_count > 1).astype(int)

    df["TEAM_PLAYER_COUNT"] = (
        df.groupby(team_keys)[player_key_col]
        .transform("nunique")
    )

    df["TEAM_TOTAL_PTS_RANK"] = (
        df.groupby(team_keys)["TOTAL_PTS"]
        .rank(method="min", ascending=False)
    )

    df["TEAM_TOTAL_AST_RANK"] = (
        df.groupby(team_keys)["TOTAL_AST"]
        .rank(method="min", ascending=False)
    )

    df["TEAM_TOTAL_MIN_RANK"] = (
        df.groupby(team_keys)["TOTAL_MIN"]
        .rank(method="min", ascending=False)
    )

    team_total_pts = df.groupby(team_keys)["TOTAL_PTS"].transform("sum")
    team_total_ast = df.groupby(team_keys)["TOTAL_AST"].transform("sum")
    team_total_min = df.groupby(team_keys)["TOTAL_MIN"].transform("sum")

    df["TEAM_TOTAL_PTS_SHARE"] = safe_divide(df["TOTAL_PTS"], team_total_pts)
    df["TEAM_TOTAL_AST_SHARE"] = safe_divide(df["TOTAL_AST"], team_total_ast)
    df["TEAM_TOTAL_MIN_SHARE"] = safe_divide(df["TOTAL_MIN"], team_total_min)

    df["IS_TEAM_TOTAL_PTS_LEADER"] = (
        df["TEAM_TOTAL_PTS_RANK"] == 1
    ).astype(int)

    df["IS_TEAM_TOTAL_MIN_LEADER"] = (
        df["TEAM_TOTAL_MIN_RANK"] == 1
    ).astype(int)

    df["TEAM_TOP3_TOTAL_PTS_FLAG"] = (
        df["TEAM_TOTAL_PTS_RANK"] <= 3
    ).astype(int)

    df["TEAM_TOP3_TOTAL_MIN_FLAG"] = (
        df["TEAM_TOTAL_MIN_RANK"] <= 3
    ).astype(int)

    team_context_agg = {}

    if "W_PCT" in df.columns:
        team_context_agg["W_PCT"] = "max"

    if "NET_RATING" in df.columns:
        team_context_agg["NET_RATING"] = "mean"

    if team_context_agg:
        team_context = (
            df.groupby(team_keys, as_index=False)
            .agg(team_context_agg)
        )

        if "W_PCT" in team_context.columns:
            team_context["LEAGUE_TEAM_W_PCT_RANK"] = (
                team_context.groupby("SEASON_END_YEAR")["W_PCT"]
                .rank(method="min", ascending=False)
            )

        if "NET_RATING" in team_context.columns:
            team_context["LEAGUE_TEAM_NET_RATING_RANK"] = (
                team_context.groupby("SEASON_END_YEAR")["NET_RATING"]
                .rank(method="min", ascending=False)
            )

        keep_cols = team_keys + [
            col for col in [
                "LEAGUE_TEAM_W_PCT_RANK",
                "LEAGUE_TEAM_NET_RATING_RANK",
            ]
            if col in team_context.columns
        ]

        df = df.merge(
            team_context[keep_cols],
            on=team_keys,
            how="left",
        )

    if "LEAGUE_TEAM_W_PCT_RANK" not in df.columns:
        df["LEAGUE_TEAM_W_PCT_RANK"] = 0.0

    if "LEAGUE_TEAM_NET_RATING_RANK" not in df.columns:
        df["LEAGUE_TEAM_NET_RATING_RANK"] = 0.0

    df["LEAGUE_TEAM_W_PCT_RANK"] = df["LEAGUE_TEAM_W_PCT_RANK"].fillna(0.0)
    df["LEAGUE_TEAM_NET_RATING_RANK"] = df["LEAGUE_TEAM_NET_RATING_RANK"].fillna(0.0)

    return df

def add_engineered_features(
    df: pd.DataFrame,
    all_star_players: pd.DataFrame | None = None,
    bref_advanced_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()

    df = add_total_features(df)
    df = add_bref_advanced_features(df, bref_advanced_stats)
    df = add_previous_features(df)
    df = add_rookie_feature(df)
    df = add_team_relative_features(df)
    df = add_all_star_features(df, all_star_players)

    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def load_award_eligibility_overrides(
    path: Path = AWARD_ELIGIBILITY_OVERRIDES_PATH,
) -> pd.DataFrame:
    expected_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "AWARD_TYPE",
        "IS_ELIGIBLE",
        "REASON",
        "SOURCE_URL",
    ]

    if not path.exists():
        return pd.DataFrame(columns=expected_cols + ["PLAYER_NAME_KEY"])

    df = pd.read_csv(path)

    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing columns in {path}: {missing}. "
            f"Expected columns: {expected_cols}"
        )

    df = df[expected_cols].copy()
    df["SEASON_END_YEAR"] = pd.to_numeric(df["SEASON_END_YEAR"], errors="coerce")
    df = df[df["SEASON_END_YEAR"].notna()].copy()
    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df[df["SEASON_END_YEAR"] >= MIN_DATA_SEASON].copy()

    df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(clean_player_name)
    df["PLAYER_NAME_KEY"] = df["PLAYER_NAME"].apply(make_player_name_key)
    df["AWARD_TYPE"] = df["AWARD_TYPE"].astype(str).str.lower().str.strip()
    df["IS_ELIGIBLE"] = (
        pd.to_numeric(df["IS_ELIGIBLE"], errors="coerce")
        .fillna(1)
        .astype(int)
    )
    df["IS_ELIGIBLE"] = df["IS_ELIGIBLE"].clip(0, 1)
    df["REASON"] = df["REASON"].fillna("").astype(str)
    df["SOURCE_URL"] = df["SOURCE_URL"].fillna("").astype(str)

    df = df.drop_duplicates(
        subset=["SEASON_END_YEAR", "PLAYER_NAME_KEY", "AWARD_TYPE"],
        keep="last",
    )

    return df.reset_index(drop=True)

def add_award_eligibility_features(
    df: pd.DataFrame,
    eligibility_overrides: pd.DataFrame | None,
) -> pd.DataFrame:
    df = df.copy()

    df["IS_ALL_NBA_ELIGIBLE"] = 1
    df["ALL_NBA_ELIGIBILITY_REASON"] = ""
    df["ALL_NBA_ELIGIBILITY_SOURCE_URL"] = ""

    df["IS_ALL_ROOKIE_ELIGIBLE"] = 1
    df["ALL_ROOKIE_ELIGIBILITY_REASON"] = ""
    df["ALL_ROOKIE_ELIGIBILITY_SOURCE_URL"] = ""

    if eligibility_overrides is None or eligibility_overrides.empty:
        return df

    for award_type, prefix in [
        ("all_nba", "ALL_NBA"),
        ("all_rookie", "ALL_ROOKIE"),
    ]:
        part = eligibility_overrides[
            eligibility_overrides["AWARD_TYPE"] == award_type
        ].copy()

        if part.empty:
            continue

        part = part.rename(
            columns={
                "IS_ELIGIBLE": f"{prefix}_ELIGIBILITY_OVERRIDE",
                "REASON": f"{prefix}_ELIGIBILITY_OVERRIDE_REASON",
                "SOURCE_URL": f"{prefix}_ELIGIBILITY_OVERRIDE_SOURCE_URL",
            }
        )

        keep_cols = [
            "SEASON_END_YEAR",
            "PLAYER_NAME_KEY",
            f"{prefix}_ELIGIBILITY_OVERRIDE",
            f"{prefix}_ELIGIBILITY_OVERRIDE_REASON",
            f"{prefix}_ELIGIBILITY_OVERRIDE_SOURCE_URL",
        ]

        df = df.merge(
            part[keep_cols],
            on=["SEASON_END_YEAR", "PLAYER_NAME_KEY"],
            how="left",
            validate="many_to_one",
        )

        override_col = f"{prefix}_ELIGIBILITY_OVERRIDE"
        reason_col = f"{prefix}_ELIGIBILITY_OVERRIDE_REASON"
        source_col = f"{prefix}_ELIGIBILITY_OVERRIDE_SOURCE_URL"

        has_override = df[override_col].notna()
        df.loc[has_override, f"IS_{prefix}_ELIGIBLE"] = (
            df.loc[has_override, override_col].astype(int)
        )
        df.loc[has_override, f"{prefix}_ELIGIBILITY_REASON"] = (
            df.loc[has_override, reason_col].fillna("").astype(str)
        )
        df.loc[has_override, f"{prefix}_ELIGIBILITY_SOURCE_URL"] = (
            df.loc[has_override, source_col].fillna("").astype(str)
        )

        df = df.drop(columns=[override_col, reason_col, source_col])

    return df

def build_labeled_dataset(
    player_stats: pd.DataFrame,
    all_nba_labels: pd.DataFrame,
    all_rookie_labels: pd.DataFrame,
    all_star_players: pd.DataFrame | None = None,
    bref_advanced_stats: pd.DataFrame | None = None,
    eligibility_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = player_stats.merge(
        all_nba_labels,
        on=["SEASON_END_YEAR", "PLAYER_NAME_KEY"],
        how="left",
    )

    df = df.merge(
        all_rookie_labels,
        on=["SEASON_END_YEAR", "PLAYER_NAME_KEY"],
        how="left",
    )

    df["ALL_NBA_LABEL"] = df["ALL_NBA_LABEL"].fillna(0).astype(int)
    df["ALL_ROOKIE_LABEL"] = df["ALL_ROOKIE_LABEL"].fillna(0).astype(int)

    df = add_rookie_candidate_flag(df)

    df = add_engineered_features(
        df,
        all_star_players=all_star_players,
        bref_advanced_stats=bref_advanced_stats,
    )

    df = add_award_eligibility_features(
        df,
        eligibility_overrides=eligibility_overrides,
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def build_all_nba_dataset(df: pd.DataFrame) -> pd.DataFrame:
    all_nba_df = df.copy()

    cols_to_drop = [
        "ALL_ROOKIE_LABEL",
        "IS_ROOKIE_CANDIDATE",
    ]

    existing_to_drop = [
        col for col in cols_to_drop
        if col in all_nba_df.columns
    ]

    all_nba_df = all_nba_df.drop(columns=existing_to_drop)

    return all_nba_df

def build_all_rookie_dataset(df: pd.DataFrame) -> pd.DataFrame:
    all_rookie_df = df[
        df["IS_ROOKIE_CANDIDATE"] == 1
    ].copy()

    cols_to_drop = [
        "ALL_NBA_LABEL",
    ]

    existing_to_drop = [
        col for col in cols_to_drop
        if col in all_rookie_df.columns
    ]

    all_rookie_df = all_rookie_df.drop(columns=existing_to_drop)

    return all_rookie_df

def print_label_info(
    df: pd.DataFrame,
    label_col: str,
    title: str,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(f"Shape: {df.shape}")
    print(
        "Seasons:",
        df["SEASON_END_YEAR"].min(),
        "-",
        df["SEASON_END_YEAR"].max(),
    )

    print()
    print(f"{label_col} counts:")
    print(df[label_col].value_counts().sort_index())

    print()
    print(f"{label_col} by season:")
    print(
        df[df[label_col] > 0]
        .groupby(["SEASON_END_YEAR", label_col])
        .size()
        .unstack(fill_value=0)
        .tail(35)
    )

def print_rookie_candidate_info(df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("ROOKIE CANDIDATES CHECK")
    print("=" * 80)

    print()
    print("Rookie candidates by season:")
    print(
        df.groupby("SEASON_END_YEAR")["IS_ROOKIE_CANDIDATE"]
        .sum()
        .astype(int)
        .tail(35)
    )

    print()
    print("Rows for target season 2026 among rookie candidates:")
    print(
        df[
            (df["SEASON_END_YEAR"] == TARGET_SEASON)
            & (df["IS_ROOKIE_CANDIDATE"] == 1)
        ].shape
    )

def print_engineered_feature_info(df: pd.DataFrame) -> None:
    new_features = [
        "TOTAL_MIN",
        "TOTAL_PTS",
        "TOTAL_REB",
        "TOTAL_AST",
        "PTS_REB_AST",
        "STOCKS",
        "AST_TOV_SIMPLE",

        "PREV_target_all_nba",
        "PLAYED_PREVIOUS_SEASON",
        "PREV_WAS_ALL_NBA",
        "PREV_ALL_NBA_LABEL",
        "PREV_ALL_NBA_VOTE_SCORE",
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
        "PREV_PER",
        "PREV_OWS",
        "PREV_DWS",
        "PREV_WS",
        "PREV_WS_PER_48",
        "PREV_OBPM",
        "PREV_DBPM",
        "PREV_BPM",
        "PREV_VORP",

        "PER",
        "OWS",
        "DWS",
        "WS",
        "WS_PER_48",
        "OBPM",
        "DBPM",
        "BPM",
        "VORP",

        "IS_ALL_STAR_THIS_SEASON",
        "PREV_ALL_STAR",
        "ALL_STAR_SELECTIONS_BEFORE_SEASON",

        "IS_ROOKIE",

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

    existing = [col for col in new_features if col in df.columns]
    missing = [col for col in new_features if col not in df.columns]

    print()
    print("=" * 80)
    print("ENGINEERED FEATURES CHECK")
    print("=" * 80)
    print(f"Expected engineered features: {len(new_features)}")
    print(f"Existing engineered features: {len(existing)}")
    print(f"Missing engineered features:  {len(missing)}")

    if missing:
        print()
        print("Missing engineered features:")
        for col in missing:
            print(f"  - {col}")

    print()
    print("NaN ratio for engineered features:")
    nan_ratio = df[existing].isna().mean().sort_values(ascending=False)
    print(nan_ratio.to_string())

def print_eligibility_info(df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("AWARD ELIGIBILITY CHECK")
    print("=" * 80)

    cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "IS_ALL_NBA_ELIGIBLE",
        "ALL_NBA_ELIGIBILITY_REASON",
    ]

    if all(col in df.columns for col in cols):
        ineligible = df[df["IS_ALL_NBA_ELIGIBLE"] == 0][cols].copy()
        print(f"All-NBA ineligible overrides: {len(ineligible)}")
        if not ineligible.empty:
            print(ineligible.sort_values(["SEASON_END_YEAR", "PLAYER_NAME"]).to_string(index=False))

def main() -> None:
    player_stats = load_player_stats()

    all_nba_labels = load_labels(
        ALL_NBA_LABELS_PATH,
        label_col="ALL_NBA_LABEL",
    )

    all_rookie_labels = load_labels(
        ALL_ROOKIE_LABELS_PATH,
        label_col="ALL_ROOKIE_LABEL",
    )

    all_star_players = load_or_fetch_all_star_players(
        raw_dir=RAW_DIR,
        min_year=int(player_stats["SEASON_END_YEAR"].min()),
        max_year=int(player_stats["SEASON_END_YEAR"].max()),
        filename=ALL_STAR_PLAYERS_PATH.name,
    )

    bref_advanced_stats = load_bref_advanced_stats(BREF_ADVANCED_STATS_PATH)
    eligibility_overrides = load_award_eligibility_overrides(AWARD_ELIGIBILITY_OVERRIDES_PATH)

    check_unmatched_labels(
        player_stats,
        all_nba_labels,
        label_col="ALL_NBA_LABEL",
    )

    check_unmatched_labels(
        player_stats,
        all_rookie_labels,
        label_col="ALL_ROOKIE_LABEL",
    )

    labeled_df = build_labeled_dataset(
        player_stats=player_stats,
        all_nba_labels=all_nba_labels,
        all_rookie_labels=all_rookie_labels,
        all_star_players=all_star_players,
        bref_advanced_stats=bref_advanced_stats,
        eligibility_overrides=eligibility_overrides,
    )

    all_nba_df = build_all_nba_dataset(labeled_df)
    all_rookie_df = build_all_rookie_dataset(labeled_df)

    labeled_df.to_csv(LABELED_OUTPUT_PATH, index=False)
    all_nba_df.to_csv(ALL_NBA_OUTPUT_PATH, index=False)
    all_rookie_df.to_csv(ALL_ROOKIE_OUTPUT_PATH, index=False)

    print_label_info(
        labeled_df,
        label_col="ALL_NBA_LABEL",
        title="GENERAL LABELED DATASET: ALL-NBA LABELS",
    )

    print_label_info(
        labeled_df,
        label_col="ALL_ROOKIE_LABEL",
        title="GENERAL LABELED DATASET: ALL-ROOKIE LABELS",
    )

    print_rookie_candidate_info(labeled_df)

    print_eligibility_info(labeled_df)

    print_engineered_feature_info(labeled_df)

    print()
    print("=" * 80)
    print("SAVED DATASETS")
    print("=" * 80)
    print(f"General labeled dataset: {LABELED_OUTPUT_PATH}")
    print(f"All-NBA dataset:         {ALL_NBA_OUTPUT_PATH}")
    print(f"All-Rookie dataset:      {ALL_ROOKIE_OUTPUT_PATH}")

    print()
    print("Rows for target season 2026:")
    print("general:", labeled_df[labeled_df["SEASON_END_YEAR"] == TARGET_SEASON].shape)
    print("all_nba:", all_nba_df[all_nba_df["SEASON_END_YEAR"] == TARGET_SEASON].shape)
    print("all_rookie:", all_rookie_df[all_rookie_df["SEASON_END_YEAR"] == TARGET_SEASON].shape)

if __name__ == "__main__":
    main()