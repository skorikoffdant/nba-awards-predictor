# data/src/features.py

from __future__ import annotations

import pandas as pd


BASIC_FEATURES = [
    "AGE",
    "GP",
    "W",
    "L",
    "W_PCT",
    "MIN",
    "PTS",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PLUS_MINUS",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "AST_TO",
    "AST_RATIO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "USG_PCT",
    "PACE",
    "PIE",
    "IS_ROOKIE",
]


TOTAL_FEATURES = [
    "TOTAL_MIN",
    "TOTAL_PTS",
    "TOTAL_REB",
    "TOTAL_AST",
    "PTS_REB_AST",
    "STOCKS",
    "AST_TOV_SIMPLE",
]


PREVIOUS_FEATURES = [
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
]


TEAM_RELATIVE_SHARE_FEATURES = [
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


ALL_STAR_FEATURES = [
    "IS_ALL_STAR_THIS_SEASON",
    "PREV_ALL_STAR",
    "ALL_STAR_SELECTIONS_BEFORE_SEASON",
]


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

BREF_STAR_FEATURES = [
    "PER",
    "OBPM",
    "DBPM",
    "BPM",
    "VORP",

    "PREV_PER",
    "PREV_OBPM",
    "PREV_DBPM",
    "PREV_BPM",
    "PREV_VORP",

    "PER_SEASON_RANK_PCT",
    "OBPM_SEASON_RANK_PCT",
    "DBPM_SEASON_RANK_PCT",
    "BPM_SEASON_RANK_PCT",
    "VORP_SEASON_RANK_PCT",
]

BREF_PREVIOUS_FEATURES = [
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


RANK_SOURCE_FEATURES = [
    "GP",
    "W",
    "W_PCT",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
    "DD2",
    "TD3",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "REB_PCT",
    "EFG_PCT",
    "TS_PCT",
    "USG_PCT",
    "PIE",
] + BREF_ADVANCED_FEATURES


SEASON_RANK_FEATURES = [
    f"{col}_SEASON_RANK_PCT"
    for col in RANK_SOURCE_FEATURES
]


BREF_SEASON_RANK_FEATURES = [
    f"{col}_SEASON_RANK_PCT"
    for col in BREF_ADVANCED_FEATURES
]


COMPACT_FEATURES = [
    "AGE",
    "GP",
    "W",
    "L",
    "W_PCT",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PTS",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
    "DD2",
    "TD3",
] + SEASON_RANK_FEATURES


BEST_FEATURES_PREVIOUS_TEAM_SHARE = (
    BASIC_FEATURES
    + TOTAL_FEATURES
    + PREVIOUS_FEATURES
    + TEAM_RELATIVE_SHARE_FEATURES
)

BEST_FEATURES_PREVIOUS_TEAM_SHARE_ALLSTAR = (
    BEST_FEATURES_PREVIOUS_TEAM_SHARE
    + ALL_STAR_FEATURES
)

BEST_FEATURES_PREVIOUS_TEAM_SHARE_BREF = (
    BEST_FEATURES_PREVIOUS_TEAM_SHARE
    + BREF_ADVANCED_FEATURES
    + BREF_PREVIOUS_FEATURES
    + BREF_SEASON_RANK_FEATURES
)

BEST_FEATURES_PREVIOUS_TEAM_SHARE_BREF_STAR = (
    BEST_FEATURES_PREVIOUS_TEAM_SHARE
    + BREF_STAR_FEATURES
)

BEST_FEATURES_PREVIOUS_TEAM_SHARE_BREF_ALLSTAR = (
    BEST_FEATURES_PREVIOUS_TEAM_SHARE_BREF
    + ALL_STAR_FEATURES
)

COMPACT_PLUS_PREVIOUS_TEAM_SHARE_FEATURES = (
    COMPACT_FEATURES
    + TOTAL_FEATURES
    + PREVIOUS_FEATURES
    + TEAM_RELATIVE_SHARE_FEATURES
)

COMPACT_PLUS_PREVIOUS_TEAM_SHARE_ALLSTAR_FEATURES = (
    COMPACT_PLUS_PREVIOUS_TEAM_SHARE_FEATURES
    + ALL_STAR_FEATURES
)

COMPACT_PLUS_PREVIOUS_TEAM_SHARE_BREF_FEATURES = (
    COMPACT_PLUS_PREVIOUS_TEAM_SHARE_FEATURES
    + BREF_ADVANCED_FEATURES
    + BREF_PREVIOUS_FEATURES
    + BREF_SEASON_RANK_FEATURES
)

COMPACT_PLUS_PREVIOUS_TEAM_SHARE_BREF_ALLSTAR_FEATURES = (
    COMPACT_PLUS_PREVIOUS_TEAM_SHARE_BREF_FEATURES
    + ALL_STAR_FEATURES
)

BEST_FEATURES_PREVIOUS_TEAM_SHARE_BREF_STAR_ALLSTAR = (
    BEST_FEATURES_PREVIOUS_TEAM_SHARE
    + BREF_STAR_FEATURES
    + ALL_STAR_FEATURES
)

REPUTATION_FEATURES = [
    "PREV_target_all_nba",
    "PREV_WAS_ALL_NBA",
    "PREV_ALL_NBA_LABEL",
    "PREV_ALL_NBA_VOTE_SCORE",
    "PREV_ALL_STAR",
    "ALL_STAR_SELECTIONS_BEFORE_SEASON",
]

CURRENT_STAR_FEATURES = [
    "IS_ALL_STAR_THIS_SEASON",
]


REPUTATION_CONTROL_FEATURES = [
    "PREV_AWARD_FLAG",
    "PREV_STAR_FLAG",
    "REPUTATION_SCORE",
    "PREV_AWARD_BUT_NOT_ALL_STAR",
    "PREV_AWARD_BUT_LOW_GP",
    "PREV_AWARD_BUT_LOW_MIN",
    "PREV_AWARD_BUT_LOW_TEAM_WIN",
    "PREV_STAR_BUT_NOT_ALL_STAR",
    "PREV_STAR_BUT_LOW_GP",
    "PREV_STAR_BUT_LOW_MIN",
    "PREV_STAR_BUT_LOW_TEAM_WIN",
    "REPUTATION_X_GP",
    "REPUTATION_X_MIN",
    "REPUTATION_X_W_PCT",
    "REPUTATION_X_IS_ALL_STAR",
    "REPUTATION_WITHOUT_CURRENT_ALL_STAR",
]


def remove_features(columns: list[str], banned: list[str]) -> list[str]:
    banned_set = set(banned)
    return [c for c in columns if c not in banned_set]

FEATURE_SETS = {
    "compact": COMPACT_FEATURES,
    "previous_team_share": BEST_FEATURES_PREVIOUS_TEAM_SHARE,
    "previous_team_share_allstar": BEST_FEATURES_PREVIOUS_TEAM_SHARE_ALLSTAR,
    "previous_team_share_bref": BEST_FEATURES_PREVIOUS_TEAM_SHARE_BREF,
    "previous_team_share_bref_allstar": BEST_FEATURES_PREVIOUS_TEAM_SHARE_BREF_ALLSTAR,
    "compact_plus_previous_team_share": COMPACT_PLUS_PREVIOUS_TEAM_SHARE_FEATURES,
    "compact_plus_previous_team_share_allstar": COMPACT_PLUS_PREVIOUS_TEAM_SHARE_ALLSTAR_FEATURES,
    "compact_plus_previous_team_share_bref": COMPACT_PLUS_PREVIOUS_TEAM_SHARE_BREF_FEATURES,
    "previous_team_share_bref_star_allstar": BEST_FEATURES_PREVIOUS_TEAM_SHARE_BREF_STAR_ALLSTAR,
    "previous_team_share_bref_star": BEST_FEATURES_PREVIOUS_TEAM_SHARE_BREF_STAR,
    "compact_plus_previous_team_share_bref_allstar": COMPACT_PLUS_PREVIOUS_TEAM_SHARE_BREF_ALLSTAR_FEATURES,
}

FEATURE_SETS["previous_team_share_no_reputation"] = remove_features(
    FEATURE_SETS["previous_team_share_allstar"],
    REPUTATION_FEATURES + CURRENT_STAR_FEATURES,
)

FEATURE_SETS["previous_team_share_current_star_only"] = (
    remove_features(
        FEATURE_SETS["previous_team_share_allstar"],
        REPUTATION_FEATURES + CURRENT_STAR_FEATURES,
    )
    + CURRENT_STAR_FEATURES
)

FEATURE_SETS["previous_team_share_no_prev_awards"] = remove_features(
    FEATURE_SETS["previous_team_share_allstar"],
    [
        "PREV_target_all_nba",
        "PREV_WAS_ALL_NBA",
        "PREV_ALL_NBA_LABEL",
        "PREV_ALL_NBA_VOTE_SCORE",
    ],
)

FEATURE_SETS["previous_team_share_allstar_reputation_control"] = (
    FEATURE_SETS["previous_team_share_allstar"]
    + REPUTATION_CONTROL_FEATURES
)

def unique_keep_order(columns: list[str]) -> list[str]:
    seen = set()
    result = []

    for col in columns:
        if col not in seen:
            result.append(col)
            seen.add(col)

    return result


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        col
        for col in unique_keep_order(columns)
        if col in df.columns
    ]


def missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        col
        for col in unique_keep_order(columns)
        if col not in df.columns
    ]


def get_feature_columns(
    df: pd.DataFrame,
    feature_set: str,
    verbose: bool = True,
) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set={feature_set}. "
            f"Available: {list(FEATURE_SETS.keys())}"
        )

    requested = FEATURE_SETS[feature_set]
    existing = existing_columns(df, requested)
    missing = missing_columns(df, requested)

    if verbose:
        print()
        print("=" * 80)
        print("FEATURE SET INFO")
        print("=" * 80)
        print(f"Feature set: {feature_set}")
        print(f"Requested features: {len(unique_keep_order(requested))}")
        print(f"Existing features:   {len(existing)}")
        print(f"Missing features:    {len(missing)}")

        if missing:
            print()
            print("Missing columns:")
            for col in missing:
                print(f"  - {col}")

    if not existing:
        raise RuntimeError(f"No usable features found for feature_set={feature_set}")

    return existing


def add_reputation_control_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prev_award = (
        df.get("PREV_WAS_ALL_NBA", 0).fillna(0).astype(float)
        + df.get("PREV_ALL_NBA_LABEL", 0).fillna(0).astype(float)
        + df.get("PREV_target_all_nba", 0).fillna(0).astype(float)
    )
    prev_award_flag = (prev_award > 0).astype(int)

    prev_star = (
        df.get("PREV_ALL_STAR", 0).fillna(0).astype(float)
        + df.get("ALL_STAR_SELECTIONS_BEFORE_SEASON", 0).fillna(0).astype(float)
    )
    prev_star_flag = (prev_star > 0).astype(int)

    current_star = df.get("IS_ALL_STAR_THIS_SEASON", 0).fillna(0).astype(float)

    gp = df.get("GP", 0).fillna(0).astype(float)
    minutes = df.get("MIN", 0).fillna(0).astype(float)
    w_pct = df.get("W_PCT", 0).fillna(0).astype(float)

    reputation_score = prev_award_flag + 0.5 * prev_star_flag + 0.25 * prev_star.clip(0, 10)

    df["PREV_AWARD_FLAG"] = prev_award_flag
    df["PREV_STAR_FLAG"] = prev_star_flag
    df["REPUTATION_SCORE"] = reputation_score

    df["PREV_AWARD_BUT_NOT_ALL_STAR"] = prev_award_flag * (1 - current_star)
    df["PREV_AWARD_BUT_LOW_GP"] = prev_award_flag * (gp < 60).astype(int)
    df["PREV_AWARD_BUT_LOW_MIN"] = prev_award_flag * (minutes < 30).astype(int)
    df["PREV_AWARD_BUT_LOW_TEAM_WIN"] = prev_award_flag * (w_pct < 0.50).astype(int)

    df["PREV_STAR_BUT_NOT_ALL_STAR"] = prev_star_flag * (1 - current_star)
    df["PREV_STAR_BUT_LOW_GP"] = prev_star_flag * (gp < 60).astype(int)
    df["PREV_STAR_BUT_LOW_MIN"] = prev_star_flag * (minutes < 30).astype(int)
    df["PREV_STAR_BUT_LOW_TEAM_WIN"] = prev_star_flag * (w_pct < 0.50).astype(int)

    df["REPUTATION_X_GP"] = reputation_score * gp
    df["REPUTATION_X_MIN"] = reputation_score * minutes
    df["REPUTATION_X_W_PCT"] = reputation_score * w_pct
    df["REPUTATION_X_IS_ALL_STAR"] = reputation_score * current_star
    df["REPUTATION_WITHOUT_CURRENT_ALL_STAR"] = reputation_score * (1 - current_star)

    return df


def add_season_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rank_cols = {}

    for col in existing_columns(df, RANK_SOURCE_FEATURES):
        rank_col = f"{col}_SEASON_RANK_PCT"

        rank_cols[rank_col] = (
            df.groupby("SEASON_END_YEAR")[col]
            .rank(method="average", ascending=False, pct=True)
        )

    if rank_cols:
        df = pd.concat(
            [df, pd.DataFrame(rank_cols, index=df.index)],
            axis=1,
        )

    df = add_reputation_control_features(df)

    return df
