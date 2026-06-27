from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROCESSED_DIR / "player_seasons_labeled.csv"

TWO_STAGE_MODEL_PATH = MODELS_DIR / "all_nba_two_stage_model.joblib"
TWO_STAGE_REPORT_PATH = MODELS_DIR / "all_nba_two_stage_report.json"

MIN_TRAIN_SEASON = 2000
MAX_TRAIN_SEASON = 2025
TARGET_SEASON = 2026

BACKTEST_START_SEASON = 2010
STAGE2_INTERNAL_START_SEASON = 2005

RANDOM_STATE = 42

TOP_K_STAGE1 = 25


CORE_RAW_FEATURES = [
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
]

ADVANCED_RAW_FEATURES = [
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
    "POSS",
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
]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)

    return df


def existing(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    existing_cols = [col for col in RANK_SOURCE_FEATURES if col in df.columns]

    new_cols = {}

    for col in existing_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        rank_col = f"{col}_SEASON_RANK_PCT"

        new_cols[rank_col] = (
            numeric.groupby(df["SEASON_END_YEAR"])
            .rank(pct=True, ascending=False)
        )

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rank_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_stage1_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Лучший feature_set из перебора: compact_plus_advanced.
    """
    cols = []
    cols += existing(CORE_RAW_FEATURES, df)
    cols += existing(ADVANCED_RAW_FEATURES, df)

    rank_features = [
        f"{col}_SEASON_RANK_PCT"
        for col in RANK_SOURCE_FEATURES
        if f"{col}_SEASON_RANK_PCT" in df.columns
    ]

    cols += rank_features

    seen = set()
    unique_cols = []

    for col in cols:
        if col not in seen:
            unique_cols.append(col)
            seen.add(col)

    return unique_cols


def encode_all_nba_target(labels: pd.Series) -> pd.Series:
    """
    Лучший target_scheme из перебора: vote_531.

    label:
    0 = none
    1 = third team
    2 = second team
    3 = first team

    target:
    0 -> 0
    1 -> 1
    2 -> 3
    3 -> 5
    """
    mapping = {
        0: 0.0,
        1: 1.0,
        2: 3.0,
        3: 5.0,
    }

    return labels.astype(int).map(mapping).astype(float)


def make_random_forest() -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=9,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def make_sample_weight(y: pd.Series) -> np.ndarray:
    """
    Лучший weight_mode из перебора: sqrt_balance.
    """
    y = pd.Series(y).astype(float)

    positive = y > 0
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())

    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y), dtype=float)

    positive_weight = float(np.sqrt(n_neg / n_pos))

    sample_weight = np.ones(len(y), dtype=float)
    sample_weight[positive.values] = positive_weight

    return sample_weight


def apply_rotation_filter(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    """
    Лучший pool_filter из перебора: rotation_players.
    """
    out = df.copy()

    if "GP" in out.columns:
        out = out[pd.to_numeric(out["GP"], errors="coerce") >= 40]

    if "MIN" in out.columns:
        out = out[pd.to_numeric(out["MIN"], errors="coerce") >= 18]

    if len(out) >= min_rows:
        return out.copy()

    return df.copy()


def train_stage1_model(train_df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    train_df = apply_rotation_filter(train_df, min_rows=100)

    X_train = train_df[feature_cols]
    y_train = encode_all_nba_target(train_df["ALL_NBA_LABEL"])
    sample_weight = make_sample_weight(y_train)

    model = make_random_forest()

    model.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weight,
    )

    return model


def add_stage1_features(
    candidate_df: pd.DataFrame,
    score_col: str = "STAGE1_SCORE",
) -> pd.DataFrame:
    """
    Дополнительные признаки для reranker-модели.
    Они описывают не только статистику игрока, но и положение игрока
    внутри списка кандидатов Stage 1.
    """
    df = candidate_df.copy()

    df = df.sort_values(
        by=[score_col, "PLAYER_NAME"],
        ascending=[False, True],
    ).reset_index(drop=True)

    df["STAGE1_RANK"] = np.arange(1, len(df) + 1)
    df["STAGE1_RANK_PCT"] = df["STAGE1_RANK"] / len(df)

    score_1 = float(df.iloc[0][score_col])

    score_5 = float(df.iloc[min(4, len(df) - 1)][score_col])
    score_10 = float(df.iloc[min(9, len(df) - 1)][score_col])
    score_15 = float(df.iloc[min(14, len(df) - 1)][score_col])

    df["STAGE1_GAP_TO_1ST"] = score_1 - df[score_col]
    df["STAGE1_GAP_TO_5TH"] = score_5 - df[score_col]
    df["STAGE1_GAP_TO_10TH"] = score_10 - df[score_col]
    df["STAGE1_GAP_TO_15TH"] = score_15 - df[score_col]

    df["STAGE1_SCORE_DIFF_PREV"] = df[score_col].shift(1) - df[score_col]
    df["STAGE1_SCORE_DIFF_NEXT"] = df[score_col] - df[score_col].shift(-1)

    df["STAGE1_SCORE_DIFF_PREV"] = df["STAGE1_SCORE_DIFF_PREV"].fillna(0.0)
    df["STAGE1_SCORE_DIFF_NEXT"] = df["STAGE1_SCORE_DIFF_NEXT"].fillna(0.0)

    # Rank inside candidate group.
    candidate_rank_cols = [
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "MIN",
        "W_PCT",
        "PLUS_MINUS",
        "NBA_FANTASY_PTS",
        "TS_PCT",
        "USG_PCT",
        "PIE",
        "NET_RATING",
    ]

    for col in candidate_rank_cols:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_CANDIDATE_RANK_PCT"] = numeric.rank(
                pct=True,
                ascending=False,
            )

    return df


def get_stage2_feature_columns(stage2_df: pd.DataFrame, stage1_feature_cols: list[str]) -> list[str]:
    extra_cols = [
        "STAGE1_SCORE",
        "STAGE1_RANK",
        "STAGE1_RANK_PCT",
        "STAGE1_GAP_TO_1ST",
        "STAGE1_GAP_TO_5TH",
        "STAGE1_GAP_TO_10TH",
        "STAGE1_GAP_TO_15TH",
        "STAGE1_SCORE_DIFF_PREV",
        "STAGE1_SCORE_DIFF_NEXT",
    ]

    candidate_rank_cols = [
        col for col in stage2_df.columns
        if col.endswith("_CANDIDATE_RANK_PCT")
    ]

    cols = []
    cols += stage1_feature_cols
    cols += extra_cols
    cols += candidate_rank_cols

    cols = [col for col in cols if col in stage2_df.columns]

    seen = set()
    unique_cols = []

    for col in cols:
        if col not in seen:
            unique_cols.append(col)
            seen.add(col)

    return unique_cols


def make_stage2_candidates_for_one_season(
    train_before_season_df: pd.DataFrame,
    season_df: pd.DataFrame,
    feature_cols: list[str],
    top_k: int = TOP_K_STAGE1,
) -> pd.DataFrame | None:
    if train_before_season_df.empty or season_df.empty:
        return None

    stage1_model = train_stage1_model(train_before_season_df, feature_cols)

    season_pool = apply_rotation_filter(season_df, min_rows=15)

    season_pool = season_pool.copy()
    season_pool["STAGE1_SCORE"] = stage1_model.predict(season_pool[feature_cols])

    candidates = (
        season_pool
        .sort_values(by=["STAGE1_SCORE", "PLAYER_NAME"], ascending=[False, True])
        .head(top_k)
        .copy()
    )

    candidates = add_stage1_features(candidates, score_col="STAGE1_SCORE")

    return candidates


def build_stage2_training_data(
    full_train_df: pd.DataFrame,
    feature_cols: list[str],
    first_stage2_season: int,
    last_stage2_season: int,
) -> pd.DataFrame:
    """
    Строим честный train для второй модели.

    Для каждого сезона s:
    - Stage 1 обучается только на сезонах < s
    - Stage 1 выбирает top-25 в сезоне s
    - эти top-25 идут в train для Stage 2

    Так Stage 2 учится на реалистичных кандидатах, а не на заранее правильных.
    """
    parts = []

    for season in range(first_stage2_season, last_stage2_season + 1):
        train_before = full_train_df[
            (full_train_df["SEASON_END_YEAR"] < season)
            & (full_train_df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        ].copy()

        season_df = full_train_df[full_train_df["SEASON_END_YEAR"] == season].copy()

        if train_before.empty or season_df.empty:
            continue

        candidates = make_stage2_candidates_for_one_season(
            train_before_season_df=train_before,
            season_df=season_df,
            feature_cols=feature_cols,
            top_k=TOP_K_STAGE1,
        )

        if candidates is not None and not candidates.empty:
            parts.append(candidates)

        print(f"Stage 2 candidates generated for season {season}")

    if not parts:
        raise RuntimeError("Could not build Stage 2 training data.")

    stage2_train_df = pd.concat(parts, axis=0, ignore_index=True)

    return stage2_train_df


def train_stage2_model(stage2_train_df: pd.DataFrame, stage2_feature_cols: list[str]) -> Pipeline:
    X_train = stage2_train_df[stage2_feature_cols]
    y_train = encode_all_nba_target(stage2_train_df["ALL_NBA_LABEL"])
    sample_weight = make_sample_weight(y_train)

    model = make_random_forest()

    model.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weight,
    )

    return model


def predict_two_stage_for_season(
    train_df: pd.DataFrame,
    test_season_df: pd.DataFrame,
    stage1_feature_cols: list[str],
    top_k: int = TOP_K_STAGE1,
) -> pd.DataFrame:
    """
    Полная двухэтапная предикция для одного test season.
    Используется в backtesting.
    """
    stage1_model = train_stage1_model(train_df, stage1_feature_cols)

    # Stage 2 train data строится только по train_df.
    train_min_season = int(train_df["SEASON_END_YEAR"].min())
    train_max_season = int(train_df["SEASON_END_YEAR"].max())

    first_stage2_season = max(STAGE2_INTERNAL_START_SEASON, train_min_season + 5)

    stage2_train_df = build_stage2_training_data(
        full_train_df=train_df,
        feature_cols=stage1_feature_cols,
        first_stage2_season=first_stage2_season,
        last_stage2_season=train_max_season,
    )

    dummy_stage2_df = add_stage1_features(
        stage2_train_df.copy(),
        score_col="STAGE1_SCORE",
    )

    stage2_feature_cols = get_stage2_feature_columns(
        dummy_stage2_df,
        stage1_feature_cols=stage1_feature_cols,
    )

    stage2_model = train_stage2_model(stage2_train_df, stage2_feature_cols)

    test_pool = apply_rotation_filter(test_season_df, min_rows=15)
    test_pool = test_pool.copy()

    test_pool["STAGE1_SCORE"] = stage1_model.predict(test_pool[stage1_feature_cols])

    candidates = (
        test_pool
        .sort_values(by=["STAGE1_SCORE", "PLAYER_NAME"], ascending=[False, True])
        .head(top_k)
        .copy()
    )

    candidates = add_stage1_features(candidates, score_col="STAGE1_SCORE")

    candidates["PRED_SCORE"] = stage2_model.predict(candidates[stage2_feature_cols])

    return candidates


def team_score_details(
    predicted_players: list[str],
    predicted_label: int,
    true_labels: dict[str, int],
) -> dict:
    points = 0
    exact_count = 0
    hit_count = 0
    near_1_count = 0
    near_2_count = 0

    for player in predicted_players:
        true_label = int(true_labels.get(player, 0))

        if true_label == 0:
            continue

        hit_count += 1
        diff = abs(predicted_label - true_label)

        if diff == 0:
            points += 10
            exact_count += 1
        elif diff == 1:
            points += 8
            near_1_count += 1
        elif diff == 2:
            points += 6
            near_2_count += 1

    bonus_by_exact_count = {
        0: 0,
        1: 0,
        2: 5,
        3: 10,
        4: 20,
        5: 40,
    }

    bonus = bonus_by_exact_count.get(exact_count, 40)
    points += bonus

    return {
        "points": int(points),
        "exact_count": int(exact_count),
        "hit_count": int(hit_count),
        "near_1_count": int(near_1_count),
        "near_2_count": int(near_2_count),
        "bonus": int(bonus),
    }


def score_all_nba(pred_df: pd.DataFrame) -> dict:
    true_labels = dict(
        zip(
            pred_df["PLAYER_NAME_KEY"],
            pred_df["ALL_NBA_LABEL"],
        )
    )

    ranked = pred_df.sort_values(
        by=["PRED_SCORE", "PLAYER_NAME"],
        ascending=[False, True],
    )

    first = ranked.iloc[0:5]["PLAYER_NAME_KEY"].tolist()
    second = ranked.iloc[5:10]["PLAYER_NAME_KEY"].tolist()
    third = ranked.iloc[10:15]["PLAYER_NAME_KEY"].tolist()

    first_details = team_score_details(first, 3, true_labels)
    second_details = team_score_details(second, 2, true_labels)
    third_details = team_score_details(third, 1, true_labels)

    total_score = (
        first_details["points"]
        + second_details["points"]
        + third_details["points"]
    )

    top_15 = ranked.iloc[0:15]
    top_15_hits = int((top_15["ALL_NBA_LABEL"] > 0).sum())

    return {
        "score": int(total_score),
        "max_score": 270,
        "top_15_hits": top_15_hits,
        "first_team": first_details,
        "second_team": second_details,
        "third_team": third_details,
    }


def backtest_two_stage(df: pd.DataFrame, stage1_feature_cols: list[str]) -> dict:
    results = []

    print()
    print("=" * 80)
    print("TWO-STAGE ALL-NBA BACKTEST")
    print("=" * 80)

    for test_season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        train_df = df[
            (df["SEASON_END_YEAR"] < test_season)
            & (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        ].copy()

        test_df = df[df["SEASON_END_YEAR"] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        pred_df = predict_two_stage_for_season(
            train_df=train_df,
            test_season_df=test_df,
            stage1_feature_cols=stage1_feature_cols,
            top_k=TOP_K_STAGE1,
        )

        metrics = score_all_nba(pred_df)

        results.append(
            {
                "season": int(test_season),
                **metrics,
            }
        )

        print(
            f"Two-stage All-NBA backtest {test_season}: "
            f"{metrics['score']} / 270 | "
            f"hits={metrics['top_15_hits']} / 15"
        )

    avg_score = float(np.mean([r["score"] for r in results]))
    avg_hits = float(np.mean([r["top_15_hits"] for r in results]))

    return {
        "task": "all_nba_two_stage",
        "average_score": avg_score,
        "average_top_15_hits": avg_hits,
        "results": results,
    }


def train_final_two_stage_model(df: pd.DataFrame, stage1_feature_cols: list[str]) -> dict:
    print()
    print("=" * 80)
    print("FINAL TWO-STAGE TRAINING")
    print("=" * 80)

    full_train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] <= MAX_TRAIN_SEASON)
    ].copy()

    final_stage1_model = train_stage1_model(full_train_df, stage1_feature_cols)

    stage2_train_df = build_stage2_training_data(
        full_train_df=full_train_df,
        feature_cols=stage1_feature_cols,
        first_stage2_season=STAGE2_INTERNAL_START_SEASON,
        last_stage2_season=MAX_TRAIN_SEASON,
    )

    stage2_feature_cols = get_stage2_feature_columns(
        stage2_train_df,
        stage1_feature_cols=stage1_feature_cols,
    )

    final_stage2_model = train_stage2_model(stage2_train_df, stage2_feature_cols)

    bundle = {
        "task": "all_nba_two_stage",
        "stage1_model": final_stage1_model,
        "stage2_model": final_stage2_model,
        "stage1_feature_cols": stage1_feature_cols,
        "stage2_feature_cols": stage2_feature_cols,
        "top_k_stage1": TOP_K_STAGE1,
        "config": {
            "stage1": {
                "feature_set": "compact_plus_advanced",
                "model_name": "random_forest",
                "target_scheme": "vote_531",
                "weight_mode": "sqrt_balance",
                "pool_filter": "rotation_players",
            },
            "stage2": {
                "model_name": "random_forest",
                "target_scheme": "vote_531",
                "weight_mode": "sqrt_balance",
                "training_candidates": f"top_{TOP_K_STAGE1}_from_stage1",
            },
        },
    }

    joblib.dump(bundle, TWO_STAGE_MODEL_PATH)

    print()
    print(f"Saved two-stage All-NBA model: {TWO_STAGE_MODEL_PATH}")
    print(f"Stage 1 features: {len(stage1_feature_cols)}")
    print(f"Stage 2 features: {len(stage2_feature_cols)}")
    print(f"Stage 2 training rows: {len(stage2_train_df)}")

    return bundle


def main() -> None:
    df = load_dataset()
    df = prepare_dataset(df)

    stage1_feature_cols = get_stage1_feature_columns(df)

    print()
    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(f"Seasons: {df['SEASON_END_YEAR'].min()} - {df['SEASON_END_YEAR'].max()}")
    print(f"Stage 1 features: {len(stage1_feature_cols)}")
    print(f"Top-K candidates from Stage 1: {TOP_K_STAGE1}")

    backtest_report = backtest_two_stage(df, stage1_feature_cols)

    print()
    print("=" * 80)
    print("TWO-STAGE VALIDATION RESULT")
    print("=" * 80)
    print(f"Average score: {backtest_report['average_score']:.2f} / 270")
    print(f"Average top-15 hits: {backtest_report['average_top_15_hits']:.2f} / 15")

    train_final_two_stage_model(df, stage1_feature_cols)

    report = {
        "baseline_best_single_model": {
            "average_score": 140.62,
            "average_top_15_hits": 12.12,
            "config": {
                "feature_set": "compact_plus_advanced",
                "model_name": "random_forest",
                "target_scheme": "vote_531",
                "weight_mode": "sqrt_balance",
                "pool_filter": "rotation_players",
            },
        },
        "two_stage": backtest_report,
        "saved_model": str(TWO_STAGE_MODEL_PATH),
    }

    TWO_STAGE_REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Saved report: {TWO_STAGE_REPORT_PATH}")


if __name__ == "__main__":
    main()