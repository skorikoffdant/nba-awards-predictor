from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pathlib import Path
import json
from itertools import product

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_NBA_DATASET_PATH = PROCESSED_DIR / "all_nba_dataset.csv"
ALL_ROOKIE_DATASET_PATH = PROCESSED_DIR / "all_rookie_dataset.csv"

ALL_NBA_MODEL_PATH = MODELS_DIR / "all_nba_model.joblib"
ALL_ROOKIE_MODEL_PATH = MODELS_DIR / "all_rookie_model.joblib"
VALIDATION_REPORT_PATH = MODELS_DIR / "validation_report.json"


MIN_TRAIN_SEASON = 2000
MAX_TRAIN_SEASON = 2025
TARGET_SEASON = 2026

BACKTEST_START_SEASON = 2010
RANDOM_STATE = 42


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


ALL_NBA_PREVIOUS_FEATURES = [
    "PLAYED_PREVIOUS_SEASON",
    "PREV_WAS_ALL_NBA",
    "PREV_ALL_NBA_VOTE_SCORE",
]


def load_dataset(path: Path, required_label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
        required_label_col,
    ]

    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(
                f"Missing column: {col} in {path}. "
                f"Run build_dataset.py first."
            )

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rank_cols = {}

    for col in existing_columns(df, RANK_SOURCE_FEATURES):
        rank_col = f"{col}_SEASON_RANK_PCT"

        rank_cols[rank_col] = (
            df.groupby("SEASON_END_YEAR")[col]
            .rank(method="average", ascending=False, pct=True)
        )

    if rank_cols:
        df = pd.concat([df, pd.DataFrame(rank_cols, index=df.index)], axis=1)

    return df


def add_all_nba_previous_award_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "PLAYER_ID" in df.columns:
        player_key_col = "PLAYER_ID"
    else:
        player_key_col = "PLAYER_NAME_KEY"

    df = df.sort_values([player_key_col, "SEASON_END_YEAR"]).reset_index(drop=True)

    prev_season = df.groupby(player_key_col)["SEASON_END_YEAR"].shift(1)
    valid_prev = prev_season == (df["SEASON_END_YEAR"] - 1)

    df["PLAYED_PREVIOUS_SEASON"] = valid_prev.astype(int)

    prev_all_nba = df.groupby(player_key_col)["ALL_NBA_LABEL"].shift(1)
    prev_all_nba = prev_all_nba.where(valid_prev, 0).fillna(0)

    df["PREV_WAS_ALL_NBA"] = (prev_all_nba > 0).astype(float)

    vote_map = {
        0.0: 0.0,
        1.0: 1.0,
        2.0: 3.0,
        3.0: 5.0,
    }

    df["PREV_ALL_NBA_VOTE_SCORE"] = (
        prev_all_nba.map(vote_map).fillna(0.0).astype(float)
    )

    return df


def prepare_all_nba_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rank_features(df)
    df = add_all_nba_previous_award_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def prepare_all_rookie_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rank_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def get_compact_feature_columns(
    df: pd.DataFrame,
    include_all_nba_previous_features: bool,
) -> list[str]:
    core_features = existing_columns(df, CORE_RAW_FEATURES)

    rank_features = [
        f"{col}_SEASON_RANK_PCT"
        for col in RANK_SOURCE_FEATURES
        if f"{col}_SEASON_RANK_PCT" in df.columns
    ]

    previous_features = []
    if include_all_nba_previous_features:
        previous_features = existing_columns(df, ALL_NBA_PREVIOUS_FEATURES)

    features = core_features + rank_features + previous_features

    seen = set()
    unique_features = []

    for col in features:
        if col not in seen:
            unique_features.append(col)
            seen.add(col)

    return unique_features


def encode_all_nba_target(df: pd.DataFrame, target_scheme: str) -> pd.Series:
    labels = df["ALL_NBA_LABEL"].astype(float)

    if target_scheme == "team_321":
        return labels

    if target_scheme == "vote_531":
        return labels.map(
            {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 3.0,
                3.0: 5.0,
            }
        ).astype(float)

    raise ValueError(f"Unknown All-NBA target scheme: {target_scheme}")


def encode_rookie_target(df: pd.DataFrame, target_scheme: str) -> pd.Series:
    labels = df["ALL_ROOKIE_LABEL"].astype(float)

    if target_scheme == "rookie_21":
        return labels

    if target_scheme == "rookie_31":
        return labels.map(
            {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 3.0,
            }
        ).astype(float)

    raise ValueError(f"Unknown All-Rookie target scheme: {target_scheme}")


def make_sample_weight(y: pd.Series, weight_mode: str) -> np.ndarray:
    y_values = np.asarray(y, dtype=float)

    if weight_mode == "none":
        return np.ones(len(y_values), dtype=float)

    if weight_mode == "sqrt_balance":
        positives = y_values > 0
        pos_count = positives.sum()
        neg_count = len(y_values) - pos_count

        weights = np.ones(len(y_values), dtype=float)

        if pos_count > 0:
            weights[positives] = np.sqrt(neg_count / pos_count)

        return weights

    raise ValueError(f"Unknown weight mode: {weight_mode}")


def make_model(model_name: str):
    if model_name == "hist_gbr":
        model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )

    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=220,
            max_depth=9,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=220,
            max_depth=9,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    elif model_name == "grad_boost":
        model = GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def fit_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target: pd.Series,
    config: dict,
):
    X_train = train_df[feature_columns]
    y_train = target

    sample_weight = make_sample_weight(y_train, config["weight_mode"])

    model = make_model(config["model_name"])

    model.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weight,
    )

    return model


def apply_all_nba_pool_filter(
    df: pd.DataFrame,
    pool_filter: str,
) -> pd.DataFrame:
    if pool_filter == "none":
        return df.copy()

    if pool_filter == "rotation_players":
        filtered = df[
            (df["GP"] >= 40)
            & (df["MIN"] >= 18)
        ].copy()

        if len(filtered) >= 15:
            return filtered

        return df.copy()

    if pool_filter == "strong_rotation_players":
        filtered = df[
            (df["GP"] >= 50)
            & (df["MIN"] >= 24)
        ].copy()

        if len(filtered) >= 15:
            return filtered

        return df.copy()

    raise ValueError(f"Unknown pool filter: {pool_filter}")


def select_top_unique_players(
    df: pd.DataFrame,
    score_col: str,
    top_n: int,
) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).copy()

    ranked = ranked.drop_duplicates(
        subset=["PLAYER_NAME_KEY"],
        keep="first",
    )

    return ranked.head(top_n).copy()


def team_score_details(
    predicted_player_keys: list[str],
    predicted_label: int,
    true_labels: dict[str, int],
) -> dict:
    points = 0
    exact_count = 0
    player_details = []

    for player_key in predicted_player_keys:
        true_label = int(true_labels.get(player_key, 0))

        if true_label == 0:
            player_points = 0
            diff = None
        else:
            diff = abs(predicted_label - true_label)

            if diff == 0:
                player_points = 10
                exact_count += 1
            elif diff == 1:
                player_points = 8
            elif diff == 2:
                player_points = 6
            else:
                player_points = 0

        points += player_points

        player_details.append(
            {
                "player_key": player_key,
                "predicted_label": predicted_label,
                "true_label": true_label,
                "points": player_points,
                "diff": diff,
            }
        )

    bonus_by_exact_count = {
        0: 0,
        1: 0,
        2: 5,
        3: 10,
        4: 20,
        5: 40,
    }

    bonus = bonus_by_exact_count.get(exact_count, 40)
    total = points + bonus

    return {
        "points_without_bonus": points,
        "exact_count": exact_count,
        "bonus": bonus,
        "total": total,
        "players": player_details,
    }


def score_all_nba(
    predictions_df: pd.DataFrame,
    true_df: pd.DataFrame,
) -> dict:
    true_labels = dict(
        zip(
            true_df.loc[true_df["ALL_NBA_LABEL"] > 0, "PLAYER_NAME_KEY"],
            true_df.loc[true_df["ALL_NBA_LABEL"] > 0, "ALL_NBA_LABEL"],
        )
    )

    top_15 = select_top_unique_players(
        predictions_df,
        score_col="PRED_SCORE",
        top_n=15,
    )

    first_keys = top_15.iloc[0:5]["PLAYER_NAME_KEY"].tolist()
    second_keys = top_15.iloc[5:10]["PLAYER_NAME_KEY"].tolist()
    third_keys = top_15.iloc[10:15]["PLAYER_NAME_KEY"].tolist()

    first_details = team_score_details(
        first_keys,
        predicted_label=3,
        true_labels=true_labels,
    )

    second_details = team_score_details(
        second_keys,
        predicted_label=2,
        true_labels=true_labels,
    )

    third_details = team_score_details(
        third_keys,
        predicted_label=1,
        true_labels=true_labels,
    )

    total_score = (
        first_details["total"]
        + second_details["total"]
        + third_details["total"]
    )

    predicted_keys = set(top_15["PLAYER_NAME_KEY"].tolist())
    true_keys = set(true_labels.keys())

    top_15_hits = len(predicted_keys & true_keys)

    return {
        "score": total_score,
        "top_hits": top_15_hits,
        "first": first_details,
        "second": second_details,
        "third": third_details,
    }


def score_all_rookie(
    predictions_df: pd.DataFrame,
    true_df: pd.DataFrame,
) -> dict:
    true_labels = dict(
        zip(
            true_df.loc[true_df["ALL_ROOKIE_LABEL"] > 0, "PLAYER_NAME_KEY"],
            true_df.loc[true_df["ALL_ROOKIE_LABEL"] > 0, "ALL_ROOKIE_LABEL"],
        )
    )

    top_10 = select_top_unique_players(
        predictions_df,
        score_col="PRED_SCORE",
        top_n=10,
    )

    first_keys = top_10.iloc[0:5]["PLAYER_NAME_KEY"].tolist()
    second_keys = top_10.iloc[5:10]["PLAYER_NAME_KEY"].tolist()

    first_details = team_score_details(
        first_keys,
        predicted_label=2,
        true_labels=true_labels,
    )

    second_details = team_score_details(
        second_keys,
        predicted_label=1,
        true_labels=true_labels,
    )

    total_score = first_details["total"] + second_details["total"]

    predicted_keys = set(top_10["PLAYER_NAME_KEY"].tolist())
    true_keys = set(true_labels.keys())

    top_10_hits = len(predicted_keys & true_keys)

    return {
        "score": total_score,
        "top_hits": top_10_hits,
        "first": first_details,
        "second": second_details,
    }


def evaluate_all_nba_config(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
) -> dict:
    season_results = []

    for test_season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        train_df = df[
            (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
            & (df["SEASON_END_YEAR"] < test_season)
        ].copy()

        test_df = df[df["SEASON_END_YEAR"] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        target = encode_all_nba_target(
            train_df,
            target_scheme=config["target_scheme"],
        )

        model = fit_model(
            train_df=train_df,
            feature_columns=feature_columns,
            target=target,
            config=config,
        )

        test_df["PRED_SCORE"] = model.predict(test_df[feature_columns])

        prediction_pool = apply_all_nba_pool_filter(
            test_df,
            pool_filter=config["pool_filter"],
        )

        score_info = score_all_nba(
            predictions_df=prediction_pool,
            true_df=test_df,
        )

        season_results.append(
            {
                "season": test_season,
                "score": score_info["score"],
                "top_15_hits": score_info["top_hits"],
            }
        )

    avg_score = float(np.mean([row["score"] for row in season_results]))
    avg_hits = float(np.mean([row["top_15_hits"] for row in season_results]))

    return {
        "config": config,
        "avg_score": avg_score,
        "avg_top_hits": avg_hits,
        "season_results": season_results,
    }


def evaluate_all_rookie_config(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
) -> dict:
    season_results = []

    for test_season in range(BACKTEST_START_SEASON, MAX_TRAIN_SEASON + 1):
        train_df = df[
            (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
            & (df["SEASON_END_YEAR"] < test_season)
        ].copy()

        test_df = df[df["SEASON_END_YEAR"] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        target = encode_rookie_target(
            train_df,
            target_scheme=config["target_scheme"],
        )

        model = fit_model(
            train_df=train_df,
            feature_columns=feature_columns,
            target=target,
            config=config,
        )

        test_df["PRED_SCORE"] = model.predict(test_df[feature_columns])

        score_info = score_all_rookie(
            predictions_df=test_df,
            true_df=test_df,
        )

        season_results.append(
            {
                "season": test_season,
                "score": score_info["score"],
                "top_10_hits": score_info["top_hits"],
            }
        )

    avg_score = float(np.mean([row["score"] for row in season_results]))
    avg_hits = float(np.mean([row["top_10_hits"] for row in season_results]))

    return {
        "config": config,
        "avg_score": avg_score,
        "avg_top_hits": avg_hits,
        "season_results": season_results,
    }


def build_all_nba_configs() -> list[dict]:
    model_names = [
        "grad_boost",
        "random_forest",
    ]

    target_schemes = [
        "team_321",
        "vote_531",
    ]

    weight_modes = [
        "none",
        "sqrt_balance",
    ]

    pool_filters = [
        "rotation_players",
    ]

    configs = []

    for model_name, target_scheme, weight_mode, pool_filter in product(
        model_names,
        target_schemes,
        weight_modes,
        pool_filters,
    ):
        configs.append(
            {
                "feature_set": "compact",
                "model_name": model_name,
                "target_scheme": target_scheme,
                "weight_mode": weight_mode,
                "pool_filter": pool_filter,
            }
        )

    return configs


def build_all_rookie_configs() -> list[dict]:
    model_names = [
        "hist_gbr",
        "extra_trees",
    ]

    target_schemes = [
        "rookie_21",
    ]

    weight_modes = [
        "none",
        "sqrt_balance",
    ]

    configs = []

    for model_name, target_scheme, weight_mode in product(
        model_names,
        target_schemes,
        weight_modes,
    ):
        configs.append(
            {
                "feature_set": "compact",
                "model_name": model_name,
                "target_scheme": target_scheme,
                "weight_mode": weight_mode,
                "pool_filter": "all_rookie_dataset_only",
            }
        )

    return configs


def find_best_all_nba_model(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    configs = build_all_nba_configs()

    print()
    print("=" * 80)
    print("ALL-NBA MODEL SEARCH")
    print("=" * 80)
    print(f"Dataset: {ALL_NBA_DATASET_PATH}")
    print(f"Configs to test: {len(configs)}")

    results = []

    for idx, config in enumerate(configs, start=1):
        result = evaluate_all_nba_config(
            df=df,
            feature_columns=feature_columns,
            config=config,
        )

        results.append(result)

        print(
            f"[{idx:03d}/{len(configs):03d}] "
            f"score={result['avg_score']:.2f}/270 | "
            f"hits={result['avg_top_hits']:.2f}/15 | "
            f"features={len(feature_columns)} | "
            f"{config}"
        )

    best = max(results, key=lambda row: row["avg_score"])

    print()
    print("=" * 80)
    print("BEST ALL-NBA CONFIG")
    print("=" * 80)
    print(f"score={best['avg_score']:.2f}/270")
    print(f"hits={best['avg_top_hits']:.2f}/15")
    print(best["config"])

    return best


def find_best_all_rookie_model(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    configs = build_all_rookie_configs()

    print()
    print("=" * 80)
    print("ALL-ROOKIE MODEL SEARCH")
    print("=" * 80)
    print(f"Dataset: {ALL_ROOKIE_DATASET_PATH}")
    print(f"Configs to test: {len(configs)}")

    results = []

    for idx, config in enumerate(configs, start=1):
        result = evaluate_all_rookie_config(
            df=df,
            feature_columns=feature_columns,
            config=config,
        )

        results.append(result)

        print(
            f"[{idx:03d}/{len(configs):03d}] "
            f"score={result['avg_score']:.2f}/180 | "
            f"hits={result['avg_top_hits']:.2f}/10 | "
            f"features={len(feature_columns)} | "
            f"{config}"
        )

    best = max(results, key=lambda row: row["avg_score"])

    print()
    print("=" * 80)
    print("BEST ALL-ROOKIE CONFIG")
    print("=" * 80)
    print(f"score={best['avg_score']:.2f}/180")
    print(f"hits={best['avg_top_hits']:.2f}/10")
    print(best["config"])

    return best


def train_final_all_nba_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    best_config: dict,
) -> dict:
    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] <= MAX_TRAIN_SEASON)
    ].copy()

    target = encode_all_nba_target(
        train_df,
        target_scheme=best_config["target_scheme"],
    )

    model = fit_model(
        train_df=train_df,
        feature_columns=feature_columns,
        target=target,
        config=best_config,
    )

    artifact = {
        "award_type": "all_nba",
        "dataset_path": str(ALL_NBA_DATASET_PATH),
        "model": model,
        "feature_columns": feature_columns,
        "config": best_config,
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
    }

    joblib.dump(artifact, ALL_NBA_MODEL_PATH)

    return artifact


def train_final_all_rookie_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    best_config: dict,
) -> dict:
    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] <= MAX_TRAIN_SEASON)
    ].copy()

    target = encode_rookie_target(
        train_df,
        target_scheme=best_config["target_scheme"],
    )

    model = fit_model(
        train_df=train_df,
        feature_columns=feature_columns,
        target=target,
        config=best_config,
    )

    artifact = {
        "award_type": "all_rookie",
        "dataset_path": str(ALL_ROOKIE_DATASET_PATH),
        "model": model,
        "feature_columns": feature_columns,
        "config": best_config,
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
    }

    joblib.dump(artifact, ALL_ROOKIE_MODEL_PATH)

    return artifact


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def save_validation_report(
    all_nba_best: dict,
    all_rookie_best: dict,
    all_nba_feature_columns: list[str],
    all_rookie_feature_columns: list[str],
) -> None:
    report = {
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
        "backtest_start_season": BACKTEST_START_SEASON,
        "all_nba": {
            "dataset_path": str(ALL_NBA_DATASET_PATH),
            "model_path": str(ALL_NBA_MODEL_PATH),
            "feature_set": "compact",
            "num_features": len(all_nba_feature_columns),
            "feature_columns": all_nba_feature_columns,
            "search_space": {
                "models": ["grad_boost", "random_forest"],
                "target_schemes": ["team_321", "vote_531"],
                "weight_modes": ["none", "sqrt_balance"],
                "pool_filters": ["rotation_players"],
            },
            "best": all_nba_best,
        },
        "all_rookie": {
            "dataset_path": str(ALL_ROOKIE_DATASET_PATH),
            "model_path": str(ALL_ROOKIE_MODEL_PATH),
            "feature_set": "compact",
            "num_features": len(all_rookie_feature_columns),
            "feature_columns": all_rookie_feature_columns,
            "search_space": {
                "models": ["hist_gbr", "extra_trees"],
                "target_schemes": ["rookie_21"],
                "weight_modes": ["none", "sqrt_balance"],
                "pool_filters": ["already_filtered_by_dataset"],
            },
            "best": all_rookie_best,
        },
    }

    VALIDATION_REPORT_PATH.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_dataset_info(
    name: str,
    df: pd.DataFrame,
    label_col: str,
    feature_columns: list[str],
) -> None:
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(
        "Seasons:",
        df["SEASON_END_YEAR"].min(),
        "-",
        df["SEASON_END_YEAR"].max(),
    )
    print(f"Target column: {label_col}")
    print(f"Compact features: {len(feature_columns)}")

    print()
    print(f"{label_col} counts:")
    print(df[label_col].value_counts().sort_index())

    print()
    print("Rows for target season:")
    print(df[df["SEASON_END_YEAR"] == TARGET_SEASON].shape)


def main() -> None:
    all_nba_df = load_dataset(
        ALL_NBA_DATASET_PATH,
        required_label_col="ALL_NBA_LABEL",
    )

    all_rookie_df = load_dataset(
        ALL_ROOKIE_DATASET_PATH,
        required_label_col="ALL_ROOKIE_LABEL",
    )

    all_nba_df = prepare_all_nba_dataset(all_nba_df)
    all_rookie_df = prepare_all_rookie_dataset(all_rookie_df)

    all_nba_feature_columns = get_compact_feature_columns(
        all_nba_df,
        include_all_nba_previous_features=True,
    )

    all_rookie_feature_columns = get_compact_feature_columns(
        all_rookie_df,
        include_all_nba_previous_features=False,
    )

    print_dataset_info(
        name="ALL-NBA DATASET INFO",
        df=all_nba_df,
        label_col="ALL_NBA_LABEL",
        feature_columns=all_nba_feature_columns,
    )

    print_dataset_info(
        name="ALL-ROOKIE DATASET INFO",
        df=all_rookie_df,
        label_col="ALL_ROOKIE_LABEL",
        feature_columns=all_rookie_feature_columns,
    )

    all_nba_best = find_best_all_nba_model(
        df=all_nba_df,
        feature_columns=all_nba_feature_columns,
    )

    all_rookie_best = find_best_all_rookie_model(
        df=all_rookie_df,
        feature_columns=all_rookie_feature_columns,
    )

    train_final_all_nba_model(
        df=all_nba_df,
        feature_columns=all_nba_feature_columns,
        best_config=all_nba_best["config"],
    )

    train_final_all_rookie_model(
        df=all_rookie_df,
        feature_columns=all_rookie_feature_columns,
        best_config=all_rookie_best["config"],
    )

    save_validation_report(
        all_nba_best=all_nba_best,
        all_rookie_best=all_rookie_best,
        all_nba_feature_columns=all_nba_feature_columns,
        all_rookie_feature_columns=all_rookie_feature_columns,
    )

    print()
    print("=" * 80)
    print("FINAL MODELS SAVED")
    print("=" * 80)
    print(f"All-NBA model:     {ALL_NBA_MODEL_PATH}")
    print(f"All-Rookie model:  {ALL_ROOKIE_MODEL_PATH}")
    print(f"Validation report: {VALIDATION_REPORT_PATH}")


if __name__ == "__main__":
    main()