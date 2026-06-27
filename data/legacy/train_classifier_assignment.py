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
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_NBA_DATASET_PATH = PROCESSED_DIR / "all_nba_dataset.csv"

MODEL_PATH = MODELS_DIR / "all_nba_binary_candidate_regressor.joblib"
REPORT_PATH = MODELS_DIR / "validation_report_binary_candidate_regressor.json"


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


def load_dataset() -> pd.DataFrame:
    if not ALL_NBA_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {ALL_NBA_DATASET_PATH}")

    df = pd.read_csv(ALL_NBA_DATASET_PATH)

    required_cols = [
        "SEASON_END_YEAR",
        "PLAYER_NAME",
        "PLAYER_NAME_KEY",
        "ALL_NBA_LABEL",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(
                f"Missing column: {col}. Run build_dataset.py first."
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


def add_previous_award_features(df: pd.DataFrame) -> pd.DataFrame:
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


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rank_features(df)
    df = add_previous_award_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_compact_feature_columns(df: pd.DataFrame) -> list[str]:
    core_features = existing_columns(df, CORE_RAW_FEATURES)

    rank_features = [
        f"{col}_SEASON_RANK_PCT"
        for col in RANK_SOURCE_FEATURES
        if f"{col}_SEASON_RANK_PCT" in df.columns
    ]

    previous_features = existing_columns(df, ALL_NBA_PREVIOUS_FEATURES)

    features = core_features + rank_features + previous_features

    seen = set()
    unique_features = []

    for col in features:
        if col not in seen:
            unique_features.append(col)
            seen.add(col)

    return unique_features


def encode_regression_target(df: pd.DataFrame, target_scheme: str) -> pd.Series:
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

    raise ValueError(f"Unknown target scheme: {target_scheme}")


def make_binary_sample_weight(y: pd.Series, mode: str) -> np.ndarray:
    y_values = np.asarray(y, dtype=int)

    if mode == "none":
        return np.ones(len(y_values), dtype=float)

    if mode == "sqrt_balance":
        positives = y_values == 1
        pos_count = positives.sum()
        neg_count = len(y_values) - pos_count

        weights = np.ones(len(y_values), dtype=float)

        if pos_count > 0:
            weights[positives] = np.sqrt(neg_count / pos_count)

        return weights

    raise ValueError(f"Unknown binary weight mode: {mode}")


def make_regression_sample_weight(y: pd.Series, mode: str) -> np.ndarray:
    y_values = np.asarray(y, dtype=float)

    if mode == "none":
        return np.ones(len(y_values), dtype=float)

    if mode == "sqrt_balance":
        positives = y_values > 0
        pos_count = positives.sum()
        neg_count = len(y_values) - pos_count

        weights = np.ones(len(y_values), dtype=float)

        if pos_count > 0:
            weights[positives] = np.sqrt(neg_count / pos_count)

        return weights

    raise ValueError(f"Unknown regression weight mode: {mode}")


def make_binary_classifier(model_name: str):
    if model_name == "hist_gbc":
        model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )

    elif model_name == "grad_boost":
        model = GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        )

    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=9,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unknown binary classifier: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def make_regressor(model_name: str):
    if model_name == "grad_boost":
        model = GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=4,
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

    else:
        raise ValueError(f"Unknown regressor: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def fit_binary_classifier(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
):
    y = (train_df["ALL_NBA_LABEL"] > 0).astype(int)

    sample_weight = make_binary_sample_weight(
        y,
        mode=config["binary_weight_mode"],
    )

    model = make_binary_classifier(config["binary_model"])

    model.fit(
        train_df[feature_columns],
        y,
        model__sample_weight=sample_weight,
    )

    return model


def fit_regressor(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
):
    y = encode_regression_target(
        train_df,
        target_scheme=config["regression_target_scheme"],
    )

    sample_weight = make_regression_sample_weight(
        y,
        mode=config["regression_weight_mode"],
    )

    model = make_regressor(config["regression_model"])

    model.fit(
        train_df[feature_columns],
        y,
        model__sample_weight=sample_weight,
    )

    return model


def predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    estimator = model.named_steps["model"]
    classes = estimator.classes_

    class_to_idx = {
        int(cls): idx
        for idx, cls in enumerate(classes)
    }

    if 1 not in class_to_idx:
        return np.zeros(len(X), dtype=float)

    return proba[:, class_to_idx[1]]


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

        if len(filtered.drop_duplicates(subset=["PLAYER_NAME_KEY"])) >= 15:
            return filtered

        return df.copy()

    if pool_filter == "strong_rotation_players":
        filtered = df[
            (df["GP"] >= 50)
            & (df["MIN"] >= 24)
        ].copy()

        if len(filtered.drop_duplicates(subset=["PLAYER_NAME_KEY"])) >= 15:
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


def make_candidate_pool(
    test_df: pd.DataFrame,
    binary_model,
    feature_columns: list[str],
    config: dict,
) -> pd.DataFrame:
    test_df = test_df.copy()

    test_df["P_ALL_NBA"] = predict_positive_probability(
        binary_model,
        test_df[feature_columns],
    )

    filtered_df = apply_all_nba_pool_filter(
        test_df,
        pool_filter=config["pool_filter"],
    )

    candidate_df = select_top_unique_players(
        filtered_df,
        score_col="P_ALL_NBA",
        top_n=config["candidate_top_k"],
    )

    if len(candidate_df) < 15:
        candidate_df = select_top_unique_players(
            test_df,
            score_col="P_ALL_NBA",
            top_n=max(15, config["candidate_top_k"]),
        )

    return candidate_df


def add_selection_score(
    candidate_df: pd.DataFrame,
    regressor,
    feature_columns: list[str],
    config: dict,
) -> pd.DataFrame:
    candidate_df = candidate_df.copy()

    candidate_df["REG_SCORE"] = regressor.predict(candidate_df[feature_columns])

    if config["selection_score_mode"] == "regression_only":
        candidate_df["FINAL_SELECT_SCORE"] = candidate_df["REG_SCORE"]
        return candidate_df

    if config["selection_score_mode"] == "binary_guarded_regression":
        candidate_df["FINAL_SELECT_SCORE"] = candidate_df["REG_SCORE"]
        return candidate_df

    if config["selection_score_mode"] == "binary_regression_blend":
        binary_rank_score = candidate_df["P_ALL_NBA"].rank(
            method="average",
            ascending=True,
            pct=True,
        )

        regression_rank_score = candidate_df["REG_SCORE"].rank(
            method="average",
            ascending=True,
            pct=True,
        )

        alpha = config["binary_blend_weight"]

        candidate_df["FINAL_SELECT_SCORE"] = (
            alpha * binary_rank_score
            + (1.0 - alpha) * regression_rank_score
        )

        return candidate_df

    raise ValueError(f"Unknown selection score mode: {config['selection_score_mode']}")


def make_final_prediction(
    candidate_df: pd.DataFrame,
    regressor,
    feature_columns: list[str],
    config: dict,
) -> pd.DataFrame:
    candidate_df = add_selection_score(
        candidate_df=candidate_df,
        regressor=regressor,
        feature_columns=feature_columns,
        config=config,
    )

    if config["selection_score_mode"] == "binary_guarded_regression":
        lock_top_n = int(config["binary_lock_top_n"])

        locked_df = select_top_unique_players(
            candidate_df,
            score_col="P_ALL_NBA",
            top_n=lock_top_n,
        )

        remaining_df = candidate_df[
            ~candidate_df["PLAYER_NAME_KEY"].isin(locked_df["PLAYER_NAME_KEY"])
        ].copy()

        fill_df = select_top_unique_players(
            remaining_df,
            score_col="REG_SCORE",
            top_n=15 - len(locked_df),
        )

        final_df = pd.concat(
            [locked_df, fill_df],
            ignore_index=True,
        )

    else:
        # Шаг 1: выбираем финальные 15 игроков.
        # Здесь можно использовать blend binary + regression.
        final_df = select_top_unique_players(
            candidate_df,
            score_col="FINAL_SELECT_SCORE",
            top_n=15,
        )

    # Шаг 2: порядок first/second/third задаём только regression score.
    # Binary probability не должна портить расстановку по пятёркам.
    final_df = final_df.sort_values(
        "REG_SCORE",
        ascending=False,
    ).reset_index(drop=True)

    final_df["PREDICTED_LABEL"] = 0
    final_df.loc[0:4, "PREDICTED_LABEL"] = 3
    final_df.loc[5:9, "PREDICTED_LABEL"] = 2
    final_df.loc[10:14, "PREDICTED_LABEL"] = 1

    return final_df

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

    return {
        "points_without_bonus": points,
        "exact_count": exact_count,
        "bonus": bonus,
        "total": points + bonus,
        "players": player_details,
    }


def score_prediction(
    prediction_df: pd.DataFrame,
    true_df: pd.DataFrame,
) -> dict:
    true_labels = dict(
        zip(
            true_df.loc[true_df["ALL_NBA_LABEL"] > 0, "PLAYER_NAME_KEY"],
            true_df.loc[true_df["ALL_NBA_LABEL"] > 0, "ALL_NBA_LABEL"],
        )
    )

    first_keys = prediction_df.loc[
        prediction_df["PREDICTED_LABEL"] == 3,
        "PLAYER_NAME_KEY",
    ].tolist()

    second_keys = prediction_df.loc[
        prediction_df["PREDICTED_LABEL"] == 2,
        "PLAYER_NAME_KEY",
    ].tolist()

    third_keys = prediction_df.loc[
        prediction_df["PREDICTED_LABEL"] == 1,
        "PLAYER_NAME_KEY",
    ].tolist()

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

    predicted_keys = set(prediction_df["PLAYER_NAME_KEY"].tolist())
    true_keys = set(true_labels.keys())

    top_15_hits = len(predicted_keys & true_keys)

    return {
        "score": total_score,
        "top_hits": top_15_hits,
        "first": first_details,
        "second": second_details,
        "third": third_details,
    }


def evaluate_config(
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

        binary_model = fit_binary_classifier(
            train_df=train_df,
            feature_columns=feature_columns,
            config=config,
        )

        regressor = fit_regressor(
            train_df=train_df,
            feature_columns=feature_columns,
            config=config,
        )

        candidate_df = make_candidate_pool(
            test_df=test_df,
            binary_model=binary_model,
            feature_columns=feature_columns,
            config=config,
        )

        prediction_df = make_final_prediction(
            candidate_df=candidate_df,
            regressor=regressor,
            feature_columns=feature_columns,
            config=config,
        )

        score_info = score_prediction(
            prediction_df=prediction_df,
            true_df=test_df,
        )

        true_keys = set(
            test_df.loc[test_df["ALL_NBA_LABEL"] > 0, "PLAYER_NAME_KEY"].tolist()
        )

        candidate_keys = set(candidate_df["PLAYER_NAME_KEY"].tolist())
        prediction_keys = set(prediction_df["PLAYER_NAME_KEY"].tolist())

        candidate_hits = len(candidate_keys & true_keys)
        final_hits = len(prediction_keys & true_keys)

        season_results.append(
            {
                "season": test_season,
                "score": score_info["score"],
                "top_15_hits": score_info["top_hits"],
                "candidate_pool_hits": candidate_hits,
                "final_hits": final_hits,
                "first_exact": score_info["first"]["exact_count"],
                "second_exact": score_info["second"]["exact_count"],
                "third_exact": score_info["third"]["exact_count"],
            }
        )

    avg_score = float(np.mean([row["score"] for row in season_results]))
    avg_top_hits = float(np.mean([row["top_15_hits"] for row in season_results]))
    avg_candidate_hits = float(
        np.mean([row["candidate_pool_hits"] for row in season_results])
    )
    avg_final_hits = float(np.mean([row["final_hits"] for row in season_results]))

    return {
        "config": config,
        "avg_score": avg_score,
        "avg_top_hits": avg_top_hits,
        "avg_candidate_hits": avg_candidate_hits,
        "avg_final_hits": avg_final_hits,
        "season_results": season_results,
    }


def build_configs() -> list[dict]:
    configs = []

    for binary_blend_weight in [
        0.0,
        0.25,
        0.50,
        0.75,
    ]:
        config = {
            "approach": "binary_candidate_pool_plus_regressor_selector",
            "feature_set": "compact",
            "binary_model": "hist_gbc",
            "binary_weight_mode": "sqrt_balance",
            "regression_model": "grad_boost",
            "regression_target_scheme": "team_321",
            "regression_weight_mode": "sqrt_balance",
            "pool_filter": "rotation_players",
            "candidate_top_k": 30,
        }

        if binary_blend_weight == 0.0:
            config["selection_score_mode"] = "regression_only"
            config["binary_blend_weight"] = 0.0
        else:
            config["selection_score_mode"] = "binary_regression_blend"
            config["binary_blend_weight"] = binary_blend_weight

        configs.append(config)

    for binary_lock_top_n in [
        5,
        8,
        10,
        12,
    ]:
        configs.append(
            {
                "approach": "binary_candidate_pool_plus_regressor_selector",
                "feature_set": "compact",
                "binary_model": "hist_gbc",
                "binary_weight_mode": "sqrt_balance",
                "regression_model": "grad_boost",
                "regression_target_scheme": "team_321",
                "regression_weight_mode": "sqrt_balance",
                "pool_filter": "rotation_players",
                "candidate_top_k": 30,
                "selection_score_mode": "binary_guarded_regression",
                "binary_blend_weight": 0.0,
                "binary_lock_top_n": binary_lock_top_n,
            }
        )

    return configs


def find_best_config(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    configs = build_configs()

    print()
    print("=" * 80)
    print("ALL-NBA BINARY CANDIDATE + REGRESSOR SELECTOR SEARCH")
    print("=" * 80)
    print(f"Configs to test: {len(configs)}", flush=True)

    results = []

    for idx, config in enumerate(configs, start=1):
        print()
        print(
            f"START [{idx:03d}/{len(configs):03d}] {config}",
            flush=True,
        )

        result = evaluate_config(
            df=df,
            feature_columns=feature_columns,
            config=config,
        )

        results.append(result)

        print(
            f"DONE  [{idx:03d}/{len(configs):03d}] "
            f"score={result['avg_score']:.2f}/270 | "
            f"hits={result['avg_top_hits']:.2f}/15 | "
            f"candidate_hits={result['avg_candidate_hits']:.2f}/15 | "
            f"final_hits={result['avg_final_hits']:.2f}/15 | "
            f"{config}",
            flush=True,
        )

    best = max(results, key=lambda row: row["avg_score"])

    print()
    print("=" * 80)
    print("BEST ALL-NBA BINARY CANDIDATE + REGRESSOR SELECTOR CONFIG")
    print("=" * 80)
    print(f"score={best['avg_score']:.2f}/270")
    print(f"hits={best['avg_top_hits']:.2f}/15")
    print(f"candidate_hits={best['avg_candidate_hits']:.2f}/15")
    print(f"final_hits={best['avg_final_hits']:.2f}/15")
    print(best["config"])

    return best


def train_final_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    best_config: dict,
) -> dict:
    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] <= MAX_TRAIN_SEASON)
    ].copy()

    binary_model = fit_binary_classifier(
        train_df=train_df,
        feature_columns=feature_columns,
        config=best_config,
    )

    regressor = fit_regressor(
        train_df=train_df,
        feature_columns=feature_columns,
        config=best_config,
    )

    artifact = {
        "award_type": "all_nba",
        "model_type": "binary_candidate_pool_plus_regressor_selector",
        "dataset_path": str(ALL_NBA_DATASET_PATH),
        "binary_model": binary_model,
        "regressor": regressor,
        "feature_columns": feature_columns,
        "config": best_config,
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
    }

    joblib.dump(artifact, MODEL_PATH)

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


def save_report(
    best: dict,
    feature_columns: list[str],
) -> None:
    report = {
        "model_type": "binary_candidate_pool_plus_regressor_selector",
        "dataset_path": str(ALL_NBA_DATASET_PATH),
        "model_path": str(MODEL_PATH),
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
        "backtest_start_season": BACKTEST_START_SEASON,
        "feature_set": "compact",
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "best": best,
        "note": (
            "Binary classifier is used only to create a wide candidate pool. "
            "The final top-15 is selected by a regressor inside this candidate pool. "
            "No team assignment is allowed to replace selected players."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_dataset_info(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    print()
    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"Dataset: {ALL_NBA_DATASET_PATH}")
    print(f"Shape: {df.shape}")
    print(
        "Seasons:",
        df["SEASON_END_YEAR"].min(),
        "-",
        df["SEASON_END_YEAR"].max(),
    )
    print(f"Features: {len(feature_columns)}")

    print()
    print("ALL_NBA_LABEL counts:")
    print(df["ALL_NBA_LABEL"].value_counts().sort_index())

    print()
    print("Rows for target season:")
    print(df[df["SEASON_END_YEAR"] == TARGET_SEASON].shape)


def main() -> None:
    df = load_dataset()
    df = prepare_dataset(df)

    feature_columns = get_compact_feature_columns(df)

    print_dataset_info(
        df=df,
        feature_columns=feature_columns,
    )

    best = find_best_config(
        df=df,
        feature_columns=feature_columns,
    )

    train_final_model(
        df=df,
        feature_columns=feature_columns,
        best_config=best["config"],
    )

    save_report(
        best=best,
        feature_columns=feature_columns,
    )

    print()
    print("=" * 80)
    print("MODEL SAVED")
    print("=" * 80)
    print(f"Model:  {MODEL_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
