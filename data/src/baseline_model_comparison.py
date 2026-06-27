from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import add_season_rank_features, get_feature_columns
import train_hgb_classifier_full_score as hgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "models" / "baseline_model_comparison"
PLOTS_DIR = OUTPUT_DIR / "plots"

MIN_TRAIN_SEASON = 2000
BACKTEST_START_SEASON = 2010
BACKTEST_END_SEASON = 2025
RANDOM_STATE = 42

ALL_NBA_FEATURE_SET = "previous_team_share_allstar"
ALL_ROOKIE_FEATURE_SET = "previous_team_share"

BASE_CONFIG = {
    "weight_mode": "positive_boost",
    "selection_mode": "expected_label",
    "team_assignment_mode": "sort_expected_label",
    "pool_filter": "none",
}


MODEL_CONFIGS = {
    "hgb_classifier": {
        "max_iter": 250,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.05,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    "extra_trees": {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    "gradient_boosting": {
        "n_estimators": 150,
        "learning_rate": 0.03,
        "max_depth": 2,
        "subsample": 0.85,
    },
    "logistic_regression": {
        "C": 0.5,
        "max_iter": 2000,
    },
}


def make_model(name: str):
    params = MODEL_CONFIGS[name]

    if name == "hgb_classifier":
        model = HistGradientBoostingClassifier(
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            l2_regularization=params["l2_regularization"],
            random_state=RANDOM_STATE,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])

    if name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])

    if name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])

    if name == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            random_state=RANDOM_STATE,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])

    if name == "logistic_regression":
        model = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver="lbfgs",
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    raise ValueError(f"Unknown model: {name}")


def load_award_dataset(award_name: str) -> pd.DataFrame:
    award_cfg = hgb.AWARD_CONFIGS[award_name]
    df = hgb.load_dataset(award_cfg["dataset_path"], award_cfg["label_col"])
    return add_season_rank_features(df)


def award_feature_set(award_name: str) -> str:
    if award_name == "all_nba":
        return ALL_NBA_FEATURE_SET
    if award_name == "all_rookie":
        return ALL_ROOKIE_FEATURE_SET
    raise ValueError(f"Unknown award: {award_name}")


def train_and_score_award(
    model_name: str,
    award_name: str,
    df: pd.DataFrame,
    test_season: int,
) -> dict:
    award_cfg = hgb.AWARD_CONFIGS[award_name]
    label_col = award_cfg["label_col"]
    feature_columns = get_feature_columns(
        df,
        feature_set=award_feature_set(award_name),
        verbose=False,
    )

    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] < test_season)
    ].copy()
    test_df = df[df["SEASON_END_YEAR"] == test_season].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError(f"Empty train/test split for {award_name} {test_season}")

    y_train = train_df[label_col].astype(int)
    sample_weight = hgb.make_sample_weight(y_train, mode=BASE_CONFIG["weight_mode"])

    model = make_model(model_name)
    model.fit(train_df[feature_columns], y_train, model__sample_weight=sample_weight)

    config = dict(BASE_CONFIG)
    config["feature_set"] = award_feature_set(award_name)
    config["model_name"] = model_name

    prediction_df = hgb.make_prediction(
        test_df=test_df,
        model=model,
        feature_columns=feature_columns,
        config=config,
        award_cfg=award_cfg,
    )

    score_info = hgb.score_prediction(
        prediction_df=prediction_df,
        true_df=test_df,
        label_col=label_col,
        award_cfg=award_cfg,
    )

    return {
        "score": int(score_info["score"]),
        "hits": int(score_info["top_hits"]),
        "max_score": int(award_cfg["max_score"]),
        "num_players": int(award_cfg["num_players"]),
    }


def evaluate_model(model_name: str, all_nba_df: pd.DataFrame, all_rookie_df: pd.DataFrame) -> list[dict]:
    rows = []

    for season in range(BACKTEST_START_SEASON, BACKTEST_END_SEASON + 1):
        nba = train_and_score_award(model_name, "all_nba", all_nba_df, season)
        rookie = train_and_score_award(model_name, "all_rookie", all_rookie_df, season)

        row = {
            "model": model_name,
            "season": season,
            "all_nba_score": nba["score"],
            "all_rookie_score": rookie["score"],
            "total_score": nba["score"] + rookie["score"],
            "all_nba_hits": nba["hits"],
            "all_rookie_hits": rookie["hits"],
        }
        rows.append(row)

        print(
            f"{model_name:<20} {season}: "
            f"total={row['total_score']:>3}/450 | "
            f"nba={row['all_nba_score']:>3}/270 | "
            f"rookie={row['all_rookie_score']:>3}/180 | "
            f"hits={row['all_nba_hits']:>2}/15 + {row['all_rookie_hits']:>2}/10",
            flush=True,
        )

    return rows


def build_summary(season_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, g in season_df.groupby("model", sort=False):
        rows.append({
            "model": model_name,
            "num_seasons": int(g["season"].nunique()),
            "score_mean": float(g["total_score"].mean()),
            "score_pct": float(g["total_score"].mean() / 450.0 * 100.0),
            "all_nba_score_mean": float(g["all_nba_score"].mean()),
            "all_rookie_score_mean": float(g["all_rookie_score"].mean()),
            "all_nba_hits_mean": float(g["all_nba_hits"].mean()),
            "all_rookie_hits_mean": float(g["all_rookie_hits"].mean()),
        })
    return pd.DataFrame(rows).sort_values("score_mean", ascending=False).reset_index(drop=True)


def make_average_score_plot(summary_df: pd.DataFrame) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / "baseline_models_average_score.png"

    ordered = summary_df.sort_values("score_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(ordered["model"], ordered["score_mean"])
    ax.set_title("Porównanie modeli bazowych")
    ax.set_xlabel("Średni wynik / 450")
    ax.set_xlim(0, 450)

    for idx, value in enumerate(ordered["score_mean"]):
        ax.text(value + 2, idx, f"{value:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return plot_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_nba_df = load_award_dataset("all_nba")
    all_rookie_df = load_award_dataset("all_rookie")

    all_rows = []
    for model_name in MODEL_CONFIGS:
        print()
        print("=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)
        all_rows.extend(evaluate_model(model_name, all_nba_df, all_rookie_df))

    season_df = pd.DataFrame(all_rows)
    summary_df = build_summary(season_df)

    season_path = OUTPUT_DIR / "baseline_model_season_results.csv"
    summary_path = OUTPUT_DIR / "baseline_model_summary.csv"
    config_path = OUTPUT_DIR / "baseline_model_configs.json"

    season_df.to_csv(season_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps(MODEL_CONFIGS, indent=2), encoding="utf-8")

    plot_path = make_average_score_plot(summary_df)

    print()
    print("=" * 80)
    print("BASELINE MODEL COMPARISON")
    print("=" * 80)
    display = summary_df.copy()
    for col in display.columns:
        if col.endswith("mean") or col.endswith("pct"):
            display[col] = pd.to_numeric(display[col], errors="coerce").round(2)
    print(display.to_string(index=False))
    print()
    print(f"[saved] {season_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {config_path}")
    print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main()
