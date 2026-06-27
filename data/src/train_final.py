from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import get_feature_columns, add_season_rank_features

import train_hgb_classifier_full_score as hgb
import train_team_reorderer as reorder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODELS_DIR = PROJECT_ROOT / "models" / "final"
FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_ALL_NBA_MODEL_PATH = FINAL_MODELS_DIR / "final_all_nba_hgb_reorderer.joblib"
FINAL_ALL_ROOKIE_MODEL_PATH = FINAL_MODELS_DIR / "final_all_rookie_hgb.joblib"
FINAL_TRAIN_REPORT_PATH = FINAL_MODELS_DIR / "final_train_report.json"

MIN_TRAIN_SEASON = 2000
MAX_TRAIN_SEASON = 2025
TARGET_SEASON = 2026


HGB_ALL_NBA_CONFIG = {
    "feature_set": "previous_team_share_allstar",
    "model_name": "hgb_classifier",
    "weight_mode": "positive_boost",
    "selection_mode": "expected_label",
    "team_assignment_mode": "sort_expected_label",
    "pool_filter": "none",
    "max_iter": 250,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.05,
}

HGB_ALL_ROOKIE_CONFIG = {
    **HGB_ALL_NBA_CONFIG,
    "feature_set": "previous_team_share",
}

REORDERER_CONFIG = {
    "base_feature_set": "previous_team_share_allstar",
    "base_weight_mode": "positive_boost",
    "base_selection_score_col": "EXPECTED_LABEL",
    "base_max_iter": 250,
    "base_learning_rate": 0.05,
    "base_max_leaf_nodes": 31,
    "base_l2_regularization": 0.05,
    "candidate_pool_start_season": 2005,
    "reorderer_model_type": "classifier",
    "reorderer_max_iter": 150,
    "reorderer_learning_rate": 0.03,
    "reorderer_max_leaf_nodes": 15,
    "reorderer_l2_regularization": 0.05,
    "reorderer_weight_mode": "team_weighted",
    "reorderer_target_mode": "spaced",
    "blend_weight": 0.25,
    "use_original_features": True,
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def fit_final_hgb_classifier(
    df: pd.DataFrame,
    label_col: str,
    feature_columns: list[str],
    config: dict,
):
    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] <= MAX_TRAIN_SEASON)
    ].copy()

    if train_df.empty:
        raise RuntimeError(f"Empty final training data for {label_col}")

    y_train = train_df[label_col].astype(int)
    sample_weight = hgb.make_sample_weight(y_train, mode=config["weight_mode"])

    model = hgb.make_hgb_classifier(config)
    model.fit(
        train_df[feature_columns],
        y_train,
        model__sample_weight=sample_weight,
    )

    return model, train_df


def fit_final_all_nba_base_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
):
    train_df = df[
        (df["SEASON_END_YEAR"] >= MIN_TRAIN_SEASON)
        & (df["SEASON_END_YEAR"] <= MAX_TRAIN_SEASON)
    ].copy()

    if train_df.empty:
        raise RuntimeError("Empty final training data for All-NBA")

    y_train = train_df[reorder.LABEL_COL].astype(int)
    sample_weight = reorder.make_sample_weight(
        y_train,
        config["base_weight_mode"],
    )

    model = reorder.make_hgb_classifier(config)
    model.fit(
        train_df[feature_columns],
        y_train,
        model__sample_weight=sample_weight,
    )

    return model, train_df


def fit_final_reorderer(
    top15_pools: dict[int, pd.DataFrame],
    feature_columns: list[str],
    config: dict,
):
    train_pools = [
        pool
        for season, pool in sorted(top15_pools.items())
        if season <= MAX_TRAIN_SEASON
    ]

    if not train_pools:
        raise RuntimeError("No top-15 pools available for final reorderer training")

    train_df = pd.concat(train_pools, ignore_index=True)
    y_label = train_df[reorder.LABEL_COL].astype(int)
    sample_weight = reorder.make_sample_weight(
        y_label,
        config["reorderer_weight_mode"],
    )

    if config["reorderer_model_type"] == "classifier":
        model = reorder.make_reorderer_classifier(config)
        model.fit(
            train_df[feature_columns],
            y_label,
            model__sample_weight=sample_weight,
        )
        return model, train_df

    if config["reorderer_model_type"] == "regressor":
        y_target = reorder.make_reorderer_target(
            y_label,
            config["reorderer_target_mode"],
        )
        model = reorder.make_reorderer_regressor(config)
        model.fit(
            train_df[feature_columns],
            y_target,
            model__sample_weight=sample_weight,
        )
        return model, train_df

    raise ValueError(f"Unknown reorderer_model_type: {config['reorderer_model_type']}")


def print_short_result(title: str, result: dict) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if "avg_total_score" in result:
        total_score = result["avg_total_score"]
        total_percent = 100.0 * result["avg_total_score_pct"]
        print(f"TOTAL={total_score:.2f}/450 ({total_percent:.2f}%)")
        print(f"All-NBA={result['all_nba']['avg_score']:.2f}/270")
        print(f"All-Rookie={result['all_rookie']['avg_score']:.2f}/180")
        print(f"All-NBA hits={result['all_nba']['avg_hits']:.2f}/15")
        print(f"All-Rookie hits={result['all_rookie']['avg_hits']:.2f}/10")
        return

    print(f"score={result['avg_score']:.2f}/270")
    print(f"hits={result['avg_hits']:.2f}/15")
    print(
        f"exact=({result['avg_first_exact']:.2f}, "
        f"{result['avg_second_exact']:.2f}, "
        f"{result['avg_third_exact']:.2f})"
    )


def run_hgb_backtest() -> dict:
    all_nba_df = hgb.load_dataset(
        hgb.AWARD_CONFIGS["all_nba"]["dataset_path"],
        hgb.AWARD_CONFIGS["all_nba"]["label_col"],
    )
    all_rookie_df = hgb.load_dataset(
        hgb.AWARD_CONFIGS["all_rookie"]["dataset_path"],
        hgb.AWARD_CONFIGS["all_rookie"]["label_col"],
    )

    all_nba_df = add_season_rank_features(all_nba_df)
    all_rookie_df = add_season_rank_features(all_rookie_df)

    result = hgb.evaluate_full_config(
        all_nba_df=all_nba_df,
        all_rookie_df=all_rookie_df,
        config=HGB_ALL_NBA_CONFIG,
    )
    print_short_result("BACKTEST: HGB BASELINE", result)
    return result


def prepare_reorderer_backtest():
    all_nba_df = reorder.load_dataset()
    base_feature_columns = get_feature_columns(
        all_nba_df,
        feature_set=REORDERER_CONFIG["base_feature_set"],
        verbose=True,
    )

    top15_pools = reorder.build_top15_pools(
        df=all_nba_df,
        feature_columns=base_feature_columns,
        config=REORDERER_CONFIG,
        candidate_pool_start_season=REORDERER_CONFIG["candidate_pool_start_season"],
    )

    first_pool = next(iter(top15_pools.values()))
    reorderer_feature_columns = reorder.get_reorderer_feature_columns(
        first_pool,
        base_feature_columns=base_feature_columns,
        use_original_features=REORDERER_CONFIG["use_original_features"],
    )

    result = reorder.evaluate_reorderer_config(
        df=all_nba_df,
        top15_pools=top15_pools,
        feature_columns=reorderer_feature_columns,
        config=REORDERER_CONFIG,
    )
    print_short_result("BACKTEST: ALL-NBA FIXED TOP-15 REORDERER", result)

    return all_nba_df, base_feature_columns, top15_pools, reorderer_feature_columns, result


def train_all_nba_artifact(
    all_nba_df: pd.DataFrame,
    base_feature_columns: list[str],
    top15_pools: dict[int, pd.DataFrame],
    reorderer_feature_columns: list[str],
    backtest_result: dict,
) -> None:
    print()
    print("=" * 80)
    print("TRAINING FINAL ALL-NBA ARTIFACT")
    print("=" * 80)

    base_model, base_train_df = fit_final_all_nba_base_model(
        df=all_nba_df,
        feature_columns=base_feature_columns,
        config=REORDERER_CONFIG,
    )

    reorderer_model, reorderer_train_df = fit_final_reorderer(
        top15_pools=top15_pools,
        feature_columns=reorderer_feature_columns,
        config=REORDERER_CONFIG,
    )

    artifact = {
        "award_type": "all_nba",
        "model_type": "hgb_plus_fixed_top15_reorderer",
        "base_model": base_model,
        "reorderer_model": reorderer_model,
        "base_feature_set": REORDERER_CONFIG["base_feature_set"],
        "base_feature_columns": base_feature_columns,
        "reorderer_feature_columns": reorderer_feature_columns,
        "config": REORDERER_CONFIG,
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
        "base_train_rows": int(len(base_train_df)),
        "reorderer_train_rows": int(len(reorderer_train_df)),
        "backtest": backtest_result,
    }

    joblib.dump(artifact, FINAL_ALL_NBA_MODEL_PATH)
    print(f"[saved] {FINAL_ALL_NBA_MODEL_PATH}")


def train_all_rookie_artifact(hgb_backtest_result: dict) -> None:
    print()
    print("=" * 80)
    print("TRAINING FINAL ALL-ROOKIE ARTIFACT")
    print("=" * 80)

    df = hgb.load_dataset(
        hgb.AWARD_CONFIGS["all_rookie"]["dataset_path"],
        hgb.AWARD_CONFIGS["all_rookie"]["label_col"],
    )
    df = add_season_rank_features(df)

    feature_columns = get_feature_columns(
        df,
        feature_set=HGB_ALL_ROOKIE_CONFIG["feature_set"],
        verbose=True,
    )

    model, train_df = fit_final_hgb_classifier(
        df=df,
        label_col=hgb.AWARD_CONFIGS["all_rookie"]["label_col"],
        feature_columns=feature_columns,
        config=HGB_ALL_ROOKIE_CONFIG,
    )

    artifact = {
        "award_type": "all_rookie",
        "model_type": "hgb_classifier",
        "model": model,
        "feature_set": HGB_ALL_ROOKIE_CONFIG["feature_set"],
        "feature_columns": feature_columns,
        "config": HGB_ALL_ROOKIE_CONFIG,
        "min_train_season": MIN_TRAIN_SEASON,
        "max_train_season": MAX_TRAIN_SEASON,
        "target_season": TARGET_SEASON,
        "train_rows": int(len(train_df)),
        "baseline_backtest": hgb_backtest_result["all_rookie"],
    }

    joblib.dump(artifact, FINAL_ALL_ROOKIE_MODEL_PATH)
    print(f"[saved] {FINAL_ALL_ROOKIE_MODEL_PATH}")


def save_report(hgb_backtest_result: dict, reorderer_backtest_result: dict) -> float:
    final_total = (
        float(reorderer_backtest_result["avg_score"])
        + float(hgb_backtest_result["all_rookie"]["avg_score"])
    )

    report = {
        "final_solution": {
            "all_nba": "HGB classifier + fixed top-15 reorderer",
            "all_rookie": "HGB classifier",
            "all_nba_model_path": str(FINAL_ALL_NBA_MODEL_PATH),
            "all_rookie_model_path": str(FINAL_ALL_ROOKIE_MODEL_PATH),
            "expected_backtest_total": final_total,
        },
        "hgb_baseline_backtest": hgb_backtest_result,
        "all_nba_reorderer_backtest": reorderer_backtest_result,
        "configs": {
            "hgb_all_nba": HGB_ALL_NBA_CONFIG,
            "hgb_all_rookie": HGB_ALL_ROOKIE_CONFIG,
            "all_nba_reorderer": REORDERER_CONFIG,
        },
    }

    FINAL_TRAIN_REPORT_PATH.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] {FINAL_TRAIN_REPORT_PATH}")

    return final_total


def main() -> None:
    print()
    print("=" * 80)
    print("FINAL TRAINING CONFIG")
    print("=" * 80)
    print("All-NBA:    HGB classifier + fixed top-15 reorderer")
    print("All-Rookie: HGB classifier")
    print(f"Train seasons: {MIN_TRAIN_SEASON}-{MAX_TRAIN_SEASON}")
    print(f"Target season: {TARGET_SEASON}")
    print(f"Models dir: {FINAL_MODELS_DIR}")

    hgb_backtest_result = run_hgb_backtest()
    (
        all_nba_df,
        base_feature_columns,
        top15_pools,
        reorderer_feature_columns,
        reorderer_backtest_result,
    ) = prepare_reorderer_backtest()

    train_all_nba_artifact(
        all_nba_df=all_nba_df,
        base_feature_columns=base_feature_columns,
        top15_pools=top15_pools,
        reorderer_feature_columns=reorderer_feature_columns,
        backtest_result=reorderer_backtest_result,
    )
    train_all_rookie_artifact(hgb_backtest_result)

    final_total = save_report(hgb_backtest_result, reorderer_backtest_result)

    print()
    print("=" * 80)
    print("FINAL MODELS READY")
    print("=" * 80)
    print(f"Backtest final total: {final_total:.2f}/450")
    print(f"All-NBA model:    {FINAL_ALL_NBA_MODEL_PATH}")
    print(f"All-Rookie model: {FINAL_ALL_ROOKIE_MODEL_PATH}")


if __name__ == "__main__":
    main()