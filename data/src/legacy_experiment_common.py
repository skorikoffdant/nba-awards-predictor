
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_OUTPUT_DIR = MODELS_DIR / "legacy_experiments"
DEFAULT_MIN_TRAIN_SEASON = 2000
DEFAULT_BACKTEST_START = 2010
DEFAULT_BACKTEST_END = 2025
DEFAULT_FEATURE_SET = "previous_team_share_allstar"
RANDOM_STATE = 42


@dataclass(frozen=True)
class AwardSpec:
    name: str
    label_col: str
    dataset_path: Path
    top_n: int
    max_label: int
    max_score: int

    @property
    def team_labels(self) -> list[int]:
        return list(range(self.max_label, 0, -1))

    @property
    def team_names(self) -> list[str]:
        if self.max_label == 3:
            return ["first", "second", "third"]
        return ["first", "second"]


AWARDS = {
    "all_nba": AwardSpec(
        name="all_nba",
        label_col="ALL_NBA_LABEL",
        dataset_path=PROCESSED_DIR / "all_nba_dataset.csv",
        top_n=15,
        max_label=3,
        max_score=270,
    ),
    "all_rookie": AwardSpec(
        name="all_rookie",
        label_col="ALL_ROOKIE_LABEL",
        dataset_path=PROCESSED_DIR / "all_rookie_dataset.csv",
        top_n=10,
        max_label=2,
        max_score=180,
    ),
}


def parse_legacy_args(description: str, default_experiment: str, awards: list[str] | None = None):
    parser = argparse.ArgumentParser(description=description)
    allowed_awards = awards or ["all", "all_nba", "all_rookie"]
    default_award = "all" if "all" in allowed_awards else allowed_awards[0]

    parser.add_argument("--experiment-name", default=default_experiment)
    parser.add_argument("--award", choices=allowed_awards, default=default_award)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--min-train-season", type=int, default=DEFAULT_MIN_TRAIN_SEASON)
    parser.add_argument("--backtest-start", type=int, default=DEFAULT_BACKTEST_START)
    parser.add_argument("--backtest-end", type=int, default=DEFAULT_BACKTEST_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / default_experiment)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--base-weight-mode", choices=["none", "positive_boost", "team_weighted", "sqrt_class_balance"], default="positive_boost")
    parser.add_argument("--base-max-iter", type=int, default=250)
    parser.add_argument("--base-learning-rate", type=float, default=0.05)
    parser.add_argument("--base-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--base-l2-regularization", type=float, default=0.05)
    return parser


def selected_awards(award_arg: str) -> list[AwardSpec]:
    if award_arg == "all":
        return [AWARDS["all_nba"], AWARDS["all_rookie"]]
    return [AWARDS[award_arg]]


def load_award_dataset(spec: AwardSpec, feature_set: str | None = None, dataset_path: Path | None = None) -> pd.DataFrame:
    path = dataset_path or spec.dataset_path
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path, low_memory=False)
    required = ["SEASON_END_YEAR", "PLAYER_NAME", "PLAYER_NAME_KEY", spec.label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df[spec.label_col] = df[spec.label_col].fillna(0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)

    try:
        from features import add_season_rank_features
        df = add_season_rank_features(df)
    except Exception:
        pass

    if spec.name == "all_nba" and "IS_ALL_NBA_ELIGIBLE" in df.columns:
        df["IS_ALL_NBA_ELIGIBLE"] = df["IS_ALL_NBA_ELIGIBLE"].fillna(1).astype(int)

    if feature_set is not None:
        get_feature_columns(df, feature_set)

    return df


def get_feature_columns(df: pd.DataFrame, feature_set: str | None = None) -> list[str]:
    if feature_set:
        try:
            from features import get_feature_columns as project_get_feature_columns
            cols = project_get_feature_columns(df, feature_set=feature_set, verbose=False)
            if cols:
                return cols
        except Exception:
            pass

    banned_exact = {
        "SEASON_END_YEAR",
        "PLAYER_ID",
        "TEAM_ID",
        "ALL_NBA_LABEL",
        "ALL_ROOKIE_LABEL",
    }
    cols = []
    for col in df.columns:
        u = col.upper()
        if col in banned_exact or u.endswith("_ID"):
            continue
        if ("VOTE" in u or "LABEL" in u or "TARGET" in u) and not u.startswith("PREV_"):
            continue
        if col.startswith("TRUE_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)

    if not cols:
        raise RuntimeError("No numeric feature columns found.")
    return cols


def apply_all_nba_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    if "IS_ALL_NBA_ELIGIBLE" not in df.columns:
        return df
    return df[df["IS_ALL_NBA_ELIGIBLE"].fillna(1).astype(int) == 1].copy()


def make_hgb_classifier(max_iter: int, learning_rate: float, max_leaf_nodes: int, l2_regularization: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_leaf_nodes=max_leaf_nodes,
                l2_regularization=l2_regularization,
                random_state=RANDOM_STATE,
            )),
        ]
    )


def make_hgb_regressor(max_iter: int, learning_rate: float, max_leaf_nodes: int, l2_regularization: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_leaf_nodes=max_leaf_nodes,
                l2_regularization=l2_regularization,
                random_state=RANDOM_STATE,
            )),
        ]
    )


def make_sample_weight(y: pd.Series, mode: str, max_label: int) -> np.ndarray:
    y_values = y.fillna(0).astype(int).to_numpy()
    weights = np.ones(len(y_values), dtype=float)

    if mode == "none":
        return weights

    if mode == "positive_boost":
        weights[y_values > 0] = 6.0
        return weights

    if mode == "team_weighted":
        for label in range(1, max_label + 1):
            # higher labels are stronger teams
            weights[y_values == label] = 3.0 + label
        return weights

    if mode == "sqrt_class_balance":
        counts = pd.Series(y_values).value_counts().to_dict()
        if counts:
            max_count = max(counts.values())
            for cls, count in counts.items():
                weights[y_values == cls] = np.sqrt(max_count / max(count, 1))
        return weights

    raise ValueError(f"Unknown weight mode: {mode}")


def fit_base_classifier(train_df: pd.DataFrame, feature_cols: list[str], spec: AwardSpec, args) -> Pipeline:
    model = make_hgb_classifier(
        max_iter=args.base_max_iter,
        learning_rate=args.base_learning_rate,
        max_leaf_nodes=args.base_max_leaf_nodes,
        l2_regularization=args.base_l2_regularization,
    )
    y = train_df[spec.label_col].astype(int)
    w = make_sample_weight(y, args.base_weight_mode, spec.max_label)
    model.fit(train_df[feature_cols], y, model__sample_weight=w)
    return model


def add_classifier_scores(df: pd.DataFrame, model: Pipeline, feature_cols: list[str], spec: AwardSpec, prefix: str = "BASE") -> pd.DataFrame:
    out = df.copy()
    raw = model.predict_proba(out[feature_cols])
    classes = list(model.named_steps["model"].classes_)

    proba = np.zeros((len(out), spec.max_label + 1), dtype=float)
    for idx, cls in enumerate(classes):
        cls = int(cls)
        if 0 <= cls <= spec.max_label:
            proba[:, cls] = raw[:, idx]

    out[f"{prefix}_P_NONE"] = proba[:, 0]
    out[f"{prefix}_P_AWARD"] = proba[:, 1:].sum(axis=1)
    expected = np.zeros(len(out), dtype=float)
    for label in range(1, spec.max_label + 1):
        out[f"{prefix}_P_LABEL_{label}"] = proba[:, label]
        expected += label * proba[:, label]
    out[f"{prefix}_EXPECTED_LABEL"] = expected

    for pred_label in range(1, spec.max_label + 1):
        value = np.zeros(len(out), dtype=float)
        for true_label in range(1, spec.max_label + 1):
            diff = abs(pred_label - true_label)
            if diff == 0:
                points = 10.0
            elif diff == 1:
                points = 8.0
            elif diff == 2:
                points = 6.0
            else:
                points = 0.0
            value += points * proba[:, true_label]
        out[f"{prefix}_TEAM_VALUE_{pred_label}"] = value

    out[f"{prefix}_BEST_TEAM_VALUE"] = out[
        [f"{prefix}_TEAM_VALUE_{label}" for label in range(1, spec.max_label + 1)]
    ].max(axis=1)
    return out


def unique_by_score(df: pd.DataFrame, score_col: str, ascending: bool = False) -> pd.DataFrame:
    ranked = df.sort_values([score_col, "PLAYER_NAME"], ascending=[ascending, True]).copy()
    ranked = ranked.drop_duplicates(subset=["PLAYER_NAME_KEY"], keep="first")
    ranked = ranked.reset_index(drop=True)
    ranked["PRED_RANK"] = np.arange(1, len(ranked) + 1)
    return ranked


def select_top_candidates(df: pd.DataFrame, spec: AwardSpec, score_col: str) -> pd.DataFrame:
    return unique_by_score(df, score_col=score_col, ascending=False).head(spec.top_n).copy()


def assign_by_rank(selected: pd.DataFrame, spec: AwardSpec, order_col: str) -> pd.DataFrame:
    out = selected.sort_values([order_col, "PLAYER_NAME"], ascending=[False, True]).head(spec.top_n).copy()
    out = out.reset_index(drop=True)
    out["PREDICTED_LABEL"] = 0
    for idx, label in enumerate(spec.team_labels):
        start = idx * 5
        end = min(start + 5, len(out))
        out.loc[start:end - 1, "PREDICTED_LABEL"] = label
    return out


def score_prediction(prediction_df: pd.DataFrame, true_df: pd.DataFrame, spec: AwardSpec) -> dict:
    true_labels = dict(
        zip(
            true_df.loc[true_df[spec.label_col] > 0, "PLAYER_NAME_KEY"],
            true_df.loc[true_df[spec.label_col] > 0, spec.label_col],
        )
    )
    pred_keys = set(prediction_df["PLAYER_NAME_KEY"].tolist())
    true_keys = set(true_labels.keys())

    score = 0.0
    exact_by_label = {label: 0 for label in range(1, spec.max_label + 1)}

    for _, row in prediction_df.iterrows():
        pred_label = int(row["PREDICTED_LABEL"])
        true_label = int(true_labels.get(row["PLAYER_NAME_KEY"], 0))
        if pred_label <= 0 or true_label <= 0:
            continue
        diff = abs(pred_label - true_label)
        if diff == 0:
            score += 10
            exact_by_label[pred_label] += 1
        elif diff == 1:
            score += 8
        elif diff == 2:
            score += 6

    for label in range(1, spec.max_label + 1):
        cnt = exact_by_label[label]
        if cnt >= 5:
            score += 40
        elif cnt == 4:
            score += 20
        elif cnt == 3:
            score += 10
        elif cnt == 2:
            score += 5

    def exact_for_team_name(name: str) -> int:
        if name == "first":
            return exact_by_label.get(spec.max_label, 0)
        if name == "second":
            return exact_by_label.get(spec.max_label - 1, 0)
        if name == "third":
            return exact_by_label.get(1, 0)
        return 0

    return {
        "score": float(score),
        "hits": int(len(pred_keys & true_keys)),
        "exact_1st": int(exact_for_team_name("first")),
        "exact_2nd": int(exact_for_team_name("second")),
        "exact_3rd": int(exact_for_team_name("third")) if spec.max_label == 3 else 0,
    }


def evaluate_base_hgb_for_season(df: pd.DataFrame, feature_cols: list[str], spec: AwardSpec, args, season: int) -> tuple[pd.DataFrame, dict]:
    train_df = df[(df["SEASON_END_YEAR"] >= args.min_train_season) & (df["SEASON_END_YEAR"] < season)].copy()
    test_df = df[df["SEASON_END_YEAR"] == season].copy()
    if spec.name == "all_nba":
        test_df = apply_all_nba_eligibility(test_df)

    model = fit_base_classifier(train_df, feature_cols, spec, args)
    scored = add_classifier_scores(test_df, model, feature_cols, spec, prefix="BASE")
    selected = select_top_candidates(scored, spec, score_col="BASE_EXPECTED_LABEL")
    prediction = assign_by_rank(selected, spec, order_col="BASE_EXPECTED_LABEL")
    info = score_prediction(prediction, test_df, spec)
    return prediction, info


def label_vote_proxy(labels: pd.Series, spec: AwardSpec) -> pd.Series:
    labels = labels.fillna(0).astype(int)
    if spec.max_label == 3:
        return labels.map({0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0}).fillna(0.0).astype(float)
    return labels.map({0: 0.0, 1: 1.0, 2: 3.0}).fillna(0.0).astype(float)


def safe_minmax(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - lo) / (hi - lo)


def print_experiment_header(name: str, description: str, args, config: dict | None = None) -> None:
    if args.quiet:
        return
    print()
    print("=" * 96)
    print(name.upper())
    print("=" * 96)
    print(description)
    print()
    print(f"award={args.award}")
    print(f"window={args.backtest_start}-{args.backtest_end}")
    print(f"feature_set={args.feature_set}")
    print(f"output_dir={args.output_dir}")
    if config:
        print()
        print("config:")
        for key, value in config.items():
            print(f"  {key}: {value}")


def print_season_line(args, experiment: str, award: str, season: int, info: dict, base_info: dict | None = None, spec: AwardSpec | None = None) -> None:
    if args.quiet:
        return

    max_score = spec.max_score if spec else ""
    top_n = spec.top_n if spec else ""
    base_score = base_info["score"] if base_info else info.get("base_score", np.nan)
    gain = info["score"] - base_score if not pd.isna(base_score) else 0.0

    print(
        f"{experiment:<32} | {award:<10} | {season}: "
        f"score={info['score']:>6.1f}/{max_score:<3} | "
        f"base={base_score:>6.1f}/{max_score:<3} | "
        f"gain={gain:+6.1f} | "
        f"hits={info['hits']:>2}/{top_n:<2} | "
        f"exact=({info['exact_1st']}, {info['exact_2nd']}, {info['exact_3rd']})"
    )


def prediction_to_rows(prediction: pd.DataFrame, season: int, experiment: str, award: str, score_cols: list[str] | None = None) -> list[dict]:
    rows = []
    score_cols = score_cols or []
    for rank, (_, row) in enumerate(prediction.reset_index(drop=True).iterrows(), start=1):
        item = {
            "experiment": experiment,
            "award": award,
            "season": int(season),
            "rank": int(rank),
            "player_name": row.get("PLAYER_NAME", ""),
            "player_key": row.get("PLAYER_NAME_KEY", ""),
            "predicted_label": int(row.get("PREDICTED_LABEL", 0)),
        }
        for col in score_cols:
            if col in row.index:
                val = row[col]
                item[col] = None if pd.isna(val) else float(val)
        rows.append(item)
    return rows


def summarize_results(season_df: pd.DataFrame, experiment: str, award: str, spec: AwardSpec, config: dict, num_features: int) -> pd.DataFrame:
    if season_df.empty:
        return pd.DataFrame([{
            "experiment": experiment,
            "award": award,
            "score_mean": np.nan,
            "score_max": spec.max_score,
            "score_pct": np.nan,
            "base_score_mean": np.nan,
            "gain_mean": np.nan,
            "hits_mean": np.nan,
            "exact_1st_mean": np.nan,
            "exact_2nd_mean": np.nan,
            "exact_3rd_mean": np.nan,
            "num_seasons": 0,
            "num_features": num_features,
            "config_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
        }])

    base_score_mean = season_df["base_score"].mean() if "base_score" in season_df.columns else np.nan
    return pd.DataFrame([{
        "experiment": experiment,
        "award": award,
        "score_mean": float(season_df["score"].mean()),
        "score_max": int(spec.max_score),
        "score_pct": float(100.0 * season_df["score"].mean() / spec.max_score),
        "base_score_mean": float(base_score_mean) if not pd.isna(base_score_mean) else np.nan,
        "gain_mean": float(season_df["score"].mean() - base_score_mean) if not pd.isna(base_score_mean) else np.nan,
        "hits_mean": float(season_df["hits"].mean()),
        "exact_1st_mean": float(season_df["exact_1st"].mean()),
        "exact_2nd_mean": float(season_df["exact_2nd"].mean()),
        "exact_3rd_mean": float(season_df["exact_3rd"].mean()),
        "num_seasons": int(len(season_df)),
        "num_features": int(num_features),
        "config_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
    }])


def write_outputs(
    experiment: str,
    output_dir: Path,
    summary_parts: list[pd.DataFrame],
    season_parts: list[pd.DataFrame],
    prediction_rows: list[dict] | None = None,
    quiet: bool = False,
) -> tuple[Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    season_df = pd.concat(season_parts, ignore_index=True) if season_parts else pd.DataFrame()

    summary_path = output_dir / f"{experiment}_summary.csv"
    season_path = output_dir / f"{experiment}_season_results.csv"
    pred_path = None

    summary_df.to_csv(summary_path, index=False)
    season_df.to_csv(season_path, index=False)

    if prediction_rows is not None:
        pred_path = output_dir / f"{experiment}_predictions.csv"
        pd.DataFrame(prediction_rows).to_csv(pred_path, index=False)

    if not quiet:
        print()
        print("=" * 96)
        print("SUMMARY")
        print("=" * 96)
        show_cols = [
            "experiment", "award", "score_mean", "score_max", "score_pct",
            "base_score_mean", "gain_mean", "hits_mean",
            "exact_1st_mean", "exact_2nd_mean", "exact_3rd_mean",
        ]
        existing = [c for c in show_cols if c in summary_df.columns]
        if existing:
            print(summary_df[existing].to_string(index=False))
        print()
        print(f"[saved summary] {summary_path}")
        print(f"[saved seasons] {season_path}")
        if pred_path is not None:
            print(f"[saved predictions] {pred_path}")

    return summary_path, season_path, pred_path
