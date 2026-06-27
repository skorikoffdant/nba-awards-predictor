from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from features import add_season_rank_features, get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
UNIFIED_RESULTS_DIR = MODELS_DIR / "unified_experiments"

RANDOM_STATE = 42
MIN_TRAIN_SEASON = 2000
DEFAULT_BACKTEST_START = 2010
DEFAULT_BACKTEST_END = 2025
TARGET_SEASON = 2026


@dataclass(frozen=True)
class AwardSpec:
    name: str
    dataset_path: Path
    label_col: str
    max_label: int
    top_n: int
    max_score: int
    default_pool_size: int

    @property
    def team_labels_desc(self) -> list[int]:
        return list(range(self.max_label, 0, -1))

    @property
    def slot_labels(self) -> list[int]:
        labels: list[int] = []
        for label in self.team_labels_desc:
            labels.extend([label] * 5)
        return labels


AWARDS = {
    "all_nba": AwardSpec(
        name="all_nba",
        dataset_path=PROCESSED_DIR / "all_nba_dataset.csv",
        label_col="ALL_NBA_LABEL",
        max_label=3,
        top_n=15,
        max_score=270,
        default_pool_size=20,
    ),
    "all_rookie": AwardSpec(
        name="all_rookie",
        dataset_path=PROCESSED_DIR / "all_rookie_dataset.csv",
        label_col="ALL_ROOKIE_LABEL",
        max_label=2,
        top_n=10,
        max_score=180,
        default_pool_size=15,
    ),
}


SUMMARY_COLUMNS = [
    "experiment",
    "award",
    "status",
    "backtest_start",
    "backtest_end",
    "seasons",
    "score_mean",
    "score_max",
    "score_pct",
    "hits_mean",
    "top_n",
    "exact_1st_mean",
    "exact_2nd_mean",
    "exact_3rd_mean",
    "features",
    "config",
    "summary_path",
    "season_results_path",
]


SEASON_COLUMNS = [
    "experiment",
    "award",
    "season",
    "score",
    "max_score",
    "score_pct",
    "hits",
    "top_n",
    "exact_1st",
    "exact_2nd",
    "exact_3rd",
    "team_1st_score",
    "team_2nd_score",
    "team_3rd_score",
]


def parse_common_args(description: str, default_experiment: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--award", choices=["all", "all_nba", "all_rookie"], default="all")
    parser.add_argument("--backtest-start", type=int, default=DEFAULT_BACKTEST_START)
    parser.add_argument("--backtest-end", type=int, default=DEFAULT_BACKTEST_END)
    parser.add_argument("--min-train-season", type=int, default=MIN_TRAIN_SEASON)
    parser.add_argument("--feature-set", default="previous_team_share_allstar")
    parser.add_argument("--output-dir", type=Path, default=UNIFIED_RESULTS_DIR / default_experiment)
    parser.add_argument("--experiment-name", default=default_experiment)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    return parser


def selected_awards(name: str) -> list[AwardSpec]:
    if name == "all":
        return [AWARDS["all_nba"], AWARDS["all_rookie"]]
    return [AWARDS[name]]


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_award_dataset(spec: AwardSpec, feature_set: str | None = None) -> pd.DataFrame:
    if not spec.dataset_path.exists():
        raise FileNotFoundError(
            f"Missing dataset for {spec.name}: {spec.dataset_path}. "
            "Run data preparation first or include data/processed in the project."
        )

    df = pd.read_csv(spec.dataset_path)
    required = ["SEASON_END_YEAR", "PLAYER_NAME", "PLAYER_NAME_KEY", spec.label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {spec.dataset_path}: {missing}")

    df["SEASON_END_YEAR"] = df["SEASON_END_YEAR"].astype(int)
    df[spec.label_col] = df[spec.label_col].fillna(0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = add_season_rank_features(df)

    if feature_set is not None:
        # Early validation. The returned list is not used here because some scripts
        # build multiple feature sets, but this gives a clear failure message.
        get_feature_columns(df, feature_set=feature_set, verbose=False)

    return df


def unique_by_score(df: pd.DataFrame, score_col: str, ascending: bool = False) -> pd.DataFrame:
    sort_cols = [score_col, "PLAYER_NAME_KEY"]
    ascending_flags = [ascending, True]
    if "PLAYER_ID" in df.columns:
        sort_cols = [score_col, "PLAYER_ID", "PLAYER_NAME_KEY"]
        ascending_flags = [ascending, True, True]

    ranked = df.sort_values(sort_cols, ascending=ascending_flags, kind="mergesort").copy()
    ranked = ranked.drop_duplicates(subset=["PLAYER_NAME_KEY"], keep="first")
    ranked = ranked.reset_index(drop=True)
    ranked["RANK"] = np.arange(1, len(ranked) + 1)
    return ranked


def make_hgb_classifier(
    max_iter: int = 250,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.05,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_hgb_regressor(
    max_iter: int = 250,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.05,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_sample_weight(y: pd.Series, mode: str, max_label: int) -> np.ndarray:
    values = y.fillna(0).astype(int).to_numpy()

    if mode == "none":
        return np.ones(len(values), dtype=float)

    if mode == "positive_boost":
        weights = np.ones(len(values), dtype=float)
        weights[values > 0] = 6.0
        return weights

    if mode == "team_weighted":
        weights = np.ones(len(values), dtype=float)
        for label in range(1, max_label + 1):
            weights[values == label] = 3.0 + label
        return weights

    if mode == "sqrt_class_balance":
        counts = pd.Series(values).value_counts().to_dict()
        max_count = max(counts.values())
        weights = np.ones(len(values), dtype=float)
        for cls, count in counts.items():
            weights[values == cls] = np.sqrt(max_count / count)
        return weights

    raise ValueError(f"Unknown weight mode: {mode}")


def proba_by_label(model: Pipeline, X: pd.DataFrame, max_label: int) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = model.named_steps["model"].classes_
    proba = np.zeros((len(X), max_label + 1), dtype=float)
    for src_idx, cls in enumerate(classes):
        cls = int(cls)
        if 0 <= cls <= max_label:
            proba[:, cls] = raw[:, src_idx]
    return proba


def points_for_team(predicted_label: int, true_label: int) -> int:
    if true_label <= 0:
        return 0
    diff = abs(int(predicted_label) - int(true_label))
    if diff == 0:
        return 10
    if diff == 1:
        return 8
    if diff == 2:
        return 6
    return 0


def add_classifier_scores(
    df: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
    spec: AwardSpec,
    prefix: str = "MODEL",
) -> pd.DataFrame:
    out = df.copy()
    proba = proba_by_label(model, out[feature_columns], max_label=spec.max_label)

    out[f"{prefix}_P_NONE"] = proba[:, 0]
    out[f"{prefix}_P_AWARD"] = proba[:, 1:].sum(axis=1)
    out[f"{prefix}_EXPECTED_LABEL"] = sum(
        label * proba[:, label] for label in range(1, spec.max_label + 1)
    )

    for label in range(1, spec.max_label + 1):
        out[f"{prefix}_P_LABEL_{label}"] = proba[:, label]

    for predicted_label in range(1, spec.max_label + 1):
        value = np.zeros(len(out), dtype=float)
        for true_label in range(1, spec.max_label + 1):
            value += points_for_team(predicted_label, true_label) * proba[:, true_label]
        out[f"{prefix}_TEAM_VALUE_{predicted_label}"] = value

    team_value_cols = [f"{prefix}_TEAM_VALUE_{label}" for label in range(1, spec.max_label + 1)]
    out[f"{prefix}_BEST_TEAM_VALUE"] = out[team_value_cols].max(axis=1)
    return out


def select_top_candidates(
    scored_df: pd.DataFrame,
    spec: AwardSpec,
    score_col: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    top_n = spec.top_n if top_n is None else int(top_n)
    ranked = unique_by_score(scored_df, score_col=score_col, ascending=False)
    return ranked.head(top_n).copy().reset_index(drop=True)


def assign_by_sort(
    selected_df: pd.DataFrame,
    spec: AwardSpec,
    sort_col: str,
) -> pd.DataFrame:
    out = selected_df.sort_values(
        [sort_col, "PLAYER_NAME_KEY"], ascending=[False, True], kind="mergesort"
    ).copy().reset_index(drop=True)
    out["PREDICTED_LABEL"] = spec.slot_labels[: len(out)]
    return out


def assign_by_hungarian(
    selected_df: pd.DataFrame,
    spec: AwardSpec,
    value_prefix: str = "MODEL",
) -> pd.DataFrame:
    selected = selected_df.copy().reset_index(drop=True)
    slot_labels = spec.slot_labels[: len(selected)]

    profit_rows = []
    for slot_label in slot_labels:
        col = f"{value_prefix}_TEAM_VALUE_{slot_label}"
        if col not in selected.columns:
            raise RuntimeError(f"Missing assignment value column: {col}")
        profit_rows.append(selected[col].to_numpy(dtype=float))

    profit_matrix = np.vstack(profit_rows)
    row_ind, col_ind = linear_sum_assignment(-profit_matrix)

    rows = []
    for row_idx, col_idx in sorted(zip(row_ind, col_ind), key=lambda x: x[0]):
        row = selected.iloc[col_idx].copy()
        row["PREDICTED_LABEL"] = slot_labels[row_idx]
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def score_prediction(prediction_df: pd.DataFrame, true_df: pd.DataFrame, spec: AwardSpec) -> dict:
    true_labels = dict(
        zip(
            true_df.loc[true_df[spec.label_col] > 0, "PLAYER_NAME_KEY"],
            true_df.loc[true_df[spec.label_col] > 0, spec.label_col].astype(int),
        )
    )

    result: dict[str, float | int] = {}
    total_score = 0
    exact_by_label = {}
    score_by_label = {}

    for label in spec.team_labels_desc:
        keys = prediction_df.loc[prediction_df["PREDICTED_LABEL"] == label, "PLAYER_NAME_KEY"].tolist()
        points = 0
        exact = 0
        for key in keys:
            true_label = int(true_labels.get(key, 0))
            points += points_for_team(label, true_label)
            if true_label == label:
                exact += 1
        bonus = {0: 0, 1: 0, 2: 5, 3: 10, 4: 20, 5: 40}.get(exact, 40)
        team_total = points + bonus
        total_score += team_total
        exact_by_label[label] = exact
        score_by_label[label] = team_total

    predicted_keys = set(prediction_df["PLAYER_NAME_KEY"].tolist())
    true_keys = set(true_labels.keys())

    result["score"] = int(total_score)
    result["max_score"] = int(spec.max_score)
    result["score_pct"] = float(total_score / spec.max_score)
    result["hits"] = int(len(predicted_keys & true_keys))
    result["top_n"] = int(spec.top_n)

    # Unified names: exact_1st / exact_2nd / exact_3rd. All-Rookie has exact_3rd = 0.
    for team_name, label in [("1st", spec.max_label), ("2nd", spec.max_label - 1), ("3rd", spec.max_label - 2)]:
        if label >= 1:
            result[f"exact_{team_name}"] = int(exact_by_label.get(label, 0))
            result[f"team_{team_name}_score"] = int(score_by_label.get(label, 0))
        else:
            result[f"exact_{team_name}"] = 0
            result[f"team_{team_name}_score"] = 0

    return result


def prediction_to_rows(
    prediction_df: pd.DataFrame,
    season: int,
    experiment: str,
    award: str,
    score_cols: Iterable[str] = (),
) -> list[dict]:
    rows = []
    cols = ["PLAYER_NAME", "PLAYER_NAME_KEY", "PREDICTED_LABEL"] + [c for c in score_cols if c in prediction_df.columns]
    for rank, row in enumerate(prediction_df.reset_index(drop=True).itertuples(index=False), start=1):
        row_dict = row._asdict()
        out = {
            "experiment": experiment,
            "award": award,
            "season": season,
            "rank": rank,
            "player": row_dict.get("PLAYER_NAME"),
            "player_key": row_dict.get("PLAYER_NAME_KEY"),
            "predicted_label": int(row_dict.get("PREDICTED_LABEL", 0)),
        }
        for col in score_cols:
            if col in row_dict:
                value = row_dict[col]
                if isinstance(value, (np.integer, np.floating)):
                    value = float(value)
                out[col] = value
        rows.append(out)
    return rows


def summarize_seasons(
    season_df: pd.DataFrame,
    experiment: str,
    award: str,
    spec: AwardSpec,
    config: dict,
    features_count: int | None,
    output_dir: Path,
) -> pd.DataFrame:
    if season_df.empty:
        row = {
            "experiment": experiment,
            "award": award,
            "status": "empty",
            "backtest_start": None,
            "backtest_end": None,
            "seasons": 0,
            "score_mean": np.nan,
            "score_max": spec.max_score,
            "score_pct": np.nan,
            "hits_mean": np.nan,
            "top_n": spec.top_n,
            "exact_1st_mean": np.nan,
            "exact_2nd_mean": np.nan,
            "exact_3rd_mean": np.nan,
            "features": features_count,
            "config": json.dumps(config, ensure_ascii=False, sort_keys=True),
            "summary_path": "",
            "season_results_path": "",
        }
        return pd.DataFrame([row], columns=SUMMARY_COLUMNS)

    row = {
        "experiment": experiment,
        "award": award,
        "status": "ok",
        "backtest_start": int(season_df["season"].min()),
        "backtest_end": int(season_df["season"].max()),
        "seasons": int(season_df["season"].nunique()),
        "score_mean": float(season_df["score"].mean()),
        "score_max": int(spec.max_score),
        "score_pct": float(season_df["score"].mean() / spec.max_score),
        "hits_mean": float(season_df["hits"].mean()),
        "top_n": int(spec.top_n),
        "exact_1st_mean": float(season_df["exact_1st"].mean()),
        "exact_2nd_mean": float(season_df["exact_2nd"].mean()),
        "exact_3rd_mean": float(season_df["exact_3rd"].mean()),
        "features": features_count,
        "config": json.dumps(config, ensure_ascii=False, sort_keys=True),
        "summary_path": str(output_dir / f"{experiment}_summary.csv"),
        "season_results_path": str(output_dir / f"{experiment}_season_results.csv"),
    }
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def write_experiment_outputs(
    experiment: str,
    output_dir: Path,
    summary_parts: list[pd.DataFrame],
    season_parts: list[pd.DataFrame],
    prediction_rows: list[dict] | None = None,
) -> tuple[Path, Path, Path | None]:
    ensure_output_dir(output_dir)

    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame(columns=SUMMARY_COLUMNS)
    season_df = pd.concat(season_parts, ignore_index=True) if season_parts else pd.DataFrame(columns=SEASON_COLUMNS)

    summary_path = output_dir / f"{experiment}_summary.csv"
    season_path = output_dir / f"{experiment}_season_results.csv"

    summary_df.to_csv(summary_path, index=False)
    season_df.to_csv(season_path, index=False)

    pred_path = None
    if prediction_rows is not None:
        pred_path = output_dir / f"{experiment}_predictions.csv"
        pd.DataFrame(prediction_rows).to_csv(pred_path, index=False)

    return summary_path, season_path, pred_path


def print_award_table(summary_df: pd.DataFrame, quiet: bool = False) -> None:
    if quiet:
        return
    cols = [
        "experiment",
        "award",
        "score_mean",
        "score_max",
        "score_pct",
        "hits_mean",
        "exact_1st_mean",
        "exact_2nd_mean",
        "exact_3rd_mean",
    ]
    out = summary_df[cols].copy()
    if not out.empty:
        out["score_pct"] = (100.0 * out["score_pct"]).round(2)
        for col in ["score_mean", "hits_mean", "exact_1st_mean", "exact_2nd_mean", "exact_3rd_mean"]:
            out[col] = out[col].round(2)
    print()
    print(out.to_string(index=False))


def print_season_line(
    quiet: bool,
    experiment: str,
    award: str,
    season: int,
    info: dict,
) -> None:
    if quiet:
        return
    print(
        f"{experiment:28s} | {award:10s} | {season}: "
        f"score={info['score']:>3}/{info['max_score']} | "
        f"hits={info['hits']:>2}/{info['top_n']} | "
        f"exact=({info['exact_1st']}, {info['exact_2nd']}, {info['exact_3rd']})",
        flush=True,
    )


def get_feature_cols(df: pd.DataFrame, feature_set: str) -> list[str]:
    cols = get_feature_columns(df, feature_set=feature_set, verbose=False)
    numeric = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric:
        raise RuntimeError(f"No numeric features for feature_set={feature_set}")
    return numeric


def add_linear_assignment_scores_from_scalar(
    df: pd.DataFrame,
    spec: AwardSpec,
    scalar_col: str,
    prefix: str = "MODEL",
) -> pd.DataFrame:
    out = df.copy()
    values = out[scalar_col].astype(float)
    for label in range(1, spec.max_label + 1):
        # For scalar ranking models, a higher scalar means a better team.
        # The assignment values are monotonic and let the generic assignment code work.
        out[f"{prefix}_TEAM_VALUE_{label}"] = values * float(label)
    out[f"{prefix}_EXPECTED_LABEL"] = values
    out[f"{prefix}_P_AWARD"] = values
    out[f"{prefix}_BEST_TEAM_VALUE"] = values
    return out


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weight-mode", choices=["none", "positive_boost", "team_weighted", "sqrt_class_balance"], default="positive_boost")
    parser.add_argument("--selection-score", choices=["expected_label", "p_award", "best_team_value"], default="expected_label")
    parser.add_argument("--assignment", choices=["sort", "hungarian"], default="sort")
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)


def selection_column(selection_score: str, prefix: str = "MODEL") -> str:
    if selection_score == "expected_label":
        return f"{prefix}_EXPECTED_LABEL"
    if selection_score == "p_award":
        return f"{prefix}_P_AWARD"
    if selection_score == "best_team_value":
        return f"{prefix}_BEST_TEAM_VALUE"
    raise ValueError(f"Unknown selection score: {selection_score}")


def assign_prediction(
    selected: pd.DataFrame,
    spec: AwardSpec,
    assignment: str,
    sort_col: str,
    value_prefix: str = "MODEL",
) -> pd.DataFrame:
    if assignment == "sort":
        return assign_by_sort(selected, spec=spec, sort_col=sort_col)
    if assignment == "hungarian":
        return assign_by_hungarian(selected, spec=spec, value_prefix=value_prefix)
    raise ValueError(f"Unknown assignment: {assignment}")


def safe_minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mn) / (mx - mn)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
