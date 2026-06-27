from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _season_map(rows: list[dict[str, Any]], score_key: str = "score", hits_key: str = "top_hits") -> dict[int, dict[str, Any]]:
    out = {}
    for row in rows or []:
        season = int(row["season"])
        out[season] = dict(row)
        if "top_15_hits" in out[season] and hits_key not in out[season]:
            out[season][hits_key] = out[season]["top_15_hits"]
        if "top_10_hits" in out[season] and hits_key not in out[season]:
            out[season][hits_key] = out[season]["top_10_hits"]
    return out


def _extract_hgb_rookie_baseline(hgb_report_path: Path) -> dict[str, Any]:
    report = _json_load(hgb_report_path)
    best = report.get("best", report)
    rookie = best["all_rookie"]
    return rookie


def _team_exact(row: dict[str, Any], label: int) -> float:
    exact_by_label = row.get("exact_by_label")
    if isinstance(exact_by_label, dict):
        return float(exact_by_label.get(str(label), exact_by_label.get(label, 0.0)))
    if label == 3:
        return float(row.get("first_exact", 0.0))
    if label == 2:
        return float(row.get("second_exact", 0.0))
    if label == 1:
        return float(row.get("third_exact", 0.0))
    return 0.0


def _base_summary_row(
    *,
    experiment: str,
    source_report: Path,
    window: str,
    all_nba_score: float,
    all_nba_hits: float,
    all_rookie_score: float,
    all_rookie_hits: float,
    all_nba_first_exact: float | None = None,
    all_nba_second_exact: float | None = None,
    all_nba_third_exact: float | None = None,
    all_rookie_first_exact: float | None = None,
    all_rookie_second_exact: float | None = None,
    config: dict[str, Any] | None = None,
    note: str = "",
) -> dict[str, Any]:
    total = float(all_nba_score) + float(all_rookie_score)
    return {
        "experiment": experiment,
        "window": window,
        "all_nba_score": round(float(all_nba_score), 6),
        "all_nba_max": 270,
        "all_nba_hits": round(float(all_nba_hits), 6),
        "all_nba_hits_max": 15,
        "all_nba_first_exact": None if all_nba_first_exact is None else round(float(all_nba_first_exact), 6),
        "all_nba_second_exact": None if all_nba_second_exact is None else round(float(all_nba_second_exact), 6),
        "all_nba_third_exact": None if all_nba_third_exact is None else round(float(all_nba_third_exact), 6),
        "all_rookie_score": round(float(all_rookie_score), 6),
        "all_rookie_max": 180,
        "all_rookie_hits": round(float(all_rookie_hits), 6),
        "all_rookie_hits_max": 10,
        "all_rookie_first_exact": None if all_rookie_first_exact is None else round(float(all_rookie_first_exact), 6),
        "all_rookie_second_exact": None if all_rookie_second_exact is None else round(float(all_rookie_second_exact), 6),
        "total_score": round(total, 6),
        "total_max": 450,
        "total_pct": round(total / 450.0 * 100.0, 6),
        "source_report": str(source_report),
        "config_json": json.dumps(config or {}, ensure_ascii=False, sort_keys=True),
        "note": note,
    }


def _write_outputs(
    *,
    experiment: str,
    out_dir: Path,
    summary_row: dict[str, Any],
    season_rows: list[dict[str, Any]],
) -> None:
    out_dir = _ensure_dir(out_dir)
    pd.DataFrame([summary_row]).to_csv(out_dir / f"{experiment}_summary.csv", index=False)
    pd.DataFrame(season_rows).to_csv(out_dir / f"{experiment}_season_results.csv", index=False)
    print()
    print("=" * 80)
    print("UNIFIED EXPERIMENT OUTPUT")
    print("=" * 80)
    print(f"summary: {out_dir / f'{experiment}_summary.csv'}")
    print(f"seasons: {out_dir / f'{experiment}_season_results.csv'}")


def write_from_hgb_full_report(report_path: Path, out_dir: Path, experiment: str = "hgb_classifier") -> None:
    report_path = Path(report_path)
    report = _json_load(report_path)
    best = report["best"]
    nba = best["all_nba"]
    rookie = best["all_rookie"]

    nba_rows = _season_map(nba.get("season_results", []))
    rookie_rows = _season_map(rookie.get("season_results", []))
    seasons = sorted(set(nba_rows) | set(rookie_rows))
    season_rows = []
    for season in seasons:
        n = nba_rows.get(season, {})
        r = rookie_rows.get(season, {})
        season_rows.append({
            "experiment": experiment,
            "season": season,
            "all_nba_score": float(n.get("score", 0.0)),
            "all_nba_hits": float(n.get("top_hits", n.get("top_15_hits", 0.0))),
            "all_nba_first_exact": _team_exact(n, 3),
            "all_nba_second_exact": _team_exact(n, 2),
            "all_nba_third_exact": _team_exact(n, 1),
            "all_rookie_score": float(r.get("score", 0.0)),
            "all_rookie_hits": float(r.get("top_hits", r.get("top_10_hits", 0.0))),
            "all_rookie_first_exact": _team_exact(r, 2),
            "all_rookie_second_exact": _team_exact(r, 1),
            "total_score": float(n.get("score", 0.0)) + float(r.get("score", 0.0)),
        })

    summary = _base_summary_row(
        experiment=experiment,
        source_report=report_path,
        window=f"{report.get('backtest_start_season', 2010)}-{report.get('max_train_season', 2025)}",
        all_nba_score=nba["avg_score"],
        all_nba_hits=nba["avg_hits"],
        all_rookie_score=rookie["avg_score"],
        all_rookie_hits=rookie["avg_hits"],
        all_nba_first_exact=_mean([x["all_nba_first_exact"] for x in season_rows]),
        all_nba_second_exact=_mean([x["all_nba_second_exact"] for x in season_rows]),
        all_nba_third_exact=_mean([x["all_nba_third_exact"] for x in season_rows]),
        all_rookie_first_exact=_mean([x["all_rookie_first_exact"] for x in season_rows]),
        all_rookie_second_exact=_mean([x["all_rookie_second_exact"] for x in season_rows]),
        config=best.get("config", {}),
        note="Main HGB classifier evaluated on All-NBA and All-Rookie.",
    )
    _write_outputs(experiment=experiment, out_dir=out_dir, summary_row=summary, season_rows=season_rows)


def write_from_xgb_report(report_path: Path, out_dir: Path, experiment: str) -> None:
    report_path = Path(report_path)
    report = _json_load(report_path)
    nba = report["all_nba"]["best"]
    rookie = report["all_rookie"]["best"]
    nba_rows = _season_map(nba.get("season_results", []))
    rookie_rows = _season_map(rookie.get("season_results", []))
    seasons = sorted(set(nba_rows) | set(rookie_rows))
    season_rows = []
    for season in seasons:
        n = nba_rows.get(season, {})
        r = rookie_rows.get(season, {})
        season_rows.append({
            "experiment": experiment,
            "season": season,
            "all_nba_score": float(n.get("score", 0.0)),
            "all_nba_hits": float(n.get("top_hits", n.get("top_15_hits", 0.0))),
            "all_rookie_score": float(r.get("score", 0.0)),
            "all_rookie_hits": float(r.get("top_hits", r.get("top_10_hits", 0.0))),
            "total_score": float(n.get("score", 0.0)) + float(r.get("score", 0.0)),
        })
    summary = _base_summary_row(
        experiment=experiment,
        source_report=report_path,
        window=f"{report.get('backtest_start_season', 2010)}-{report.get('max_train_season', 2025)}",
        all_nba_score=nba["avg_score"],
        all_nba_hits=nba.get("avg_top_hits", nba.get("avg_hits", 0.0)),
        all_rookie_score=rookie["avg_score"],
        all_rookie_hits=rookie.get("avg_top_hits", rookie.get("avg_hits", 0.0)),
        config={"all_nba": nba.get("config", {}), "all_rookie": rookie.get("config", {})},
        note="XGBoost line evaluated on All-NBA and All-Rookie.",
    )
    _write_outputs(experiment=experiment, out_dir=out_dir, summary_row=summary, season_rows=season_rows)


def write_from_all_nba_stage2_report(
    report_path: Path,
    hgb_report_path: Path,
    out_dir: Path,
    experiment: str,
    best_key: str = "best",
) -> None:
    report_path = Path(report_path)
    hgb_report_path = Path(hgb_report_path)
    report = _json_load(report_path)
    best = report[best_key]
    rookie = _extract_hgb_rookie_baseline(hgb_report_path)
    rookie_rows = _season_map(rookie.get("season_results", []))
    nba_rows = _season_map(best.get("season_results", []))
    seasons = sorted(set(nba_rows) | set(rookie_rows))
    season_rows = []
    for season in seasons:
        n = nba_rows.get(season, {})
        r = rookie_rows.get(season, {})
        season_rows.append({
            "experiment": experiment,
            "season": season,
            "all_nba_score": float(n.get("score", 0.0)),
            "all_nba_hits": float(n.get("top_hits", n.get("top_15_hits", 0.0))),
            "all_nba_first_exact": float(n.get("first_exact", 0.0)),
            "all_nba_second_exact": float(n.get("second_exact", 0.0)),
            "all_nba_third_exact": float(n.get("third_exact", 0.0)),
            "all_rookie_score": float(r.get("score", 0.0)),
            "all_rookie_hits": float(r.get("top_hits", r.get("top_10_hits", 0.0))),
            "all_rookie_first_exact": _team_exact(r, 2),
            "all_rookie_second_exact": _team_exact(r, 1),
            "total_score": float(n.get("score", 0.0)) + float(r.get("score", 0.0)),
        })
    summary = _base_summary_row(
        experiment=experiment,
        source_report=report_path,
        window="2010-2025",
        all_nba_score=best.get("avg_score", best.get("score", 0.0)),
        all_nba_hits=best.get("avg_hits", best.get("hits", 0.0)),
        all_rookie_score=rookie["avg_score"],
        all_rookie_hits=rookie["avg_hits"],
        all_nba_first_exact=best.get("avg_first_exact", best.get("first_exact")),
        all_nba_second_exact=best.get("avg_second_exact", best.get("second_exact")),
        all_nba_third_exact=best.get("avg_third_exact", best.get("third_exact")),
        all_rookie_first_exact=_mean([x["all_rookie_first_exact"] for x in season_rows]),
        all_rookie_second_exact=_mean([x["all_rookie_second_exact"] for x in season_rows]),
        config=best.get("config", best),
        note="All-NBA Stage 2 line; All-Rookie component is the HGB baseline for the same seasons.",
    )
    _write_outputs(experiment=experiment, out_dir=out_dir, summary_row=summary, season_rows=season_rows)


def _score_selected_rows(selected: pd.DataFrame) -> dict[str, float]:
    true_by_pred = {3: [], 2: [], 1: []}
    for _, row in selected.iterrows():
        pred = int(row["predicted_label"])
        true = int(row["true_label"])
        if pred in true_by_pred:
            true_by_pred[pred].append(true)

    def team_score(pred_label: int, true_labels: list[int]) -> tuple[float, float]:
        points = 0.0
        exact = 0
        for true in true_labels:
            if true == 0:
                continue
            diff = abs(pred_label - true)
            if diff == 0:
                points += 10
                exact += 1
            elif diff == 1:
                points += 8
            elif diff == 2:
                points += 6
        bonus = {0: 0, 1: 0, 2: 5, 3: 10, 4: 20, 5: 40}.get(exact, 40)
        return points + bonus, exact

    s1, e1 = team_score(3, true_by_pred[3])
    s2, e2 = team_score(2, true_by_pred[2])
    s3, e3 = team_score(1, true_by_pred[1])
    true_keys = set(selected.loc[selected["true_label"].astype(int) > 0, "player_key"])
    pred_keys = set(selected["player_key"])
    return {
        "score": s1 + s2 + s3,
        "top_hits": float(len(pred_keys & true_keys)),
        "first_exact": float(e1),
        "second_exact": float(e2),
        "third_exact": float(e3),
    }


def write_from_stage1_swap_report(
    report_path: Path,
    predictions_path: Path,
    hgb_report_path: Path,
    out_dir: Path,
    experiment: str = "stage1_swap_selector",
) -> None:
    report_path = Path(report_path)
    predictions_path = Path(predictions_path)
    hgb_report_path = Path(hgb_report_path)
    report = _json_load(report_path)
    best = report["best_by_score"]
    rookie = _extract_hgb_rookie_baseline(hgb_report_path)
    rookie_rows = _season_map(rookie.get("season_results", []))

    season_rows = []
    if predictions_path.exists():
        pred = pd.read_csv(predictions_path)
        skip_cols = {"score", "hits", "first_exact", "second_exact", "third_exact", "min_hits", "max_hits"}
        for col, val in best.items():
            if col not in pred.columns or col in skip_cols:
                continue
            if pd.isna(val):
                pred = pred[pred[col].isna()]
                continue
            if pd.api.types.is_numeric_dtype(pred[col]):
                pred = pred[np.isclose(pd.to_numeric(pred[col], errors="coerce"), float(val), equal_nan=False)]
            elif pred[col].dtype == bool:
                pred = pred[pred[col] == bool(val)]
            else:
                pred = pred[pred[col].astype(str) == str(val)]
        if {"score", "top_hits", "season"}.issubset(pred.columns):
            for _, row in pred.sort_values("season").iterrows():
                season = int(row["season"])
                r = rookie_rows.get(season, {})
                season_rows.append({
                    "experiment": experiment,
                    "season": season,
                    "all_nba_score": float(row.get("score", 0.0)),
                    "all_nba_hits": float(row.get("top_hits", 0.0)),
                    "all_nba_first_exact": float(row.get("first_exact", 0.0)),
                    "all_nba_second_exact": float(row.get("second_exact", 0.0)),
                    "all_nba_third_exact": float(row.get("third_exact", 0.0)),
                    "all_rookie_score": float(r.get("score", 0.0)),
                    "all_rookie_hits": float(r.get("top_hits", r.get("top_10_hits", 0.0))),
                    "all_rookie_first_exact": _team_exact(r, 2),
                    "all_rookie_second_exact": _team_exact(r, 1),
                    "total_score": float(row.get("score", 0.0)) + float(r.get("score", 0.0)),
                })
        else:
            for season, g in pred.groupby("season"):
                n = _score_selected_rows(g)
                r = rookie_rows.get(int(season), {})
                season_rows.append({
                    "experiment": experiment,
                    "season": int(season),
                    "all_nba_score": n["score"],
                    "all_nba_hits": n["top_hits"],
                    "all_nba_first_exact": n["first_exact"],
                    "all_nba_second_exact": n["second_exact"],
                    "all_nba_third_exact": n["third_exact"],
                    "all_rookie_score": float(r.get("score", 0.0)),
                    "all_rookie_hits": float(r.get("top_hits", r.get("top_10_hits", 0.0))),
                    "all_rookie_first_exact": _team_exact(r, 2),
                    "all_rookie_second_exact": _team_exact(r, 1),
                    "total_score": n["score"] + float(r.get("score", 0.0)),
                })

    if not season_rows:
        # Fallback: summary only.
        season_rows = []

    summary = _base_summary_row(
        experiment=experiment,
        source_report=report_path,
        window="2010-2025",
        all_nba_score=best["score"],
        all_nba_hits=best["hits"],
        all_rookie_score=rookie["avg_score"],
        all_rookie_hits=rookie["avg_hits"],
        all_nba_first_exact=best.get("first_exact"),
        all_nba_second_exact=best.get("second_exact"),
        all_nba_third_exact=best.get("third_exact"),
        all_rookie_first_exact=_mean([x.get("all_rookie_first_exact", 0.0) for x in season_rows]) if season_rows else None,
        all_rookie_second_exact=_mean([x.get("all_rookie_second_exact", 0.0) for x in season_rows]) if season_rows else None,
        config=best,
        note="Conservative All-NBA Stage 1 swap selector; All-Rookie component is the HGB baseline.",
    )
    _write_outputs(experiment=experiment, out_dir=out_dir, summary_row=summary, season_rows=season_rows)


def _is_wide_summary(df: pd.DataFrame) -> bool:
    return {"experiment", "total_score", "all_nba_score", "all_rookie_score"}.issubset(df.columns)


def _is_long_award_summary(df: pd.DataFrame) -> bool:
    return {"experiment", "award", "score_mean", "score_max", "hits_mean"}.issubset(df.columns)


def _convert_long_award_summary(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    rows = []
    for experiment, g in df.groupby("experiment"):
        by_award = {str(row["award"]): row for _, row in g.iterrows()}
        nba = by_award.get("all_nba", {})
        rookie = by_award.get("all_rookie", {})
        all_nba_score = float(nba.get("score_mean", 0.0)) if len(nba) else 0.0
        all_rookie_score = float(rookie.get("score_mean", 0.0)) if len(rookie) else 0.0
        total = all_nba_score + all_rookie_score
        rows.append({
            "experiment": experiment,
            "window": f"{int(df['backtest_start'].min())}-{int(df['backtest_end'].max())}" if {"backtest_start", "backtest_end"}.issubset(df.columns) else "",
            "all_nba_score": all_nba_score,
            "all_nba_max": 270,
            "all_nba_hits": float(nba.get("hits_mean", 0.0)) if len(nba) else 0.0,
            "all_nba_hits_max": 15,
            "all_nba_first_exact": float(nba.get("exact_1st_mean", np.nan)) if len(nba) else np.nan,
            "all_nba_second_exact": float(nba.get("exact_2nd_mean", np.nan)) if len(nba) else np.nan,
            "all_nba_third_exact": float(nba.get("exact_3rd_mean", np.nan)) if len(nba) else np.nan,
            "all_rookie_score": all_rookie_score,
            "all_rookie_max": 180,
            "all_rookie_hits": float(rookie.get("hits_mean", 0.0)) if len(rookie) else 0.0,
            "all_rookie_hits_max": 10,
            "all_rookie_first_exact": float(rookie.get("exact_1st_mean", np.nan)) if len(rookie) else np.nan,
            "all_rookie_second_exact": float(rookie.get("exact_2nd_mean", np.nan)) if len(rookie) else np.nan,
            "total_score": total,
            "total_max": 450,
            "total_pct": total / 450.0 * 100.0,
            "source_report": str(source_path),
            "config_json": "",
            "note": "Converted from per-award unified summary.",
        })
    return pd.DataFrame(rows)


def _convert_long_award_seasons(df: pd.DataFrame) -> pd.DataFrame:
    if not {"experiment", "award", "season", "score", "hits"}.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    for (experiment, season), g in df.groupby(["experiment", "season"]):
        by_award = {str(row["award"]): row for _, row in g.iterrows()}
        nba = by_award.get("all_nba", {})
        rookie = by_award.get("all_rookie", {})
        nscore = float(nba.get("score", 0.0)) if len(nba) else 0.0
        rscore = float(rookie.get("score", 0.0)) if len(rookie) else 0.0
        rows.append({
            "experiment": experiment,
            "season": int(season),
            "all_nba_score": nscore,
            "all_nba_hits": float(nba.get("hits", 0.0)) if len(nba) else 0.0,
            "all_nba_first_exact": float(nba.get("exact_1st", np.nan)) if len(nba) else np.nan,
            "all_nba_second_exact": float(nba.get("exact_2nd", np.nan)) if len(nba) else np.nan,
            "all_nba_third_exact": float(nba.get("exact_3rd", np.nan)) if len(nba) else np.nan,
            "all_rookie_score": rscore,
            "all_rookie_hits": float(rookie.get("hits", 0.0)) if len(rookie) else 0.0,
            "all_rookie_first_exact": float(rookie.get("exact_1st", np.nan)) if len(rookie) else np.nan,
            "all_rookie_second_exact": float(rookie.get("exact_2nd", np.nan)) if len(rookie) else np.nan,
            "total_score": nscore + rscore,
        })
    return pd.DataFrame(rows)


def _load_hgb_rookie_rows_for_collect(out_dir: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    hgb_report = out_dir.parents[0] / "hgb_classifier_full_score_report.json"
    if not hgb_report.exists():
        hgb_report = Path("models/hgb_classifier_full_score_report.json")
    if not hgb_report.exists():
        return {}, {}
    rookie = _extract_hgb_rookie_baseline(hgb_report)
    return rookie, _season_map(rookie.get("season_results", []))


def _convert_bubble_summary(summary_path: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    if df.empty or "mean_after_score" not in df.columns:
        return pd.DataFrame()
    best = df.sort_values(
        [c for c in ["mean_after_hits", "mean_after_score", "mean_hit_gain", "mean_score_gain"] if c in df.columns],
        ascending=False,
    ).iloc[0]
    rookie, _ = _load_hgb_rookie_rows_for_collect(out_dir)
    rookie_score = float(rookie.get("avg_score", 0.0)) if rookie else 0.0
    rookie_hits = float(rookie.get("avg_hits", 0.0)) if rookie else 0.0
    nba_score = float(best.get("mean_after_score", 0.0))
    total = nba_score + rookie_score
    return pd.DataFrame([{
        "experiment": "top40_bubble_reranker",
        "window": "from_predictions_file",
        "all_nba_score": nba_score,
        "all_nba_max": 270,
        "all_nba_hits": float(best.get("mean_after_hits", 0.0)),
        "all_nba_hits_max": 15,
        "all_nba_first_exact": np.nan,
        "all_nba_second_exact": np.nan,
        "all_nba_third_exact": np.nan,
        "all_rookie_score": rookie_score,
        "all_rookie_max": 180,
        "all_rookie_hits": rookie_hits,
        "all_rookie_hits_max": 10,
        "all_rookie_first_exact": np.nan,
        "all_rookie_second_exact": np.nan,
        "total_score": total,
        "total_max": 450,
        "total_pct": total / 450.0 * 100.0,
        "source_report": str(summary_path),
        "config_json": best.to_json(force_ascii=False),
        "note": "Converted from top-40 bubble reranker output. All-Rookie component is HGB baseline.",
    }])


def _convert_bubble_seasons(scores_path: Path, out_dir: Path) -> pd.DataFrame:
    if not scores_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(scores_path)
    if df.empty or "after_score" not in df.columns:
        return pd.DataFrame()
    _, rookie_rows = _load_hgb_rookie_rows_for_collect(out_dir)
    rows = []
    for _, row in df.iterrows():
        season = int(row["season"])
        r = rookie_rows.get(season, {})
        rows.append({
            "experiment": "top40_bubble_reranker",
            "season": season,
            "all_nba_score": float(row.get("after_score", 0.0)),
            "all_nba_hits": float(row.get("after_hits", 0.0)),
            "all_nba_first_exact": float(row.get("after_exact_1st", np.nan)),
            "all_nba_second_exact": float(row.get("after_exact_2nd", np.nan)),
            "all_nba_third_exact": float(row.get("after_exact_3rd", np.nan)),
            "all_rookie_score": float(r.get("score", 0.0)),
            "all_rookie_hits": float(r.get("top_hits", r.get("top_10_hits", 0.0))),
            "all_rookie_first_exact": _team_exact(r, 2) if r else np.nan,
            "all_rookie_second_exact": _team_exact(r, 1) if r else np.nan,
            "total_score": float(row.get("after_score", 0.0)) + float(r.get("score", 0.0)),
        })
    return pd.DataFrame(rows)


def collect_unified_outputs(out_dir: Path, output_prefix: str = "comparison") -> None:
    out_dir = Path(out_dir)
    summaries = []
    seasons = []

    for path in sorted(out_dir.rglob("*_summary.csv")):
        if path.name.startswith(output_prefix):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if _is_wide_summary(df):
            summaries.append(df)
        elif _is_long_award_summary(df):
            summaries.append(_convert_long_award_summary(df, path))
        elif "mean_after_score" in df.columns:
            summaries.append(_convert_bubble_summary(path, out_dir))

    for path in sorted(out_dir.rglob("*_season_results.csv")):
        if path.name.startswith(output_prefix):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if {"experiment", "season", "all_nba_score", "all_rookie_score", "total_score"}.issubset(df.columns):
            seasons.append(df)
        elif {"experiment", "award", "season", "score", "hits"}.issubset(df.columns):
            seasons.append(_convert_long_award_seasons(df))

    for path in sorted(out_dir.rglob("*best_season_scores.csv")):
        if "top40_bubble" in path.name:
            converted = _convert_bubble_seasons(path, out_dir)
            if not converted.empty:
                seasons.append(converted)

    if summaries:
        summary = pd.concat([df for df in summaries if df is not None and not df.empty], ignore_index=True)
        if not summary.empty and "total_score" in summary.columns:
            summary = summary.sort_values("total_score", ascending=False)
        summary.to_csv(out_dir / f"{output_prefix}_summary.csv", index=False)
    if seasons:
        non_empty = [df for df in seasons if df is not None and not df.empty]
        if non_empty:
            pd.concat(non_empty, ignore_index=True).to_csv(out_dir / f"{output_prefix}_season_results.csv", index=False)
