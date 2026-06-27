from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"
LEGACY_DIR = PROJECT_ROOT / "data" / "legacy"
MODELS_DIR = PROJECT_ROOT / "models"
CORE_OUTPUT_DIR = MODELS_DIR / "unified_best_experiments"
LEGACY_OUTPUT_DIR = MODELS_DIR / "legacy_experiments"
DEFAULT_OUTPUT_DIR = MODELS_DIR / "report_experiments"

SEASON_SCORE_MAX = 450

XGB_EXPERIMENTS = {
    "xgb_ranker": {
        "script": LEGACY_DIR / "train_xgb_ranker.py",
        "report": MODELS_DIR / "validation_report_xgb_ranker.json",
    },
    "xgb_regressor": {
        "script": LEGACY_DIR / "train_xgb_regressor.py",
        "report": MODELS_DIR / "validation_report_xgb_regressor.json",
    },
}

DEFAULT_EXPERIMENTS = ["core", "legacy", "xgb_ranker", "xgb_regressor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run report experiments, require every experiment to produce both All-NBA "
            "and All-Rookie season scores, and generate bar charts with mean-score lines."
        )
    )
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS, help="core, legacy, xgb_ranker, xgb_regressor, all")
    parser.add_argument("--backtest-start", type=int, default=2010)
    parser.add_argument("--backtest-end", type=int, default=2025)
    parser.add_argument("--min-train-season", type=int, default=2000)
    parser.add_argument("--feature-set", default="previous_team_share_allstar")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-run", action="store_true", help="Only collect existing outputs and regenerate plots.")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--clean-run-outputs", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--verbose-runs", action="store_true", help="Do not pass --quiet to legacy runner.")
    parser.add_argument("--no-xgb", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_experiments(raw: list[str], no_xgb: bool) -> list[str]:
    names = ["core", "legacy", "xgb_ranker", "xgb_regressor"] if "all" in raw else list(raw)
    if no_xgb:
        names = [x for x in names if not x.startswith("xgb_")]
    known = {"core", "legacy", "xgb_ranker", "xgb_regressor"}
    unknown = [x for x in names if x not in known]
    if unknown:
        raise SystemExit(f"Unknown experiments: {unknown}. Available: all, {sorted(known)}")
    out, seen = [], set()
    for name in names:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def run_cmd(name: str, cmd: list[str], continue_on_error: bool) -> None:
    print()
    print("=" * 100)
    print(f"RUN {name}")
    print("=" * 100)
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        message = f"{name} failed with return code {completed.returncode}"
        if continue_on_error:
            print(f"[WARN] {message}; continuing")
        else:
            raise SystemExit(message)
    else:
        print(f"[OK] {name}")


def run_core(args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(SRC_DIR / "run_best_experiments.py"), "--experiments", "core"]
    if args.clean_run_outputs:
        cmd.append("--clean-output")
    run_cmd("core_best_experiments", cmd, args.continue_on_error)


def run_legacy(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(LEGACY_DIR / "run_legacy_experiments.py"),
        "--experiments", "all",
        "--backtest-start", str(args.backtest_start),
        "--backtest-end", str(args.backtest_end),
        "--min-train-season", str(args.min_train_season),
        "--feature-set", args.feature_set,
        "--continue-on-error",
    ]
    if args.clean_run_outputs:
        cmd.append("--clean-output")
    if not args.verbose_runs:
        cmd.append("--quiet")
    run_cmd("legacy_experiments", cmd, args.continue_on_error)


def run_xgb(args: argparse.Namespace, name: str) -> None:
    spec = XGB_EXPERIMENTS[name]
    if not spec["script"].exists():
        print(f"[skip] {name}: missing script {spec['script']}")
        return
    run_cmd(name, [sys.executable, str(spec["script"])], args.continue_on_error)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warn] cannot read {path}: {exc}")
        return pd.DataFrame()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def core_seasons() -> pd.DataFrame:
    path = CORE_OUTPUT_DIR / "comparison_season_results.csv"
    df = load_csv(path)
    if df.empty:
        return df
    if not {"experiment", "season"}.issubset(df.columns):
        print(f"[warn] unexpected core season schema: {path}")
        return pd.DataFrame()
    out = df.copy()
    for col in ["all_nba_score", "all_rookie_score"]:
        if col not in out.columns:
            raise RuntimeError(f"Core output is missing required column {col}. Do not fill with another model.")
    out["season"] = out["season"].astype(int)
    out["source_group"] = "core"
    out["all_nba_score"] = pd.to_numeric(out["all_nba_score"], errors="coerce")
    out["all_rookie_score"] = pd.to_numeric(out["all_rookie_score"], errors="coerce")
    out["total_score"] = out["all_nba_score"] + out["all_rookie_score"]
    out["complete_450"] = out[["all_nba_score", "all_rookie_score"]].notna().all(axis=1)
    return out[["experiment", "season", "all_nba_score", "all_rookie_score", "total_score", "source_group", "complete_450"]]


def normalize_long_award(df: pd.DataFrame, source_group: str) -> pd.DataFrame:
    if df.empty or not {"award", "season", "score"}.issubset(df.columns):
        return pd.DataFrame()
    exp_col = "runner_experiment" if "runner_experiment" in df.columns else "experiment"
    if exp_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for (exp, season), g in df.groupby([exp_col, "season"], sort=True):
        by_award = {str(row["award"]): row for _, row in g.iterrows()}
        has_nba = "all_nba" in by_award
        has_rookie = "all_rookie" in by_award
        if not (has_nba and has_rookie):
            missing = []
            if not has_nba:
                missing.append("all_nba")
            if not has_rookie:
                missing.append("all_rookie")
            print(f"[warn] dropping incomplete season result for {exp} {season}: missing {missing}")
            continue
        nba_score = safe_float(by_award["all_nba"].get("score"))
        rookie_score = safe_float(by_award["all_rookie"].get("score"))
        rows.append({
            "experiment": str(exp),
            "season": int(season),
            "all_nba_score": nba_score,
            "all_rookie_score": rookie_score,
            "total_score": nba_score + rookie_score,
            "source_group": source_group,
            "complete_450": True,
        })
    return pd.DataFrame(rows)


def normalize_wide(df: pd.DataFrame, source_group: str) -> pd.DataFrame:
    if df.empty or not {"season", "experiment", "all_nba_score", "all_rookie_score"}.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    out["season"] = out["season"].astype(int)
    out["all_nba_score"] = pd.to_numeric(out["all_nba_score"], errors="coerce")
    out["all_rookie_score"] = pd.to_numeric(out["all_rookie_score"], errors="coerce")
    out["complete_450"] = out[["all_nba_score", "all_rookie_score"]].notna().all(axis=1)
    bad = out[~out["complete_450"]]
    if not bad.empty:
        for _, row in bad.iterrows():
            print(f"[warn] dropping incomplete wide result for {row['experiment']} {row['season']}")
    out = out[out["complete_450"]].copy()
    out["total_score"] = out["all_nba_score"] + out["all_rookie_score"]
    out["source_group"] = source_group
    return out[["experiment", "season", "all_nba_score", "all_rookie_score", "total_score", "source_group", "complete_450"]]


def legacy_seasons() -> pd.DataFrame:
    path = LEGACY_OUTPUT_DIR / "legacy_comparison_season_results.csv"
    df = load_csv(path)
    if df.empty:
        return df
    wide = normalize_wide(df, "legacy")
    if not wide.empty:
        return wide
    return normalize_long_award(df, "legacy")


def convert_xgb_report(report_path: Path, experiment: str, source_group: str = "legacy") -> pd.DataFrame:
    if not report_path.exists():
        print(f"[skip] {experiment}: missing report {report_path}")
        return pd.DataFrame()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] cannot read XGB report {report_path}: {exc}")
        return pd.DataFrame()

    rows = []
    for season in range(2010, 2026):
        pieces = {}
        for award in ["all_nba", "all_rookie"]:
            best = report.get(award, {}).get("best", {})
            for item in best.get("season_results", []):
                if int(item.get("season", -1)) == season:
                    pieces[award] = item
                    break
        if "all_nba" not in pieces or "all_rookie" not in pieces:
            print(f"[warn] dropping incomplete XGB result for {experiment} {season}")
            continue
        nba_score = safe_float(pieces["all_nba"].get("score"))
        rookie_score = safe_float(pieces["all_rookie"].get("score"))
        rows.append({
            "experiment": experiment,
            "season": season,
            "all_nba_score": nba_score,
            "all_rookie_score": rookie_score,
            "total_score": nba_score + rookie_score,
            "source_group": source_group,
            "complete_450": True,
        })
    return pd.DataFrame(rows)


def collect_all_outputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    core_df = core_seasons()
    if not core_df.empty:
        parts.append(core_df)
    legacy_df = legacy_seasons()
    if not legacy_df.empty:
        parts.append(legacy_df)
    for name in ["xgb_ranker", "xgb_regressor"]:
        report_df = convert_xgb_report(XGB_EXPERIMENTS[name]["report"], name)
        if not report_df.empty:
            parts.append(report_df)

    if not parts:
        raise RuntimeError("No complete /450 experiment season outputs found.")

    season_df = pd.concat(parts, ignore_index=True)
    season_df = season_df[(season_df["season"] >= args.backtest_start) & (season_df["season"] <= args.backtest_end)].copy()
    season_df["total_score"] = season_df["all_nba_score"] + season_df["all_rookie_score"]

    summary_rows = []
    for exp, g in season_df.groupby("experiment", sort=False):
        summary_rows.append({
            "experiment": exp,
            "source_group": ",".join(sorted(set(g["source_group"].astype(str)))),
            "num_seasons": int(g["season"].nunique()),
            "score_mean": float(g["total_score"].mean()),
            "score_max": SEASON_SCORE_MAX,
            "score_pct": float(g["total_score"].mean() / SEASON_SCORE_MAX * 100.0),
            "all_nba_score_mean": float(g["all_nba_score"].mean()),
            "all_rookie_score_mean": float(g["all_rookie_score"].mean()),
            "complete_450": bool(g["complete_450"].all()),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("score_mean", ascending=False).reset_index(drop=True)
    return season_df.sort_values(["experiment", "season"]).reset_index(drop=True), summary_df


def safe_filename(name: str) -> str:
    chars = []
    for ch in name.lower():
        chars.append(ch if (ch.isalnum() or ch in {"_", "-"}) else "_")
    return "".join(chars).strip("_") or "experiment"


def make_plots(season_df: pd.DataFrame, summary_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    for exp, g in season_df.groupby("experiment", sort=False):
        g = g.sort_values("season")
        mean_score = float(g["total_score"].mean())

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(g["season"].astype(str), g["total_score"])
        ax.axhline(mean_score, linestyle="--", linewidth=1.5, label=f"średnia = {mean_score:.2f}")
        ax.set_title(f"{exp}: wynik sezonowy / {SEASON_SCORE_MAX}")
        ax.set_xlabel("Sezon")
        ax.set_ylabel(f"Wynik / {SEASON_SCORE_MAX}")
        ax.set_ylim(0, SEASON_SCORE_MAX)
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{safe_filename(exp)}_season_scores.png", dpi=160)
        plt.close(fig)

    if summary_df.empty:
        return

    ordered = summary_df.sort_values("score_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(ordered))))
    ax.barh(ordered["experiment"], ordered["score_mean"])
    ax.set_title(f"Średni wynik modeli / {SEASON_SCORE_MAX}")
    ax.set_xlabel(f"Średni wynik / {SEASON_SCORE_MAX}")
    ax.set_xlim(0, SEASON_SCORE_MAX)
    fig.tight_layout()
    fig.savefig(plots_dir / "average_total_score_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(ordered))))
    ax.barh(ordered["experiment"], ordered["all_nba_score_mean"], label="All-NBA")
    ax.barh(
        ordered["experiment"],
        ordered["all_rookie_score_mean"],
        left=ordered["all_nba_score_mean"],
        label="All-Rookie",
    )
    ax.set_title("Składowe średniego wyniku")
    ax.set_xlabel(f"Średni wynik / {SEASON_SCORE_MAX}")
    ax.set_xlim(0, SEASON_SCORE_MAX)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "score_components_all_nba_all_rookie.png", dpi=160)
    plt.close(fig)

    if "hgb_classifier" in set(summary_df["experiment"]):
        baseline = float(summary_df.loc[summary_df["experiment"] == "hgb_classifier", "score_mean"].iloc[0])
        improvement = summary_df.copy()
        improvement["improvement"] = improvement["score_mean"] - baseline
        improvement = improvement.sort_values("improvement", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(improvement))))
        ax.barh(improvement["experiment"], improvement["improvement"])
        ax.axvline(0, linewidth=1)
        ax.set_title("Różnica względem HGB baseline")
        ax.set_xlabel("Zmiana średniego wyniku")
        fig.tight_layout()
        fig.savefig(plots_dir / "improvement_vs_hgb_baseline.png", dpi=160)
        plt.close(fig)

    top_experiments = summary_df.head(4)["experiment"].astype(str).tolist()
    if "hgb_classifier" in set(summary_df["experiment"]) and "hgb_classifier" not in top_experiments:
        top_experiments = top_experiments[:3] + ["hgb_classifier"]

    top_seasons = season_df[season_df["experiment"].isin(top_experiments)].copy()
    if not top_seasons.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        for exp, g in top_seasons.groupby("experiment", sort=False):
            g = g.sort_values("season")
            ax.plot(g["season"], g["total_score"], marker="o", linewidth=1.8, label=exp)
        ax.set_title(f"Wynik sezonowy najlepszych modeli / {SEASON_SCORE_MAX}")
        ax.set_xlabel("Sezon")
        ax.set_ylabel(f"Wynik / {SEASON_SCORE_MAX}")
        ax.set_ylim(0, SEASON_SCORE_MAX)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "top_models_season_lines.png", dpi=160)
        plt.close(fig)

    if "hgb_classifier" in set(season_df["experiment"]) and not summary_df.empty:
        best_experiment = str(summary_df.iloc[0]["experiment"])
        if best_experiment != "hgb_classifier":
            pivot = season_df[season_df["experiment"].isin([best_experiment, "hgb_classifier"])]
            pivot = pivot.pivot_table(index="season", columns="experiment", values="total_score", aggfunc="mean")
            if {best_experiment, "hgb_classifier"}.issubset(pivot.columns):
                diff = pivot[best_experiment] - pivot["hgb_classifier"]

                fig, ax = plt.subplots(figsize=(11, 5))
                ax.bar(diff.index.astype(str), diff.values)
                ax.axhline(0, linewidth=1)
                ax.set_title(f"{best_experiment}: różnica sezonowa względem HGB baseline")
                ax.set_xlabel("Sezon")
                ax.set_ylabel("Różnica punktów")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                fig.savefig(plots_dir / "best_model_vs_hgb_by_season.png", dpi=160)
                plt.close(fig)

def main() -> None:
    args = parse_args()
    experiments = resolve_experiments(args.experiments, args.no_xgb)

    if args.clean_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        if "core" in experiments:
            run_core(args)
        if "legacy" in experiments:
            run_legacy(args)
        for name in ["xgb_ranker", "xgb_regressor"]:
            if name in experiments:
                run_xgb(args, name)

    season_df, summary_df = collect_all_outputs(args)
    season_path = args.output_dir / "report_experiment_season_scores.csv"
    summary_path = args.output_dir / "report_experiment_summary.csv"
    season_df.to_csv(season_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if not args.no_plots:
        make_plots(season_df, summary_df, args.output_dir / "plots")

    print()
    print("=" * 100)
    print("REPORT EXPERIMENT SUMMARY")
    print("=" * 100)
    display = summary_df.copy()
    for col in ["score_mean", "score_pct", "all_nba_score_mean", "all_rookie_score_mean"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(2)
    print(display.to_string(index=False))
    print()
    print(f"[saved seasons] {season_path}")
    print(f"[saved summary] {summary_path}")
    if not args.no_plots:
        print(f"[saved plots]   {args.output_dir / 'plots'}")


if __name__ == "__main__":
    main()