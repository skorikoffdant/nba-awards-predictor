from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = PROJECT_ROOT / "data" / "legacy"
SRC_DIR = PROJECT_ROOT / "data" / "src"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "legacy_experiments"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    script: Path
    extra_args: tuple[str, ...] = ()


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "all_rookie_reorderer": ExperimentSpec(
        name="all_rookie_reorderer",
        script=LEGACY_DIR / "train_all_rookie_reorderer.py",
    ),
    "candidate_pool_stage2": ExperimentSpec(
        name="candidate_pool_stage2",
        script=LEGACY_DIR / "train_candidate_pool_stage2.py",
    ),
    "top40_bubble_reranker": ExperimentSpec(
        name="top40_bubble_reranker",
        script=LEGACY_DIR / "experiment_top40_bubble_reranker.py",
    ),
    "top40_vote_support_reranker": ExperimentSpec(
        name="top40_vote_support_reranker",
        script=LEGACY_DIR / "experiment_top40_vote_support_reranker.py",
        extra_args=("--award", "all"),
    ),
    "all_nba_vote_regressor": ExperimentSpec(
        name="all_nba_vote_regressor",
        script=LEGACY_DIR / "train_all_nba_vote_regressor.py",
    ),
    "rank_ensemble": ExperimentSpec(
        name="rank_ensemble",
        script=LEGACY_DIR / "experiment_all_nba_rank_ensemble.py",
    ),
}

DEFAULT_EXPERIMENTS = [
    "all_rookie_reorderer",
    "candidate_pool_stage2",
    "top40_bubble_reranker",
    "top40_vote_support_reranker",
    "all_nba_vote_regressor",
    "rank_ensemble",
]


def resolve_experiment_names(raw: list[str]) -> list[str]:
    if raw == ["all"]:
        return DEFAULT_EXPERIMENTS

    unknown = [name for name in raw if name not in EXPERIMENTS]
    if unknown:
        available = ", ".join(["all"] + sorted(EXPERIMENTS))
        raise SystemExit(f"Unknown experiments: {unknown}. Available: {available}")

    return raw


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run_one(spec: ExperimentSpec, args: argparse.Namespace) -> int:
    if not spec.script.exists():
        print(f"[skip] {spec.name}: missing script {spec.script}")
        return 0 if args.skip_missing else 1

    out_dir = args.output_dir / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(spec.script),
        "--backtest-start",
        str(args.backtest_start),
        "--backtest-end",
        str(args.backtest_end),
        "--min-train-season",
        str(args.min_train_season),
        "--feature-set",
        args.feature_set,
        "--output-dir",
        str(out_dir),
        *spec.extra_args,
    ]

    if args.quiet:
        cmd.append("--quiet")
    if args.save_predictions:
        cmd.append("--save-predictions")

    print()
    print("=" * 96)
    print(f"RUN {spec.name}")
    print("=" * 96)
    print(" ".join(cmd))

    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if completed.returncode == 0:
        print(f"[OK] {spec.name}")
    else:
        print(f"[FAIL] {spec.name} returncode={completed.returncode}")

    return int(completed.returncode)


def add_source_columns(df: pd.DataFrame, experiment_name: str, source_path: Path) -> pd.DataFrame:
    df = df.copy()
    if "runner_experiment" not in df.columns:
        df.insert(0, "runner_experiment", experiment_name)
    if "source_file" not in df.columns:
        df.insert(1, "source_file", str(source_path))
    return df


def collect_outputs(output_dir: Path) -> tuple[Path | None, Path | None]:
    summary_parts = []
    season_parts = []

    for exp_dir in sorted(output_dir.iterdir() if output_dir.exists() else []):
        if not exp_dir.is_dir():
            continue

        experiment_name = exp_dir.name

        for path in sorted(exp_dir.glob("*_summary.csv")):
            try:
                summary_parts.append(add_source_columns(pd.read_csv(path), experiment_name, path))
            except Exception as exc:
                print(f"[warn] cannot read summary {path}: {exc}")

        for path in sorted(exp_dir.glob("*_season_results.csv")):
            try:
                season_parts.append(add_source_columns(pd.read_csv(path), experiment_name, path))
            except Exception as exc:
                print(f"[warn] cannot read season results {path}: {exc}")

    summary_path = None
    season_path = None

    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True)
        sort_cols = [col for col in ["score_mean", "gain_mean", "hits_mean"] if col in summary.columns]
        if sort_cols:
            summary = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        summary_path = output_dir / "legacy_comparison_summary.csv"
        summary.to_csv(summary_path, index=False)

    if season_parts:
        seasons = pd.concat(season_parts, ignore_index=True)
        sort_cols = [col for col in ["runner_experiment", "award", "season"] if col in seasons.columns]
        if sort_cols:
            seasons = seasons.sort_values(sort_cols)
        season_path = output_dir / "legacy_comparison_season_results.csv"
        seasons.to_csv(season_path, index=False)

    return summary_path, season_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legacy NBA Awards experiments and collect comparable CSV outputs.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help="Experiments to run, or 'all'.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backtest-start", type=int, default=2010)
    parser.add_argument("--backtest-end", type=int, default=2025)
    parser.add_argument("--min-train-season", type=int, default=2000)
    parser.add_argument("--feature-set", default="previous_team_share_allstar")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    names = resolve_experiment_names(args.experiments)

    if args.clean_output:
        remove_dir(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 96)
    print("LEGACY EXPERIMENT RUNNER")
    print("=" * 96)
    print(f"experiments={names}")
    print(f"window={args.backtest_start}-{args.backtest_end}")
    print(f"feature_set={args.feature_set}")
    print(f"output_dir={args.output_dir}")

    failures: list[tuple[str, int]] = []

    for name in names:
        code = run_one(EXPERIMENTS[name], args)
        if code != 0:
            failures.append((name, code))
            if not args.continue_on_error:
                break

    summary_path, season_path = collect_outputs(args.output_dir)

    print()
    print("=" * 96)
    print("COLLECTED OUTPUTS")
    print("=" * 96)
    if summary_path is not None:
        print(f"[saved summary] {summary_path}")
    else:
        print("[warn] no summary files collected")

    if season_path is not None:
        print(f"[saved seasons] {season_path}")
    else:
        print("[warn] no season result files collected")

    if failures:
        print()
        print("FAILURES")
        for name, code in failures:
            print(f"  {name}: returncode={code}")
        raise SystemExit(1)

    print("[OK] legacy experiments finished")


if __name__ == "__main__":
    main()
