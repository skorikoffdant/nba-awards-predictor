from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from unified_best_output import collect_unified_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "src"
MODELS_DIR = PROJECT_ROOT / "models"
UNIFIED_OUTPUT_DIR = MODELS_DIR / "unified_best_experiments"

CORE_EXPERIMENTS = [
    "hgb_classifier",
    "team_reorderer",
    "team_pairwise",
]

EXTRA_EXPERIMENTS = [
    "stage1_swap_selector",
    "topk_vote_support_reranker",
    "top40_bubble_reranker",
    "xgb_ranker",
    "xgb_regressor",
]

ALL_EXPERIMENTS = CORE_EXPERIMENTS + EXTRA_EXPERIMENTS


def command_for(name: str) -> list[str]:
    if name == "hgb_classifier":
        return [
            str(SRC_DIR / "train_hgb_classifier_full_score.py"),
            "--feature-set", "previous_team_share_allstar",
            "--weight-mode", "positive_boost",
            "--selection-mode", "expected_label",
            "--team-assignment-mode", "sort_expected_label",
            "--pool-filter", "none",
            "--max-iter", "250",
            "--learning-rate", "0.05",
            "--max-leaf-nodes", "31",
            "--l2-regularization", "0.05",
        ]

    if name == "team_reorderer":
        return [
            str(SRC_DIR / "train_team_reorderer.py"),
            "--base-feature-set", "previous_team_share_allstar",
            "--base-weight-mode", "positive_boost",
            "--base-selection-score-col", "EXPECTED_LABEL",
            "--base-max-iter", "250",
            "--base-learning-rate", "0.05",
            "--base-max-leaf-nodes", "31",
            "--base-l2-regularization", "0.05",
            "--candidate-pool-start-season", "2005",
            "--reorderer-model-types", "classifier",
            "--reorderer-max-iters", "150",
            "--reorderer-learning-rates", "0.03",
            "--reorderer-max-leaf-nodes", "15",
            "--reorderer-l2-regularization", "0.05",
            "--reorderer-weight-modes", "team_weighted",
            "--reorderer-target-modes", "spaced",
            "--blend-weights", "0.25",
            "--use-original-features",
        ]

    if name == "team_pairwise":
        return [
            str(SRC_DIR / "train_team_pairwise.py"),
            "--base-feature-set", "previous_team_share_allstar",
            "--base-weight-mode", "positive_boost",
            "--base-selection-score-col", "EXPECTED_LABEL",
            "--base-max-iter", "250",
            "--base-learning-rate", "0.05",
            "--base-max-leaf-nodes", "31",
            "--base-l2-regularization", "0.05",
            "--candidate-pool-start-season", "2005",
            "--pairwise-max-iters", "80",
            "--pairwise-learning-rates", "0.035",
            "--pairwise-max-leaf-nodes", "15",
            "--pairwise-l2-regularization", "0.05",
            "--pair-weight-modes", "team_gap",
            "--blend-weights", "0.25",
            "--use-original-features",
        ]

    if name == "stage1_swap_selector":
        return [
            str(SRC_DIR / "train_stage1_swap_selector.py"),
            "--primary-feature-set", "previous_team_share_allstar",
            "--base-weight-mode", "positive_boost",
            "--base-max-iter", "250",
            "--base-learning-rate", "0.05",
            "--base-max-leaf-nodes", "31",
            "--base-l2-regularization", "0.05",
            "--candidate-pool-start-season", "2005",
            "--pool-sizes", "20",
            "--selection-modes", "swap",
            "--remove-zone-starts", "15",
            "--max-swaps", "1",
            "--margins", "0.01",
            "--selector-blends", "0.5",
            "--selector-max-iters", "120",
            "--selector-learning-rates", "0.03",
            "--selector-max-leaf-nodes", "15",
            "--selector-l2-regularization", "0.05",
            "--selector-weight-modes", "positive_3",
            "--assignment-score-cols", "base",
        ]

    if name == "xgb_ranker":
        return [str(SRC_DIR / "train_xgb_ranker.py")]

    if name == "xgb_regressor":
        return [str(SRC_DIR / "train_xgb_regressor.py")]

    if name == "topk_vote_support_reranker":
        return [
            str(SRC_DIR / "experiment_top40_vote_support_reranker.py"),
            "--award", "all",
            "--backtest-start", "2010",
            "--backtest-end", "2025",
            "--output-dir", str(UNIFIED_OUTPUT_DIR / "topk_vote_support_reranker"),
        ]

    if name == "top40_bubble_reranker":
        return [
            str(SRC_DIR / "experiment_top40_bubble_reranker.py"),
            "--output-dir", str(UNIFIED_OUTPUT_DIR / "top40_bubble_reranker"),
        ]

    raise KeyError(name)


def run_command(name: str) -> None:
    cmd = [sys.executable] + command_for(name)

    print()
    print("=" * 80)
    print(f"RUN EXPERIMENT: {name}")
    print("=" * 80)
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def resolve_experiments(raw: list[str]) -> list[str]:
    if raw == ["core"]:
        return list(CORE_EXPERIMENTS)
    if raw == ["all"]:
        return list(ALL_EXPERIMENTS)
    return list(raw)


def ordered_for_dependencies(experiments: list[str]) -> list[str]:
    ordered = []

    needs_hgb = {
        "team_reorderer",
        "team_pairwise",
        "stage1_swap_selector",
        "top40_bubble_reranker",
    }

    if any(name in experiments for name in needs_hgb):
        ordered.append("hgb_classifier")

    for name in experiments:
        if name not in ordered:
            ordered.append(name)

    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["core"],
        help=(
            "core, all, or explicit names: "
            "hgb_classifier team_reorderer team_pairwise "
            "stage1_swap_selector topk_vote_support_reranker top40_bubble_reranker "
            "xgb_ranker xgb_regressor"
        ),
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only collect already generated *_summary.csv and *_season_results.csv files.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove old CSV outputs before running.",
    )
    args = parser.parse_args()

    experiments = resolve_experiments(args.experiments)
    known = set(ALL_EXPERIMENTS)
    unknown = [x for x in experiments if x not in known]
    if unknown:
        raise SystemExit(f"Unknown experiments: {unknown}. Available: {sorted(known)}")

    UNIFIED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for path in UNIFIED_OUTPUT_DIR.rglob("*.csv"):
            path.unlink()

    if not args.skip_run:
        for name in ordered_for_dependencies(experiments):
            run_command(name)

    collect_unified_outputs(UNIFIED_OUTPUT_DIR)

    print()
    print("=" * 80)
    print("UNIFIED COMPARISON FILES")
    print("=" * 80)
    print(UNIFIED_OUTPUT_DIR / "comparison_summary.csv")
    print(UNIFIED_OUTPUT_DIR / "comparison_season_results.csv")


if __name__ == "__main__":
    main()
