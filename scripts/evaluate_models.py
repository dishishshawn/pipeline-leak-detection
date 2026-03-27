#!/usr/bin/env python
"""Evaluate realtime leak-detection models with scenario-level metrics.

Generates isolated simulator runs per scenario, scores every live-eligible
model, calibrates alert thresholds, and ranks models by operational metrics.

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --runs-per-scenario 10 --target-fpr 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.harness import (
    EvaluationReport,
    run_evaluation,
    save_report,
    compute_metrics,
)
from src.evaluation.alert_policy import AlertPolicy, AlertPolicyConfig
from src.models.artifacts import discover_model_artifacts
from src.models.predict import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Only evaluate models that work with the realtime simulator's SCADA output
EVAL_ALLOWLIST = {
    "Realtime Random Forest",
    "Realtime Xgboost",
    "Realtime Lightgbm",
    "Realtime Hybrid Ensemble",
    "Realtime Isolation Forest",
    "Robust Random Forest",
    "Robust Xgboost",
    "Robust Lightgbm",
    "Robust Hybrid Ensemble",
    "Robust Isolation Forest",
}


def load_eval_models() -> dict[str, object]:
    """Load all evaluation-eligible models."""
    artifacts = discover_model_artifacts(str(MODEL_DIR), "scada")
    models = {}
    for label, artifact in artifacts.items():
        if label not in EVAL_ALLOWLIST:
            continue
        try:
            model = load_model(artifact.path)
            models[label] = model
            logger.info("Loaded: %s (%s)", label, artifact.path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", label, exc)
    return models


def print_report(report: EvaluationReport) -> None:
    """Print a formatted ranking table to stdout."""
    if not report.metrics:
        print("No models produced metrics.")
        return

    # Sort by: lowest FPR on steady_state, then fastest detection
    ranked = sorted(
        report.metrics,
        key=lambda m: (
            m.fpr_steady_state if not _isnan(m.fpr_steady_state) else 999,
            m.detection_delay_slow_seep if not _isnan(m.detection_delay_slow_seep) else 999,
        ),
    )

    header = (
        f"{'Model':<30} {'Threshold':>9} {'FPR-SS':>8} {'FPR-DS':>8} "
        f"{'Delay':>7} {'Det%':>6} {'Sens':>6} {'Stab':>6}"
    )
    print("\n" + "=" * len(header))
    print("MODEL EVALUATION REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for m in ranked:
        print(
            f"{m.model_name:<30} {m.calibrated_threshold:>9.4f} "
            f"{_fmt(m.fpr_steady_state):>8} {_fmt(m.fpr_demand_shock):>8} "
            f"{_fmt(m.detection_delay_slow_seep):>7} "
            f"{_fmt(m.detection_rate_slow_seep, pct=True):>6} "
            f"{_fmt(m.sensitivity_micro_leak, pct=True):>6} "
            f"{_fmt(m.alert_stability):>6}"
        )

    print("-" * len(header))
    print("\nFPR-SS  = False positive rate on steady_state (lower is better)")
    print("FPR-DS  = False positive rate on demand_shock (lower is better)")
    print("Delay   = Steps from leak onset to detection (lower is better)")
    print("Det%    = Detection rate on slow_seep runs (higher is better)")
    print("Sens    = Sensitivity on micro-leak rows (higher is better)")
    print("Stab    = Alert toggle rate (lower is better)")
    print(f"\nThresholds calibrated at target FPR = {report.config.get('target_fpr', '?')}")


def print_alert_policy_comparison(
    scores_df,
    report: EvaluationReport,
) -> None:
    """Compare raw threshold vs alert policy on toggle count."""
    if scores_df.empty or not report.metrics:
        return

    print("\n\nALERT POLICY COMPARISON (persistence=3, cooldown=10)")
    print("-" * 65)
    header = f"{'Model':<30} {'Raw Toggles':>12} {'Policy Toggles':>15}"
    print(header)
    print("-" * 65)

    for m in report.metrics:
        threshold = m.calibrated_threshold
        model_scores = scores_df[scores_df["model_name"] == m.model_name]

        # Raw toggle rate
        raw_stability = m.alert_stability

        # Policy-filtered toggle rate
        policy = AlertPolicy(AlertPolicyConfig(
            threshold=threshold,
            persistence_ticks=3,
            cooldown_ticks=10,
        ))
        runs_with_leaks = model_scores.groupby("run_id")["target"].max()
        leak_run_ids = runs_with_leaks[runs_with_leaks == 1].index
        policy_toggles: list[float] = []
        for run_id in leak_run_ids:
            run_df = model_scores[model_scores["run_id"] == run_id].sort_values("step_index")
            alerts = policy.process_series(run_df["leak_score"].values)
            toggles = float(abs(alerts.astype(int)[1:] - alerts.astype(int)[:-1]).sum())
            if len(run_df) > 0:
                policy_toggles.append(toggles / len(run_df))
        policy_avg = float(sum(policy_toggles) / len(policy_toggles)) if policy_toggles else float("nan")

        print(f"{m.model_name:<30} {_fmt(raw_stability):>12} {_fmt(policy_avg):>15}")


def _isnan(v: float) -> bool:
    try:
        return v != v  # NaN != NaN
    except Exception:
        return True


def _fmt(v: float, pct: bool = False) -> str:
    if _isnan(v):
        return "N/A"
    if pct:
        return f"{v:.0%}"
    return f"{v:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate leak detection models")
    parser.add_argument("--runs-per-scenario", type=int, default=5)
    parser.add_argument("--steps-per-run", type=int, default=120)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--base-seed", type=int, default=9999)
    parser.add_argument("--output-dir", type=str, default=str(MODEL_DIR.parent / "reports"))
    args = parser.parse_args()

    # Load models
    models = load_eval_models()
    if not models:
        logger.error("No evaluation-eligible models found in %s", MODEL_DIR)
        sys.exit(1)
    logger.info("Loaded %d models for evaluation", len(models))

    # Run evaluation
    report, scores_df = run_evaluation(
        models=models,
        runs_per_scenario=args.runs_per_scenario,
        steps_per_run=args.steps_per_run,
        target_fpr=args.target_fpr,
        base_seed=args.base_seed,
    )

    # Print results
    print_report(report)
    print_alert_policy_comparison(scores_df, report)

    # Save
    save_report(report, args.output_dir)
    details_path = Path(args.output_dir) / "evaluation_details.csv"
    if not scores_df.empty:
        scores_df.to_csv(details_path, index=False)
        logger.info("Saved detailed scores to %s", details_path)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
