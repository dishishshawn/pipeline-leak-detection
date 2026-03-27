"""Scenario-level model evaluation harness.

Generates isolated simulator runs per scenario preset, scores every model,
and computes five operational metrics:

1. FPR on steady_state
2. FPR on demand_shock
3. Time-to-detection on slow_seep
4. Sensitivity on micro_leak (low-severity leak rows)
5. Alert stability (toggle count in detection windows)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.features.engineer import build_features
from src.models.predict import predict_leak_score, supports_leak_score
from src.simulation.core import (
    PipelineTelemetrySimulator,
    SimulationConfig,
    make_default_profiles,
)
from src.simulation.scenarios import build_scenarios, get_scenario_presets

logger = logging.getLogger(__name__)

WARMUP_ROWS = 5
DEFAULT_SCENARIOS = ["steady_state", "slow_seep", "demand_shock", "compound_incident", "micro_leak"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_evaluation_runs(
    scenario_keys: Sequence[str] | None = None,
    runs_per_scenario: int = 5,
    steps_per_run: int = 120,
    segment_count: int = 3,
    base_seed: int = 9999,
) -> pd.DataFrame:
    """Generate isolated simulator runs tagged with run_id and scenario_key."""
    if scenario_keys is None:
        available = {p.key for p in get_scenario_presets()}
        scenario_keys = [k for k in DEFAULT_SCENARIOS if k in available]

    all_frames: list[pd.DataFrame] = []
    profiles = make_default_profiles(segment_count)
    segment_ids = [p.segment_id for p in profiles]

    for scenario_key in scenario_keys:
        for run_idx in range(runs_per_scenario):
            seed = base_seed + hash(scenario_key) % 10_000 + run_idx
            config = SimulationConfig(
                segment_profiles=profiles,
                seed=seed,
            )
            scenarios = build_scenarios(scenario_key, segment_ids)
            sim = PipelineTelemetrySimulator(config, scenarios=scenarios)
            sim.start()
            sim.advance_steps(steps_per_run)
            df = sim.snapshot()
            if df.empty:
                continue
            run_id = f"{scenario_key}_run_{run_idx}"
            df["run_id"] = run_id
            df["scenario_key"] = scenario_key
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-run feature engineering
# ---------------------------------------------------------------------------

def engineer_features_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Apply build_features per run_id to prevent cross-run rolling stat leakage."""
    parts = []
    for run_id, group in df.groupby("run_id", sort=False):
        featured = build_features(group)
        # Drop warmup rows where rolling stats are unreliable
        n_segments = featured["segment_id"].nunique()
        warmup_count = WARMUP_ROWS * n_segments
        if len(featured) > warmup_count:
            featured = featured.iloc[warmup_count:]
        parts.append(featured)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_all_models(
    df: pd.DataFrame,
    models: dict[str, object],
) -> pd.DataFrame:
    """Score every model on the evaluation data. Returns long-form results."""
    # Identify feature columns (drop metadata & labels)
    meta_cols = {
        "timestamp", "run_id", "scenario_key", "event_type", "target",
        "scenario_context", "leak_severity", "pump_efficiency",
        "alarm_triggered",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]

    results: list[pd.DataFrame] = []
    for model_name, model in models.items():
        if not supports_leak_score(model):
            logger.warning("Skipping %s: no predict support", model_name)
            continue
        try:
            X = df[feature_cols]
            scores = predict_leak_score(model, X)
        except Exception as exc:
            logger.warning("Scoring failed for %s: %s", model_name, exc)
            continue

        result = df[["run_id", "scenario_key", "segment_id", "target", "leak_severity"]].copy()
        result["model_name"] = model_name
        result["leak_score"] = scores
        # Assign a step index within each run
        result["step_index"] = result.groupby("run_id").cumcount()
        results.append(result)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    model_name: str
    fpr_steady_state: float = float("nan")
    fpr_demand_shock: float = float("nan")
    detection_delay_slow_seep: float = float("nan")
    detection_rate_slow_seep: float = float("nan")
    sensitivity_micro_leak: float = float("nan")
    alert_stability: float = float("nan")
    calibrated_threshold: float = float("nan")
    threshold_used: float = 0.5


def _fpr_on_scenario(scores_df: pd.DataFrame, scenario_key: str, threshold: float) -> float:
    """False positive rate: fraction of non-leak rows scoring above threshold."""
    subset = scores_df[scores_df["scenario_key"] == scenario_key]
    if subset.empty:
        return float("nan")
    negatives = subset[subset["target"] == 0]
    if negatives.empty:
        return float("nan")
    return float((negatives["leak_score"] >= threshold).mean())


def _detection_delay(scores_df: pd.DataFrame, scenario_key: str, threshold: float) -> tuple[float, float]:
    """Average time-to-detection (steps) and detection rate across runs.

    For each run, find the first step where target=1 (onset), then the first
    step at or after onset where leak_score >= threshold (detection).
    """
    subset = scores_df[scores_df["scenario_key"] == scenario_key]
    if subset.empty:
        return float("nan"), float("nan")

    delays: list[float] = []
    total_runs = 0
    detected_runs = 0

    for run_id, run_df in subset.groupby("run_id"):
        leak_rows = run_df[run_df["target"] == 1]
        if leak_rows.empty:
            continue
        total_runs += 1
        onset_step = leak_rows["step_index"].min()
        post_onset = run_df[run_df["step_index"] >= onset_step]
        detections = post_onset[post_onset["leak_score"] >= threshold]
        if detections.empty:
            continue
        detected_runs += 1
        detection_step = detections["step_index"].min()
        delays.append(detection_step - onset_step)

    if total_runs == 0:
        return float("nan"), float("nan")
    detection_rate = detected_runs / total_runs
    avg_delay = float(np.mean(delays)) if delays else float("nan")
    return avg_delay, detection_rate


def _sensitivity_micro_leak(scores_df: pd.DataFrame, threshold: float) -> float:
    """Recall on low-severity leak rows (severity < 0.3) across all leak scenarios."""
    leak_rows = scores_df[
        (scores_df["target"] == 1) & (scores_df["leak_severity"] < 0.3)
    ]
    if leak_rows.empty:
        # Fall back: try micro_leak scenario specifically
        leak_rows = scores_df[
            (scores_df["scenario_key"] == "micro_leak") & (scores_df["target"] == 1)
        ]
    if leak_rows.empty:
        return float("nan")
    return float((leak_rows["leak_score"] >= threshold).mean())


def _alert_stability(scores_df: pd.DataFrame, threshold: float) -> float:
    """Average toggle rate (score crossing threshold) per detection window.

    Lower is better — measures alert chatter.
    """
    # Filter to runs that contain true leaks
    runs_with_leaks = scores_df.groupby("run_id")["target"].max()
    leak_run_ids = runs_with_leaks[runs_with_leaks == 1].index

    toggle_rates: list[float] = []
    for run_id in leak_run_ids:
        run_df = scores_df[scores_df["run_id"] == run_id].sort_values("step_index")
        alerts = (run_df["leak_score"] >= threshold).astype(int)
        toggles = (alerts.diff().abs().fillna(0)).sum()
        window_len = len(run_df)
        if window_len > 0:
            toggle_rates.append(toggles / window_len)

    if not toggle_rates:
        return float("nan")
    return float(np.mean(toggle_rates))


def compute_metrics(
    scores_df: pd.DataFrame,
    model_name: str,
    threshold: float = 0.5,
) -> ModelMetrics:
    """Compute all five metrics for one model at the given threshold."""
    model_scores = scores_df[scores_df["model_name"] == model_name]

    return ModelMetrics(
        model_name=model_name,
        fpr_steady_state=_fpr_on_scenario(model_scores, "steady_state", threshold),
        fpr_demand_shock=_fpr_on_scenario(model_scores, "demand_shock", threshold),
        detection_delay_slow_seep=_detection_delay(model_scores, "slow_seep", threshold)[0],
        detection_rate_slow_seep=_detection_delay(model_scores, "slow_seep", threshold)[1],
        sensitivity_micro_leak=_sensitivity_micro_leak(model_scores, threshold),
        alert_stability=_alert_stability(model_scores, threshold),
        threshold_used=threshold,
    )


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(
    scores_df: pd.DataFrame,
    model_name: str,
    target_fpr: float = 0.05,
) -> float:
    """Calibrate an alert threshold using both non-leak noise ceiling and leak onset signal.

    Hybrid approach:
    1. Compute the (1 - target_fpr) percentile of non-leak scores (noise ceiling)
    2. Compute the median score at leak onset (first rows where target=1 in slow_seep)
    3. Set threshold = midpoint between noise ceiling and leak onset signal
    4. Enforce floor of 0.15 and cap of 0.85
    """
    model_scores = scores_df[scores_df["model_name"] == model_name]

    # Step 1: noise ceiling from non-leak scenarios
    non_leak = model_scores[
        model_scores["scenario_key"].isin(["steady_state", "demand_shock"])
    ]
    if non_leak.empty:
        noise_ceiling = 0.0
    else:
        noise_ceiling = float(np.percentile(non_leak["leak_score"].values, (1.0 - target_fpr) * 100))

    # Step 2: leak onset signal — scores at the first few rows where target=1
    leak_scenarios = model_scores[model_scores["scenario_key"].isin(["slow_seep", "micro_leak"])]
    onset_scores: list[float] = []
    for run_id, run_df in leak_scenarios.groupby("run_id"):
        leak_rows = run_df[run_df["target"] == 1].sort_values("step_index")
        if leak_rows.empty:
            continue
        # Take scores from the first 3 leak rows (onset region)
        onset_region = leak_rows.head(3)
        onset_scores.extend(onset_region["leak_score"].tolist())

    if onset_scores:
        leak_onset_signal = float(np.median(onset_scores))
    else:
        leak_onset_signal = 0.5

    # Step 3: threshold = midpoint between noise ceiling and leak onset
    threshold = (noise_ceiling + leak_onset_signal) / 2.0

    # Step 4: enforce meaningful bounds
    return round(max(0.15, min(0.85, threshold)), 4)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    metrics: list[ModelMetrics] = field(default_factory=list)
    calibrated_thresholds: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def run_evaluation(
    models: dict[str, object],
    scenario_keys: Sequence[str] | None = None,
    runs_per_scenario: int = 5,
    steps_per_run: int = 120,
    target_fpr: float = 0.05,
    base_seed: int = 9999,
) -> tuple[EvaluationReport, pd.DataFrame]:
    """End-to-end evaluation: generate data, score, calibrate, compute metrics.

    Returns (report, detailed_scores_df).
    """
    logger.info("Generating evaluation runs...")
    raw_df = generate_evaluation_runs(
        scenario_keys=scenario_keys,
        runs_per_scenario=runs_per_scenario,
        steps_per_run=steps_per_run,
        base_seed=base_seed,
    )
    logger.info("Generated %d rows across %d runs", len(raw_df), raw_df["run_id"].nunique())

    logger.info("Engineering features per run...")
    featured_df = engineer_features_per_run(raw_df)
    logger.info("Feature engineering done: %d rows", len(featured_df))

    logger.info("Scoring all models...")
    scores_df = score_all_models(featured_df, models)
    if scores_df.empty:
        logger.warning("No models produced scores.")
        return EvaluationReport(), pd.DataFrame()

    # Calibrate thresholds
    calibrated: dict[str, float] = {}
    for model_name in scores_df["model_name"].unique():
        calibrated[model_name] = calibrate_threshold(scores_df, model_name, target_fpr)
    logger.info("Calibrated thresholds: %s", calibrated)

    # Compute metrics at both default (0.5) and calibrated thresholds
    all_metrics: list[ModelMetrics] = []
    for model_name in scores_df["model_name"].unique():
        # Metrics at calibrated threshold
        m = compute_metrics(scores_df, model_name, calibrated[model_name])
        m.calibrated_threshold = calibrated[model_name]
        all_metrics.append(m)

    report = EvaluationReport(
        metrics=all_metrics,
        calibrated_thresholds=calibrated,
        config={
            "runs_per_scenario": runs_per_scenario,
            "steps_per_run": steps_per_run,
            "target_fpr": target_fpr,
            "base_seed": base_seed,
            "scenarios": list(scores_df["scenario_key"].unique()),
        },
    )
    return report, scores_df


def save_report(report: EvaluationReport, output_dir: str | Path) -> None:
    """Save evaluation results to JSON and summary text."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON with all metrics + thresholds
    data = {
        "config": report.config,
        "calibrated_thresholds": report.calibrated_thresholds,
        "models": [asdict(m) for m in report.metrics],
    }
    json_path = out / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved results to %s", json_path)
