#!/usr/bin/env python
"""Validate whether a SCADA-like synthetic dataset is useful for model training.

This script is deliberately skeptical. It checks whether the dataset:
- contains long enough histories to resemble deployment conditions
- represents slow leaks and nuisance disturbances
- includes instrumentation imperfections
- avoids making leaks trivially separable with simple rules

Outputs:
  reports/synthetic_validation/summary.json
  reports/synthetic_validation/summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.engineer import build_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ValidationSummary:
    rows: int
    segment_count: int
    history_count: int
    median_history_hours: float
    max_history_hours: float
    leak_row_rate: float
    slow_leak_event_count: int
    abrupt_leak_event_count: int
    borderline_leak_event_count: int
    disturbance_row_rate: float
    missing_value_rate: float
    stale_sensor_rate: float
    raw_sensor_auc: float
    dynamics_auc: float
    disturbance_false_positive_rate: float
    leak_disturbance_overlap: float
    warnings: list[str]


def infer_history_id(df: pd.DataFrame) -> pd.Series:
    for col in ("history_id", "scenario_id", "run_id"):
        if col in df.columns:
            return df[col].astype(str)

    if {"segment_id", "timestamp"}.issubset(df.columns):
        ordered = df.copy().reset_index()
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
        ts_reset = ordered.groupby("segment_id")["timestamp"].diff().dt.total_seconds().fillna(1).lt(0)
        group_ids = ts_reset.groupby(ordered["segment_id"]).cumsum().astype(str)
        return pd.Series(
            ordered["segment_id"].astype(str) + "|hist_" + group_ids,
            index=ordered["index"],
        ).reindex(df.index)

    sort_cols = ["segment_id", "timestamp"] if {"segment_id", "timestamp"}.issubset(df.columns) else df.columns.tolist()
    ordered = df.sort_values(sort_cols).copy()
    key_cols = [col for col in ("segment_id", "scenario_context") if col in ordered.columns]
    if not key_cols:
        return pd.Series(np.arange(len(df)), index=df.index, dtype=str)
    combined = ordered[key_cols].astype(str).agg("|".join, axis=1)
    groups = combined.ne(combined.shift()).cumsum().astype(str)
    return groups.reindex(df.index)


def history_duration_hours(df: pd.DataFrame, history_ids: pd.Series) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(dtype=float)
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["history_id"] = history_ids.values
    grouped = work.groupby("history_id")["timestamp"]
    hours = (grouped.max() - grouped.min()).dt.total_seconds().div(3600.0)
    return hours.fillna(0.0)


def classify_leak_events(df: pd.DataFrame) -> tuple[int, int, int]:
    if "target" not in df.columns:
        return 0, 0, 0

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work.get("timestamp"), errors="coerce")
    work = work.sort_values(
        [col for col in ("segment_id", "timestamp") if col in work.columns]
    ).reset_index(drop=True)
    work["active"] = work["target"].astype(int)
    change = work["active"].ne(work["active"].shift()).cumsum()
    leak_runs = work[work["active"] == 1].groupby(change)

    slow = abrupt = borderline = 0
    for _, event in leak_runs:
        max_sev = float(event.get("leak_severity", pd.Series([0.0])).fillna(0).max())
        duration = len(event)
        if max_sev < 0.18:
            borderline += 1
        elif duration >= 30 and max_sev < 0.45:
            slow += 1
        else:
            abrupt += 1
    return slow, abrupt, borderline


def disturbance_mask(df: pd.DataFrame) -> pd.Series:
    if "scenario_context" not in df.columns:
        return pd.Series(False, index=df.index)
    tags = [
        "demand",
        "pump_wear",
        "valve",
        "sensor",
        "maintenance",
        "startup",
        "shutdown",
        "disturb",
        "drift",
    ]
    pattern = "|".join(tags)
    return df["scenario_context"].astype(str).str.contains(pattern, case=False, regex=True)


def stale_sensor_rate(df: pd.DataFrame) -> float:
    sensor_cols = [c for c in ("pressure", "flow_rate", "temperature", "pump_speed") if c in df.columns]
    if not sensor_cols:
        return 0.0
    ordered = df.sort_values([c for c in ("segment_id", "timestamp") if c in df.columns]).copy()
    repeats = []
    for col in sensor_cols:
        same = ordered.groupby("segment_id")[col].diff().fillna(1e9).abs() < 1e-9
        repeats.append(float(same.mean()))
    return float(np.mean(repeats))


def _cross_validated_auc(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[float, np.ndarray]:
    group_labels = y.groupby(groups).max().reindex(groups.values).values
    unique_groups = np.unique(groups.values)
    if len(unique_groups) < 2 or y.nunique() < 2:
        return float("nan"), np.full(len(y), 0.5)

    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        return float("nan"), np.full(len(y), 0.5)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = np.full(len(y), np.nan)
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=int)

    for train_idx, test_idx in sgkf.split(X_arr, group_labels, groups.values):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_arr[train_idx])
        X_test = scaler.transform(X_arr[test_idx])
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_arr[train_idx])
        scores[test_idx] = model.predict_proba(X_test)[:, 1]

    mask = ~np.isnan(scores)
    if mask.sum() == 0 or len(np.unique(y_arr[mask])) < 2:
        return float("nan"), np.nan_to_num(scores, nan=0.5)
    return float(roc_auc_score(y_arr[mask], scores[mask])), np.nan_to_num(scores, nan=0.5)


def compute_validation_summary(df: pd.DataFrame) -> ValidationSummary:
    history_ids = infer_history_id(df)
    durations = history_duration_hours(df, history_ids)
    slow, abrupt, borderline = classify_leak_events(df)
    disturbance = disturbance_mask(df)

    missing_rate = float(df.isna().mean().mean()) if len(df) else 0.0
    stale_rate = stale_sensor_rate(df)

    featured = build_features(df)
    y = featured["target"].astype(int)
    groups = infer_history_id(featured)

    raw_cols = [c for c in ("pressure", "flow_rate", "temperature", "pump_speed", "energy_consumption") if c in featured.columns]
    dyn_cols = [
        c for c in (
            "pressure_delta",
            "flow_rate_delta",
            "pressure_pct_delta",
            "flow_pct_delta",
            "pressure_roll_z",
            "flow_roll_z",
            "pressure_delta_deviation",
            "flow_neg_streak",
            "pressure_baseline_dev",
            "flow_baseline_dev",
            "pressure_segment_deviation",
            "flow_segment_deviation",
        )
        if c in featured.columns
    ]

    raw_auc, raw_scores = _cross_validated_auc(featured[raw_cols], y, groups) if raw_cols else (float("nan"), np.full(len(y), 0.5))
    dynamics_auc, dyn_scores = _cross_validated_auc(featured[dyn_cols], y, groups) if dyn_cols else (float("nan"), np.full(len(y), 0.5))

    disturbance_negative = disturbance & (featured["target"].astype(int) == 0)
    disturbance_fpr = float((dyn_scores[disturbance_negative] >= 0.5).mean()) if disturbance_negative.any() else float("nan")

    leak_rows = featured["target"].astype(int) == 1
    leak_disturbance_overlap = float("nan")
    if "pressure_delta" in featured.columns and leak_rows.any() and disturbance_negative.any():
        leak_abs = np.abs(featured.loc[leak_rows, "pressure_delta"].to_numpy())
        dist_abs = np.abs(featured.loc[disturbance_negative, "pressure_delta"].to_numpy())
        leak_lo, leak_hi = np.quantile(leak_abs, [0.1, 0.9])
        dist_lo, dist_hi = np.quantile(dist_abs, [0.1, 0.9])
        overlap = max(0.0, min(leak_hi, dist_hi) - max(leak_lo, dist_lo))
        span = max(leak_hi, dist_hi) - min(leak_lo, dist_lo)
        leak_disturbance_overlap = float(overlap / span) if span > 0 else 0.0
    warnings: list[str] = []

    if len(durations) and durations.median() < 24:
        warnings.append("Median history is still under 24 hours; this remains short of deployment-style context.")
    if raw_auc == raw_auc and raw_auc > 0.93:
        warnings.append("Raw sensors alone separate leaks too easily; synthetic shortcuts may still exist.")
    if dynamics_auc == dynamics_auc and raw_auc == raw_auc and dynamics_auc - raw_auc < 0.02:
        warnings.append("Dynamic features add little over raw sensors; event context may be too clean.")
    if disturbance_fpr == disturbance_fpr and disturbance_fpr < 0.02:
        warnings.append("Non-leak disturbances may be too easy; false-positive traps look weak.")
    if slow == 0 or borderline == 0:
        warnings.append("Slow/borderline leak coverage is thin.")
    if missing_rate < 0.001 and stale_rate < 0.005:
        warnings.append("Instrumentation realism is light; missingness/staleness barely appears.")

    return ValidationSummary(
        rows=int(len(df)),
        segment_count=int(df["segment_id"].nunique()) if "segment_id" in df.columns else 0,
        history_count=int(history_ids.nunique()),
        median_history_hours=float(durations.median()) if len(durations) else 0.0,
        max_history_hours=float(durations.max()) if len(durations) else 0.0,
        leak_row_rate=float(df["target"].mean()) if "target" in df.columns and len(df) else 0.0,
        slow_leak_event_count=int(slow),
        abrupt_leak_event_count=int(abrupt),
        borderline_leak_event_count=int(borderline),
        disturbance_row_rate=float(disturbance.mean()) if len(df) else 0.0,
        missing_value_rate=missing_rate,
        stale_sensor_rate=stale_rate,
        raw_sensor_auc=raw_auc,
        dynamics_auc=dynamics_auc,
        disturbance_false_positive_rate=disturbance_fpr,
        leak_disturbance_overlap=leak_disturbance_overlap,
        warnings=warnings,
    )


def write_outputs(summary: ValidationSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    payload = asdict(summary)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Synthetic Data Validation",
        "",
        f"- Rows: {summary.rows}",
        f"- Segments: {summary.segment_count}",
        f"- Histories: {summary.history_count}",
        f"- Median history length: {summary.median_history_hours:.1f} hours",
        f"- Max history length: {summary.max_history_hours:.1f} hours",
        f"- Leak row rate: {summary.leak_row_rate:.2%}",
        f"- Slow leak events: {summary.slow_leak_event_count}",
        f"- Abrupt leak events: {summary.abrupt_leak_event_count}",
        f"- Borderline leak events: {summary.borderline_leak_event_count}",
        f"- Disturbance row rate: {summary.disturbance_row_rate:.2%}",
        f"- Missing value rate: {summary.missing_value_rate:.2%}",
        f"- Stale sensor rate: {summary.stale_sensor_rate:.2%}",
        f"- Raw-sensor AUC: {summary.raw_sensor_auc:.4f}" if summary.raw_sensor_auc == summary.raw_sensor_auc else "- Raw-sensor AUC: N/A",
        f"- Dynamics AUC: {summary.dynamics_auc:.4f}" if summary.dynamics_auc == summary.dynamics_auc else "- Dynamics AUC: N/A",
        f"- Disturbance false-positive rate: {summary.disturbance_false_positive_rate:.2%}" if summary.disturbance_false_positive_rate == summary.disturbance_false_positive_rate else "- Disturbance false-positive rate: N/A",
    ]
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic SCADA training data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample/realtime_training_data.csv",
        help="Synthetic dataset CSV to inspect",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/synthetic_validation",
        help="Directory for JSON/Markdown outputs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    summary = compute_validation_summary(df)
    write_outputs(summary, Path(args.output_dir))
    log.info("Validation summary written to %s", args.output_dir)
    if summary.warnings:
        for warning in summary.warnings:
            log.warning(warning)


if __name__ == "__main__":
    main()
