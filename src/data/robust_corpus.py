from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.dataset_adapters import get_target_series, normalize
from src.data.dataset_registry import get_dataset
from src.data.loader import REQUIRED_COLUMNS

DEFAULT_START_TIME = "2026-01-01 00:00:00"
_PETROBRAS_PROCESSED_CSV = "petrobras_3w_scada.csv"
_DEFAULT_SCADA_VALUES = {
    "valve_status": 1,
    "pump_state": 1,
    "pump_speed": 0.0,
    "compressor_state": 1,
    "energy_consumption": 0.0,
    "alarm_triggered": 0,
}


def _normalize_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0).astype(int)

    text = series.astype(str).str.strip().str.lower()
    positives = {
        "1",
        "true",
        "yes",
        "y",
        "leak",
        "burst",
        "fault",
        "warning",
        "alarm",
        "anomaly",
        "incident",
        "abnormal",
    }
    negatives = {"0", "false", "no", "n", "normal", "ok", "none", "nominal"}
    mapped = text.map(lambda value: 1 if value in positives else 0 if value in negatives else np.nan)
    return mapped.fillna(0).astype(int)


def _factorize_segments(series: pd.Series | None, length: int) -> pd.Series:
    if series is None:
        return pd.Series(np.ones(length, dtype=int))
    labels = series.fillna("segment-1").astype(str)
    codes, _ = pd.factorize(labels, sort=True)
    return pd.Series(codes + 1, index=series.index if hasattr(series, "index") else None)


def _coerce_temperature(series: pd.Series | None, length: int) -> pd.Series:
    if series is None:
        return pd.Series(np.full(length, 20.0))
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(float(numeric.dropna().median()))
    return pd.Series(np.full(length, 20.0), index=series.index)


def _ensure_scada_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Backfill standard SCADA columns for processed telemetry exports."""
    frame = frame.copy()

    if "segment_id" not in frame.columns:
        segment_source = None
        for candidate in ("segment_id", "scenario_id", "well_id", "source_file"):
            if candidate in frame.columns:
                segment_source = frame[candidate]
                break
        frame["segment_id"] = _factorize_segments(segment_source, len(frame))

    for column, default in _DEFAULT_SCADA_VALUES.items():
        if column not in frame.columns:
            frame[column] = default

    if "event_type" not in frame.columns:
        if "target" in frame.columns:
            target = _normalize_binary_target(frame["target"])
            frame["event_type"] = np.where(target.eq(1), "leak", "normal")
        else:
            frame["event_type"] = "normal"

    return frame


def rebalance_binary_frame(
    df: pd.DataFrame,
    *,
    max_negative_ratio: float = 3.0,
    random_state: int = 42,
) -> pd.DataFrame:
    if df.empty or "target" not in df.columns:
        return df

    positives = df[df["target"] == 1]
    negatives = df[df["target"] == 0]
    if positives.empty or negatives.empty:
        return df.reset_index(drop=True)

    max_negatives = int(len(positives) * max_negative_ratio)
    if len(negatives) <= max_negatives:
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    sampled_negatives = negatives.sample(n=max_negatives, random_state=random_state)
    return (
        pd.concat([positives, sampled_negatives], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def to_scada_training_frame(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty or "pressure" not in df.columns or "flow_rate" not in df.columns:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source_dataset"])

    target = get_target_series(df, dataset_name)
    if target is None:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source_dataset"])

    target = _normalize_binary_target(target)

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        timestamps = pd.to_datetime(
            pd.date_range(DEFAULT_START_TIME, periods=len(df), freq="1min")
        )

    segment_source = None
    for candidate in ("segment_id", "sensor_id", "asset_id", "source_file"):
        if candidate in df.columns:
            segment_source = df[candidate]
            break

    pressure = pd.to_numeric(df["pressure"], errors="coerce")
    flow_rate = pd.to_numeric(df["flow_rate"], errors="coerce")
    temperature = _coerce_temperature(df["temperature"] if "temperature" in df.columns else None, len(df))

    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "segment_id": _factorize_segments(segment_source, len(df)),
            "pressure": pressure,
            "flow_rate": flow_rate,
            "temperature": temperature,
            "valve_status": 1,
            "pump_state": 1,
            "pump_speed": 0.0,
            "compressor_state": 1,
            "energy_consumption": 0.0,
            "alarm_triggered": 0,
            "event_type": np.where(target.eq(1), "leak", "normal"),
            "target": target,
            "source_dataset": dataset_name,
        }
    )
    result = result.dropna(subset=["timestamp", "pressure", "flow_rate", "temperature", "target"])
    return result.reset_index(drop=True)


def _preferred_live_training_frame(name: str) -> pd.DataFrame:
    meta = get_dataset(name)
    processed_dir = Path(meta["local_processed"])

    if name == "petrobras_3w":
        processed_csv = processed_dir / _PETROBRAS_PROCESSED_CSV
        if processed_csv.exists():
            return load_scada_like_csv(processed_csv, name)

    return pd.DataFrame()


def load_dataset_for_live_training(name: str, nrows: int | None = None) -> pd.DataFrame:
    meta = get_dataset(name)
    if not meta.get("telemetry_compatible") or not meta.get("live_training_eligible"):
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source_dataset"])

    preferred_frame = _preferred_live_training_frame(name)
    if not preferred_frame.empty:
        return preferred_frame

    return to_scada_training_frame(normalize(name, nrows=nrows), name)


def load_scada_like_csv(path: str | Path, source_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = _ensure_scada_columns(frame)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required SCADA columns: {missing}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["source_dataset"] = source_name
    return frame.dropna(subset=["timestamp", "pressure", "flow_rate", "temperature", "target"]).reset_index(drop=True)


def build_robust_training_corpus(
    *,
    simulator_paths: Sequence[str | Path],
    dataset_names: Sequence[str],
    output_path: str | Path,
    max_rows_per_source: int | None = 25000,
    max_negative_ratio: float = 3.0,
    random_state: int = 42,
) -> dict:
    frames: list[pd.DataFrame] = []
    used_sources: dict[str, int] = {}
    skipped_sources: dict[str, str] = {}

    for simulator_path in simulator_paths:
        path = Path(simulator_path)
        if not path.exists():
            skipped_sources[str(path)] = "missing_file"
            continue
        simulator_frame = load_scada_like_csv(path, path.stem)
        simulator_frame = rebalance_binary_frame(
            simulator_frame,
            max_negative_ratio=max_negative_ratio,
            random_state=random_state,
        )
        if max_rows_per_source is not None and len(simulator_frame) > max_rows_per_source:
            simulator_frame = simulator_frame.sample(n=max_rows_per_source, random_state=random_state)
        frames.append(simulator_frame.reset_index(drop=True))
        used_sources[path.stem] = len(simulator_frame)

    for dataset_name in dataset_names:
        meta = get_dataset(dataset_name)
        if not meta.get("telemetry_compatible"):
            skipped_sources[dataset_name] = "not_telemetry_compatible"
            continue
        if not meta.get("live_training_eligible"):
            skipped_sources[dataset_name] = "not_live_training_eligible"
            continue

        frame = load_dataset_for_live_training(dataset_name)
        if frame.empty:
            skipped_sources[dataset_name] = "no_usable_rows"
            continue
        frame = rebalance_binary_frame(
            frame,
            max_negative_ratio=max_negative_ratio,
            random_state=random_state,
        )
        if max_rows_per_source is not None and len(frame) > max_rows_per_source:
            frame = frame.sample(n=max_rows_per_source, random_state=random_state)
        frames.append(frame.reset_index(drop=True))
        used_sources[dataset_name] = len(frame)

    if not frames:
        raise ValueError("No usable datasets were available for the robust training corpus.")

    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(output_path, index=False)

    summary = {
        "output_path": str(output_path),
        "row_count": int(len(corpus)),
        "positive_count": int(corpus["target"].sum()),
        "negative_count": int((corpus["target"] == 0).sum()),
        "source_rows": used_sources,
        "skipped_sources": skipped_sources,
        "columns": corpus.columns.tolist(),
    }

    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = [
    "build_robust_training_corpus",
    "load_dataset_for_live_training",
    "rebalance_binary_frame",
    "to_scada_training_frame",
]
