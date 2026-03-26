"""
Dataset adapters - normalize raw tabular files to a canonical schema.
Does not modify or depend on loader.py schema validation.

Canonical columns (all optional except dataset_name):
  timestamp, asset_id, segment_id, sensor_id,
  pressure, flow_rate, temperature, vibration, rpm, power,
  status_label, leak_label, anomaly_label, target_localization,
  dataset_name
Original columns are preserved alongside canonical ones.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.dataset_registry import get_dataset, raw_data_paths

logger = logging.getLogger(__name__)

CANONICAL_COLS = [
    "timestamp",
    "asset_id",
    "segment_id",
    "sensor_id",
    "pressure",
    "flow_rate",
    "temperature",
    "vibration",
    "rpm",
    "power",
    "status_label",
    "leak_label",
    "anomaly_label",
    "target_localization",
    "dataset_name",
]

_KEY_TO_CANONICAL = {
    "timestamp": "timestamp",
    "asset_id": "asset_id",
    "segment_id": "segment_id",
    "sensor_id": "sensor_id",
    "pressure": "pressure",
    "flow_rate": "flow_rate",
    "temperature": "temperature",
    "vibration": "vibration",
    "rpm": "rpm",
    "power": "power",
    "status_label": "status_label",
    "leak_label": "leak_label",
    "anomaly_label": "anomaly_label",
    "target_localization": "target_localization",
}


def _read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, nrows=nrows)
    raise ValueError(f"Unsupported dataset file type: {path}")


def load_raw(name: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load all matching raw files for a dataset and concatenate them."""
    paths = raw_data_paths(name)
    if not paths:
        logger.warning("Dataset '%s' not downloaded - no matching raw files found.", name)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            logger.info("Loading %s from %s", name, path)
            frame = _read_table(path, nrows=nrows)
        except ValueError:
            logger.debug("Skipping unsupported file type for %s: %s", name, path)
            continue
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def _source_candidates(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def apply_column_map(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """
    Add canonical columns by aliasing original columns.
    Does not rename or drop originals.
    """
    for canonical_key, source_spec in column_map.items():
        candidates = _source_candidates(source_spec)
        if not candidates:
            continue
        canonical_name = _KEY_TO_CANONICAL.get(canonical_key, canonical_key)
        for candidate in candidates:
            if candidate in df.columns:
                if canonical_name not in df.columns:
                    df[canonical_name] = df[candidate]
                break
        else:
            logger.debug(
                "No mapped column found for key '%s'. Candidates: %s",
                canonical_key,
                candidates,
            )
    return df


def normalize(name: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw data and apply canonical column mapping.
    Adds dataset_name column.
    Returns empty DataFrame if data is not available.
    """
    meta = get_dataset(name)
    df = load_raw(name, nrows=nrows)
    if df.empty:
        return df

    df = apply_column_map(df, meta.get("column_map", {}))

    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            logger.debug("Could not parse timestamp column for %s", name)

    df["dataset_name"] = name
    logger.info("Normalized %s: %d rows, %d cols", name, len(df), len(df.columns))
    return df


def get_target_series(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Return the target column as a Series, or None if not present."""
    meta = get_dataset(name)
    target_col = meta.get("target")
    if target_col and target_col in df.columns:
        return df[target_col]
    if "target" in df.columns:
        return df["target"]
    if "leak_label" in df.columns:
        return df["leak_label"]
    return None


def save_processed(df: pd.DataFrame, name: str, suffix: str = "normalized") -> str:
    """Save processed DataFrame to data/processed/<name>/<suffix>.parquet."""
    meta = get_dataset(name)
    out_dir = Path(meta["local_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{suffix}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved %s -> %s", name, out_path)
    return str(out_path)
