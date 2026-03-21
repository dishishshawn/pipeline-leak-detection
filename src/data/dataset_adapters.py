"""
Dataset adapters — normalize raw CSVs to canonical schema.
Does NOT modify or depend on the existing loader.py schema validation.

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

from src.data.dataset_registry import get_dataset, raw_csv_path

logger = logging.getLogger(__name__)

CANONICAL_COLS = [
    "timestamp", "asset_id", "segment_id", "sensor_id",
    "pressure", "flow_rate", "temperature", "vibration", "rpm", "power",
    "status_label", "leak_label", "anomaly_label", "target_localization",
    "dataset_name",
]

# Maps registry column_map keys → canonical names
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


def load_raw(name: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load raw CSV for a dataset. Returns empty DataFrame if not downloaded."""
    csv = raw_csv_path(name)
    if csv is None:
        logger.warning("Dataset '%s' not downloaded — raw CSV not found.", name)
        return pd.DataFrame()
    logger.info("Loading %s from %s", name, csv)
    return pd.read_csv(csv, nrows=nrows, low_memory=False)


def apply_column_map(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """
    Add canonical columns by aliasing original columns.
    Does not rename or drop originals.
    """
    for canonical_key, src_col in column_map.items():
        if src_col is None:
            continue
        canonical_name = _KEY_TO_CANONICAL.get(canonical_key, canonical_key)
        if src_col in df.columns and canonical_name not in df.columns:
            df[canonical_name] = df[src_col]
        elif src_col not in df.columns:
            logger.debug("Column '%s' not found in dataset (mapped from key '%s')", src_col, canonical_key)
    return df


def normalize(name: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw CSV and apply canonical column mapping.
    Adds dataset_name column.
    Returns empty DataFrame if data not available.
    """
    meta = get_dataset(name)
    df = load_raw(name, nrows=nrows)
    if df.empty:
        return df

    column_map = meta.get("column_map", {})
    df = apply_column_map(df, column_map)

    # Parse timestamp if available
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            logger.debug("Could not parse timestamp column for %s", name)

    df["dataset_name"] = name
    logger.info("Normalized %s: %d rows, %d cols", name, len(df), len(df.columns))
    return df


def get_target_series(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Return target column as a Series, or None if not present."""
    meta = get_dataset(name)
    target_col = meta.get("target")
    if target_col and target_col in df.columns:
        return df[target_col]
    # Fall back to canonical leak_label
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
    logger.info("Saved %s → %s", name, out_path)
    return str(out_path)
