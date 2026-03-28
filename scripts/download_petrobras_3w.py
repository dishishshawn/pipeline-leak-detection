#!/usr/bin/env python
"""Download and prepare the Petrobras 3W dataset for leak detection training.

The 3W dataset contains ~1,984 parquet files (~1.8 GB) of real oil well
telemetry from Petrobras. This script clones the dataset directory only
(sparse checkout) and converts it to a single training-ready CSV.

Usage:
    python scripts/download_petrobras_3w.py
    python scripts/download_petrobras_3w.py --max-files 100
    python scripts/download_petrobras_3w.py --skip-download   # Just rebuild CSV from existing files

Output:
    data/raw/petrobras_3w/          Raw parquet files (gitignored)
    data/processed/petrobras_3w/    Processed training CSV

Event types in 3W:
    0: NORMAL
    1: ABRUPT_INCREASE_OF_BSW
    2: SPURIOUS_CLOSURE_OF_DHSV
    3: SEVERE_SLUGGING
    4: FLOW_INSTABILITY
    5: RAPID_PRODUCTIVITY_LOSS
    6: QUICK_RESTRICTION_IN_PCK
    7: SCALING_IN_PCK
    8: HYDRATE_IN_PRODUCTION_LINE
    9: HYDRATE_IN_SERVICE_LINE

For leak detection, events 1-9 are mapped to target=1 (anomaly/fault),
event 0 maps to target=0 (normal).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "petrobras_3w"
PROCESSED_DIR = ROOT / "data" / "processed" / "petrobras_3w"

# 3W dataset columns we care about (pressure, temperature, flow)
KEEP_COLUMNS = [
    "P-PDG",       # Downhole pressure gauge (Pa)
    "P-TPT",       # Xmas-tree pressure transmitter (Pa)
    "P-MON-CKP",   # Upstream choke pressure (Pa)
    "P-JUS-CKP",   # Downstream choke pressure (Pa)
    "T-PDG",       # Downhole temperature (C)
    "T-TPT",       # Xmas-tree temperature (C)
    "QGL",         # Gas lift flow rate (m3/s)
    "class",       # Event label (0-9)
]

# Map 3W columns to our SCADA schema
COLUMN_RENAME = {
    "P-PDG": "P_inlet",       # Highest pressure point (downhole)
    "P-TPT": "P_mid",         # Mid-point pressure (Xmas tree)
    "P-MON-CKP": "P_outlet",  # Upstream choke = near outlet
    "T-PDG": "T_inlet",       # Downhole temp
    "T-TPT": "T_mid",         # Xmas-tree temp
    "QGL": "Q_inlet",         # Gas lift flow
}

EVENT_NAMES = {
    0: "NORMAL",
    1: "ABRUPT_INCREASE_OF_BSW",
    2: "SPURIOUS_CLOSURE_OF_DHSV",
    3: "SEVERE_SLUGGING",
    4: "FLOW_INSTABILITY",
    5: "RAPID_PRODUCTIVITY_LOSS",
    6: "QUICK_RESTRICTION_IN_PCK",
    7: "SCALING_IN_PCK",
    8: "HYDRATE_IN_PRODUCTION_LINE",
    9: "HYDRATE_IN_SERVICE_LINE",
}


def download_3w(max_files: int | None = None) -> None:
    """Clone the 3W dataset from GitHub (dataset directory only)."""
    import subprocess

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    repo_dir = RAW_DIR / "3W"
    if repo_dir.exists() and any(repo_dir.rglob("*.parquet")):
        n_existing = len(list(repo_dir.rglob("*.parquet")))
        log.info("3W repo already exists with %d parquet files, skipping clone.", n_existing)
        return

    log.info("Cloning Petrobras 3W dataset (this may take a few minutes)...")
    # Use git clone with depth=1 to minimize download
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/petrobras/3W.git", str(repo_dir)],
        check=True,
    )
    log.info("Clone complete.")


def find_parquet_files(max_files: int | None = None) -> list[Path]:
    """Find all parquet data files in the 3W dataset directory."""
    repo_dir = RAW_DIR / "3W"
    dataset_dir = repo_dir / "dataset"

    if not dataset_dir.exists():
        log.error("Dataset directory not found at %s", dataset_dir)
        log.error("Run without --skip-download first.")
        sys.exit(1)

    # Parquet files are in subdirectories named by event type (0-9)
    files = sorted(dataset_dir.rglob("*.parquet"))
    # Filter out fold config files
    files = [f for f in files if "folds" not in str(f)]

    log.info("Found %d parquet files", len(files))

    if max_files and len(files) > max_files:
        # Sample proportionally from each event type directory
        by_event: dict[str, list[Path]] = {}
        for f in files:
            event_dir = f.parent.name
            by_event.setdefault(event_dir, []).append(f)

        sampled: list[Path] = []
        per_event = max(1, max_files // len(by_event))
        for event_dir, event_files in sorted(by_event.items()):
            sampled.extend(event_files[:per_event])

        log.info("Sampled %d files (max %d per event type)", len(sampled), per_event)
        return sampled

    return files


def load_and_process_file(path: Path) -> pd.DataFrame | None:
    """Load a single 3W parquet file and convert to our schema."""
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        log.warning("Failed to read %s: %s", path.name, exc)
        return None

    if df.empty:
        return None

    # Extract event type from parent directory name
    event_dir = path.parent.name
    well_id = path.stem

    # Keep only columns we need (some files may not have all)
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    if not available:
        log.warning("No usable columns in %s", path.name)
        return None

    result = df[available].copy()

    # Handle timestamp index
    if isinstance(df.index, pd.DatetimeIndex):
        result["timestamp"] = df.index
    elif "timestamp" in df.columns:
        result["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # Use integer index as seconds
        result["timestamp"] = pd.to_datetime(range(len(df)), unit="s", origin="2020-01-01")

    # Extract class/label
    if "class" in result.columns:
        result["event_class"] = result["class"].fillna(0).astype(int)
        result.drop(columns=["class"], inplace=True)
    else:
        # Infer from directory name
        try:
            result["event_class"] = int(event_dir)
        except ValueError:
            result["event_class"] = 0

    # Map to binary target: 0 = normal, 1 = anomaly/fault
    result["target"] = (result["event_class"] != 0).astype(int)
    result["event_type"] = result["event_class"].map(EVENT_NAMES).fillna("UNKNOWN")

    # Rename columns to our SCADA schema
    for old_name, new_name in COLUMN_RENAME.items():
        if old_name in result.columns:
            result.rename(columns={old_name: new_name}, inplace=True)

    # Add metadata
    result["well_id"] = well_id
    result["scenario_id"] = f"{event_dir}_{well_id}"

    # Drop rows where all sensor columns are NaN
    sensor_cols = [c for c in result.columns if c.startswith(("P_", "T_", "Q_"))]
    if sensor_cols:
        result.dropna(subset=sensor_cols, how="all", inplace=True)

    return result


def build_processed_dataset(files: list[Path], downsample_step: int = 10) -> pd.DataFrame:
    """Load all files, process, and combine into one training dataset."""
    all_dfs: list[pd.DataFrame] = []
    event_counts: dict[str, int] = {}

    t0 = time.time()
    for i, f in enumerate(files):
        df = load_and_process_file(f)
        if df is None or df.empty:
            continue

        # Downsample (3W has 1Hz data, we want ~0.1Hz for manageable size)
        if downsample_step > 1:
            df = df.iloc[::downsample_step].reset_index(drop=True)

        all_dfs.append(df)

        # Track event distribution
        event_type = df["event_type"].iloc[0] if not df.empty else "UNKNOWN"
        event_counts[event_type] = event_counts.get(event_type, 0) + len(df)

        if (i + 1) % 100 == 0:
            log.info("  Processed %d / %d files (%.0fs elapsed)",
                     i + 1, len(files), time.time() - t0)

    if not all_dfs:
        log.error("No data loaded from any files.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    elapsed = time.time() - t0

    log.info("Loaded %d files -> %d rows in %.1fs", len(all_dfs), len(combined), elapsed)
    log.info("Event distribution:")
    for event, count in sorted(event_counts.items()):
        pct = 100 * count / len(combined)
        log.info("  %s: %d rows (%.1f%%)", event, count, pct)

    return combined


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SCADA-compatible derived columns for downstream compatibility."""
    # Fill missing sensor columns with reasonable defaults
    if "P_outlet" not in df.columns and "P_mid" in df.columns:
        # Estimate outlet from mid with small drop
        df["P_outlet"] = df["P_mid"] * 0.85

    if "Q_outlet" not in df.columns and "Q_inlet" in df.columns:
        # Approximate outlet flow (slightly less than inlet for normal, much less for fault)
        df["Q_outlet"] = df["Q_inlet"] * np.where(df["target"] == 1, 0.85, 0.98)

    if "T_outlet" not in df.columns:
        if "T_mid" in df.columns:
            df["T_outlet"] = df["T_mid"] - 2.0  # Slight cooling
        elif "T_inlet" in df.columns:
            df["T_outlet"] = df["T_inlet"] - 5.0

    # Standard SCADA columns expected by the dashboard
    if "pressure" not in df.columns and "P_inlet" in df.columns:
        df["pressure"] = df["P_inlet"]
    if "flow_rate" not in df.columns and "Q_inlet" in df.columns:
        df["flow_rate"] = df["Q_inlet"]
    if "temperature" not in df.columns and "T_inlet" in df.columns:
        df["temperature"] = df["T_inlet"]

    # Fill remaining NaNs with forward/backward fill per well
    sensor_cols = [c for c in df.columns if c.startswith(("P_", "T_", "Q_", "pressure", "flow_rate", "temperature"))]
    for col in sensor_cols:
        df[col] = df.groupby("well_id")[col].transform(lambda s: s.ffill().bfill())

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and process Petrobras 3W dataset")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Limit number of parquet files to process")
    ap.add_argument("--downsample", type=int, default=10,
                    help="Keep every Nth row (default: 10, i.e. 0.1Hz from 1Hz)")
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip git clone, use existing files")
    args = ap.parse_args()

    # Step 1: Download
    if not args.skip_download:
        download_3w(max_files=args.max_files)

    # Step 2: Find files
    files = find_parquet_files(max_files=args.max_files)
    if not files:
        log.error("No parquet files found. Check data/raw/petrobras_3w/3W/dataset/")
        return

    # Step 3: Process
    combined = build_processed_dataset(files, downsample_step=args.downsample)
    if combined.empty:
        return

    # Step 4: Derive extra features
    combined = add_derived_features(combined)

    # Step 5: Export
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "petrobras_3w_scada.csv"
    combined.to_csv(out_path, index=False)
    log.info("Saved processed dataset -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Summary
    log.info("=== DATASET SUMMARY ===")
    log.info("  Total rows:     %d", len(combined))
    log.info("  Normal rows:    %d (%.1f%%)",
             (combined["target"] == 0).sum(),
             100 * (combined["target"] == 0).mean())
    log.info("  Anomaly rows:   %d (%.1f%%)",
             (combined["target"] == 1).sum(),
             100 * (combined["target"] == 1).mean())
    log.info("  Wells:          %d", combined["well_id"].nunique())
    log.info("  Columns:        %s", list(combined.columns))
    log.info("  Output:         %s", out_path)


if __name__ == "__main__":
    main()
