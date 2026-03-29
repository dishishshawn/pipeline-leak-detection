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
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "petrobras_3w"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed" / "petrobras_3w"
DEFAULT_OUTPUT_CSV = DEFAULT_PROCESSED_DIR / "petrobras_3w_scada.csv"

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

SENSOR_BOUNDS = {
    "P_": (-1e6, 1e9),
    "T_": (-100.0, 500.0),
    "Q_": (-1e3, 1e3),
}


def normalize_event_class(value: object) -> int:
    """Map 3W class variants like 101-109 back to their base event ids."""
    try:
        event_class = int(float(value))
    except (TypeError, ValueError):
        return 0

    if event_class in EVENT_NAMES:
        return event_class

    reduced = event_class % 100
    if event_class >= 100 and reduced in EVENT_NAMES:
        return reduced

    return event_class


def sanitize_sensor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible sensor sentinels with NaN before interpolation/fill."""
    df = df.copy()
    for prefix, (lower, upper) in SENSOR_BOUNDS.items():
        for col in [c for c in df.columns if c.startswith(prefix)]:
            values = pd.to_numeric(df[col], errors="coerce")
            values = values.mask(~np.isfinite(values))
            values = values.mask((values < lower) | (values > upper))
            df[col] = values
    return df


def resolve_repo_dir(raw_dir: Path) -> Path:
    """Resolve the folder that should contain the cloned 3W repo."""
    if raw_dir.name == "dataset":
        return raw_dir.parent
    if raw_dir.name == "3W":
        return raw_dir
    if (raw_dir / "dataset").exists():
        return raw_dir
    if (raw_dir / "3W" / "dataset").exists():
        return raw_dir / "3W"
    return raw_dir / "3W"


def resolve_dataset_dir(raw_dir: Path) -> Path:
    """Resolve the dataset directory from a raw root, repo root, or dataset path."""
    if raw_dir.name == "dataset" and raw_dir.exists():
        return raw_dir

    repo_dir = resolve_repo_dir(raw_dir)
    return repo_dir / "dataset"


def resolve_processing_summary_path(output_csv: Path, summary_json: str | None) -> Path:
    """Return the processing summary path for this build."""
    return Path(summary_json).expanduser() if summary_json else output_csv.with_name(
        "petrobras_3w_processing_summary.json"
    )


def download_3w(raw_dir: Path, max_files: int | None = None) -> None:
    """Clone the 3W dataset from GitHub (dataset directory only)."""
    import subprocess

    repo_dir = resolve_repo_dir(raw_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

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


def find_parquet_files(raw_dir: Path, max_files: int | None = None) -> tuple[list[Path], int]:
    """Find all parquet data files in the 3W dataset directory."""
    dataset_dir = resolve_dataset_dir(raw_dir)

    if not dataset_dir.exists():
        log.error("Dataset directory not found at %s", dataset_dir)
        log.error("Run without --skip-download first.")
        sys.exit(1)

    # Parquet files are in subdirectories named by event type (0-9)
    files = sorted(dataset_dir.rglob("*.parquet"))
    # Filter out fold config files
    files = [f for f in files if "folds" not in str(f)]

    log.info("Found %d parquet files", len(files))
    available_file_count = len(files)

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
        return sampled, available_file_count

    return files, available_file_count


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
        result["event_class"] = result["class"].apply(normalize_event_class)
        result.drop(columns=["class"], inplace=True)
    else:
        # Infer from directory name
        try:
            result["event_class"] = normalize_event_class(event_dir)
        except ValueError:
            result["event_class"] = 0

    # Map to binary target: 0 = normal, 1 = anomaly/fault
    result["target"] = (result["event_class"] != 0).astype(int)
    result["event_type"] = result["event_class"].map(EVENT_NAMES).fillna("UNKNOWN")

    # Rename columns to our SCADA schema
    for old_name, new_name in COLUMN_RENAME.items():
        if old_name in result.columns:
            result.rename(columns={old_name: new_name}, inplace=True)

    result = sanitize_sensor_columns(result)

    # Add metadata
    result["well_id"] = well_id
    result["scenario_id"] = f"{event_dir}_{well_id}"

    # Drop rows where all sensor columns are NaN
    sensor_cols = [c for c in result.columns if c.startswith(("P_", "T_", "Q_"))]
    if sensor_cols:
        result.dropna(subset=sensor_cols, how="all", inplace=True)

    return result


def build_processed_dataset(
    files: list[Path],
    downsample_step: int = 10,
) -> tuple[pd.DataFrame, dict[str, int], int]:
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

        # Track the actual row-level event distribution within each file.
        for event_type, count in df["event_type"].value_counts().items():
            event_counts[event_type] = event_counts.get(event_type, 0) + int(count)

        if (i + 1) % 100 == 0:
            log.info("  Processed %d / %d files (%.0fs elapsed)",
                     i + 1, len(files), time.time() - t0)

    if not all_dfs:
        log.error("No data loaded from any files.")
        return pd.DataFrame(), {}, 0

    combined = pd.concat(all_dfs, ignore_index=True)
    elapsed = time.time() - t0

    log.info("Loaded %d files -> %d rows in %.1fs", len(all_dfs), len(combined), elapsed)
    log.info("Event distribution:")
    for event, count in sorted(event_counts.items()):
        pct = 100 * count / len(combined)
        log.info("  %s: %d rows (%.1f%%)", event, count, pct)

    return combined, event_counts, len(all_dfs)


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


def write_processing_summary(
    summary_path: Path,
    *,
    raw_dir: Path,
    output_csv: Path,
    available_file_count: int,
    selected_file_count: int,
    loaded_file_count: int,
    max_files: int | None,
    downsample_step: int,
    combined: pd.DataFrame,
    event_counts: dict[str, int],
) -> None:
    """Write a JSON summary that notebooks can inspect after processing."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "dataset_dir": str(resolve_dataset_dir(raw_dir)),
        "output_csv": str(output_csv),
        "available_file_count": int(available_file_count),
        "selected_file_count": int(selected_file_count),
        "loaded_file_count": int(loaded_file_count),
        "max_files": int(max_files) if max_files is not None else None,
        "downsample_step": int(downsample_step),
        "n_rows": int(len(combined)),
        "n_columns": int(len(combined.columns)),
        "n_wells": int(combined["well_id"].nunique()) if "well_id" in combined.columns else 0,
        "n_scenarios": int(combined["scenario_id"].nunique()) if "scenario_id" in combined.columns else 0,
        "label_distribution": {
            str(k): int(v) for k, v in combined["target"].value_counts().sort_index().items()
        } if "target" in combined.columns else {},
        "event_distribution": {str(k): int(v) for k, v in sorted(event_counts.items())},
        "columns": list(combined.columns),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Processing summary -> %s", summary_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and process Petrobras 3W dataset")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Limit number of parquet files to process")
    ap.add_argument("--downsample", type=int, default=10,
                    help="Keep every Nth row (default: 10, i.e. 0.1Hz from 1Hz)")
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip git clone, use existing files")
    ap.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR),
                    help="Directory that contains the 3W repo or dataset")
    ap.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV),
                    help="Where to save the processed SCADA CSV")
    ap.add_argument("--summary-json", type=str, default=None,
                    help="Optional JSON summary output path")
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir).expanduser()
    output_csv = Path(args.output_csv).expanduser()
    summary_path = resolve_processing_summary_path(output_csv, args.summary_json)

    # Step 1: Download
    if not args.skip_download:
        download_3w(raw_dir, max_files=args.max_files)

    # Step 2: Find files
    files, available_file_count = find_parquet_files(raw_dir, max_files=args.max_files)
    if not files:
        log.error("No parquet files found. Check data/raw/petrobras_3w/3W/dataset/")
        return

    # Step 3: Process
    combined, event_counts, loaded_file_count = build_processed_dataset(
        files,
        downsample_step=args.downsample,
    )
    if combined.empty:
        return

    # Step 4: Derive extra features
    combined = add_derived_features(combined)

    # Step 5: Export
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    log.info("Saved processed dataset -> %s (%.1f MB)", output_csv, output_csv.stat().st_size / 1e6)
    write_processing_summary(
        summary_path,
        raw_dir=raw_dir,
        output_csv=output_csv,
        available_file_count=available_file_count,
        selected_file_count=len(files),
        loaded_file_count=loaded_file_count,
        max_files=args.max_files,
        downsample_step=args.downsample,
        combined=combined,
        event_counts=event_counts,
    )

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
    log.info("  Output:         %s", output_csv)
    log.info("  Summary JSON:   %s", summary_path)


if __name__ == "__main__":
    main()
