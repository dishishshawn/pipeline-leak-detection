"""
Cheap EDA for Phase 1 datasets (scada_pipeline, water_leak).
Outputs:
  reports/dataset_checks/<name>_eda.md
  reports/dataset_checks/<name>_eda.json
  data/processed/<name>/normalized.parquet

Run:
  python scripts/run_eda.py
  python scripts/run_eda.py --datasets scada_pipeline water_leak
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_adapters import normalize, get_target_series, save_processed
from src.data.dataset_registry import is_downloaded

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "dataset_checks"
NUMERIC_CANONICAL = ["pressure", "flow_rate", "temperature", "vibration", "rpm", "power"]
PHASE1_DATASETS = ["scada_pipeline", "water_leak"]


def eda_summary(df: pd.DataFrame, name: str, target_col: str | None) -> dict:
    summary = {
        "dataset": name,
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": list(df.columns),
        "null_summary": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "numeric_stats": {},
        "label_distribution": None,
        "timestamp_usable": False,
        "labels_usable": False,
        "dataset_verdict": "unknown",
    }

    # Numeric stats for canonical sensor columns present in df
    for col in NUMERIC_CANONICAL:
        if col in df.columns:
            s = df[col].dropna()
            summary["numeric_stats"][col] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "nulls": int(df[col].isnull().sum()),
            }

    # Timestamp usability
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(ts):
            n_unique = ts.nunique()
            summary["timestamp_usable"] = n_unique > 1
            summary["timestamp_range"] = [str(ts.min()), str(ts.max())]
            summary["timestamp_unique"] = int(n_unique)
        else:
            summary["timestamp_usable"] = False

    # Label distribution
    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False)
        summary["label_distribution"] = vc.to_dict()
        n_classes = vc.shape[0]
        majority_pct = float(vc.iloc[0] / len(df))
        summary["n_classes"] = n_classes
        summary["majority_class_pct"] = round(majority_pct, 4)
        summary["labels_usable"] = n_classes >= 2 and majority_pct < 0.99
    elif "leak_label" in df.columns:
        vc = df["leak_label"].value_counts(dropna=False)
        summary["label_distribution"] = vc.to_dict()
        summary["labels_usable"] = vc.shape[0] >= 2

    # Overall verdict
    too_small = summary["row_count"] < 500
    no_labels = not summary["labels_usable"]
    if too_small:
        summary["dataset_verdict"] = "too_small"
    elif no_labels:
        summary["dataset_verdict"] = "no_usable_labels"
    elif summary["row_count"] >= 10000:
        summary["dataset_verdict"] = "usable_large"
    else:
        summary["dataset_verdict"] = "usable_small"

    return summary


def write_markdown(summary: dict, out_path: Path):
    lines = [
        f"# EDA: {summary['dataset']}",
        "",
        f"**Rows:** {summary['row_count']:,}  |  **Cols:** {summary['col_count']}  |  **Verdict:** `{summary['dataset_verdict']}`",
        "",
        "## Columns",
        ", ".join(f"`{c}`" for c in summary["columns"]),
        "",
        "## Null Summary",
    ]
    if summary["null_summary"]:
        for col, n in summary["null_summary"].items():
            lines.append(f"- `{col}`: {n:,} nulls")
    else:
        lines.append("No nulls detected.")

    lines += ["", "## Numeric Stats"]
    for col, stats in summary["numeric_stats"].items():
        lines.append(
            f"- **{col}**: mean={stats['mean']}, std={stats['std']}, "
            f"min={stats['min']}, max={stats['max']}, nulls={stats['nulls']}"
        )

    lines += ["", "## Labels"]
    if summary["label_distribution"]:
        lines.append(f"Classes: {summary.get('n_classes', '?')}  |  Majority: {summary.get('majority_class_pct', '?'):.1%}")
        for cls, cnt in summary["label_distribution"].items():
            lines.append(f"- `{cls}`: {cnt:,}")
        lines.append(f"\n**Labels usable:** {summary['labels_usable']}")
    else:
        lines.append("No label column detected.")

    if "timestamp_range" in summary:
        lines += [
            "", "## Timestamp",
            f"Range: {summary['timestamp_range'][0]} → {summary['timestamp_range'][1]}",
            f"Unique timestamps: {summary.get('timestamp_unique', '?')}",
            f"Usable: {summary['timestamp_usable']}",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("EDA markdown → %s", out_path)


def run_eda(name: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not is_downloaded(name):
        logger.warning("Skipping '%s' — not downloaded. See DOWNLOAD_DATA.md.", name)
        return None

    logger.info("=== EDA: %s ===", name)
    df = normalize(name)
    if df.empty:
        logger.warning("Empty DataFrame for %s", name)
        return None

    # Find target col from registry
    from src.data.dataset_registry import get_dataset
    meta = get_dataset(name)
    target_col = meta.get("target")

    summary = eda_summary(df, name, target_col)

    # Save outputs
    json_path = REPORTS_DIR / f"{name}_eda.json"
    md_path = REPORTS_DIR / f"{name}_eda.md"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("EDA JSON → %s", json_path)
    write_markdown(summary, md_path)

    save_processed(df, name, suffix="normalized")

    # Print quick verdict
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"  Rows: {summary['row_count']:,}  Cols: {summary['col_count']}")
    print(f"  Verdict: {summary['dataset_verdict']}")
    print(f"  Labels usable: {summary['labels_usable']}")
    print(f"  Timestamp usable: {summary['timestamp_usable']}")
    if summary["label_distribution"]:
        print(f"  Label dist: {summary['label_distribution']}")
    print(f"{'='*50}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run cheap EDA on Phase 1 datasets")
    parser.add_argument("--datasets", nargs="+", default=PHASE1_DATASETS)
    args = parser.parse_args()

    results = {}
    for name in args.datasets:
        result = run_eda(name)
        if result:
            results[name] = result

    if not results:
        print("\nNo datasets available. Run: python scripts/download_data.py")
        print("Or follow instructions in DOWNLOAD_DATA.md")
        return

    print("\n=== PHASE 1 EDA SUMMARY ===")
    for name, s in results.items():
        print(f"  {name}: {s['dataset_verdict']} | rows={s['row_count']:,} | labels={s['labels_usable']}")


if __name__ == "__main__":
    main()
