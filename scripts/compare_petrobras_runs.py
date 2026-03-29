#!/usr/bin/env python
"""Compare one or more Petrobras training summaries and export a flat CSV."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def collect_summary_paths(summary_glob: str | None, summaries: list[str]) -> list[Path]:
    """Collect summary files from explicit paths and an optional glob."""
    paths = [Path(path).expanduser().resolve() for path in summaries]
    if summary_glob:
        paths.extend(Path(path).expanduser().resolve() for path in sorted(glob.glob(summary_glob)))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def summary_to_rows(summary: dict, source_path: Path) -> list[dict[str, object]]:
    """Flatten a training summary to one row per model."""
    models = summary.get("models", {})
    if not models:
        return []

    best_model = summary.get("best_model", {}).get("name")
    if best_model is None:
        best_model = max(models.items(), key=lambda item: item[1].get("roc_auc", float("-inf")))[0]

    run_label = summary.get("run_label")
    if run_label is None:
        run_label = source_path.parent.parent.name if source_path.parent.name == "models" else source_path.parent.name
    label_distribution = summary.get("label_distribution", {})
    rows: list[dict[str, object]] = []
    for model_name, metrics in models.items():
        rows.append(
            {
                "run_label": run_label,
                "model": model_name,
                "is_best_model": model_name == best_model,
                "roc_auc": metrics.get("roc_auc"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "n_total": summary.get("n_total"),
                "n_train": summary.get("n_train"),
                "n_test": summary.get("n_test"),
                "normal_rows": label_distribution.get("0", label_distribution.get(0)),
                "anomaly_rows": label_distribution.get("1", label_distribution.get(1)),
                "dataset": summary.get("dataset"),
                "output_dir": summary.get("output_dir"),
                "summary_path": str(source_path),
                "generated_at_utc": summary.get("generated_at_utc"),
            }
        )
    return rows


def build_comparison_frame(summary_paths: list[Path]) -> pd.DataFrame:
    """Load all summaries and return a flat comparison DataFrame."""
    rows: list[dict[str, object]] = []
    for path in summary_paths:
        with open(path, "r", encoding="utf-8") as f:
            rows.extend(summary_to_rows(json.load(f), path))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["run_label", "roc_auc", "f1", "model"],
        ascending=[True, False, False, True],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Petrobras training runs")
    ap.add_argument("summaries", nargs="*", help="Explicit training summary JSON paths")
    ap.add_argument("--summary-glob", type=str, default=None,
                    help="Glob for training summary JSON files")
    ap.add_argument("--output-csv", type=str, default=None,
                    help="Optional output CSV path")
    args = ap.parse_args()

    summary_paths = collect_summary_paths(args.summary_glob, args.summaries)
    if not summary_paths:
        raise SystemExit("No summary files matched.")

    frame = build_comparison_frame(summary_paths)
    if frame.empty:
        raise SystemExit("Matched summary files contained no model rows.")

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
        print(f"Saved comparison CSV -> {output_csv}")
    else:
        print(frame.to_csv(index=False))


if __name__ == "__main__":
    main()
