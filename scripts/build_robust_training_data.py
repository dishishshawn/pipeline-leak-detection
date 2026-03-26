from __future__ import annotations

import argparse
import json

from src.data.robust_corpus import build_robust_training_corpus

DEFAULT_SIMULATOR_PATH = "data/sample/realtime_training_data.csv"
DEFAULT_DATASETS = ["water_leak", "mendeley_water_testbed"]
DEFAULT_OUTPUT = "data/processed/robust_realtime/robust_realtime_training_data.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a robust live-training corpus from simulator and external telemetry datasets."
    )
    parser.add_argument(
        "--simulator-paths",
        nargs="+",
        default=[DEFAULT_SIMULATOR_PATH],
        help="SCADA-shaped simulator CSVs to include.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset-registry keys to include when present locally.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path for the fused training corpus.",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=25000,
        help="Cap per-source rows so one dataset does not dominate the corpus.",
    )
    parser.add_argument(
        "--max-negative-ratio",
        type=float,
        default=3.0,
        help="Maximum negative-to-positive ratio retained per source.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for sampling and balancing.",
    )
    args = parser.parse_args()

    summary = build_robust_training_corpus(
        simulator_paths=args.simulator_paths,
        dataset_names=args.datasets,
        output_path=args.output,
        max_rows_per_source=args.max_rows_per_source,
        max_negative_ratio=args.max_negative_ratio,
        random_state=args.random_state,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
