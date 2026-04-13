"""Generate long-horizon synthetic SCADA training data for realtime models.

This replaces the old pile-of-short-episodes approach with a hybrid generator:
- stochastic operational histories spanning days or weeks
- leak events shaped by physics-derived priors
- nuisance disturbances and sensor realism layered onto the histories
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic.generator import SyntheticHistoryConfig, save_synthetic_histories

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic realtime training data")
    parser.add_argument("--histories", type=int, default=8, help="Number of long histories to generate")
    parser.add_argument("--days-per-history", type=int, default=7, help="Length of each history in days")
    parser.add_argument("--segment-count", type=int, default=3, help="Number of pipeline segments")
    parser.add_argument("--step-minutes", type=int, default=1, help="Sampling cadence in minutes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample/realtime_training_data.csv",
        help="Synthetic dataset CSV path",
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        default="data/sample/realtime_training_metadata.csv",
        help="Event metadata CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticHistoryConfig(
        n_histories=args.histories,
        days_per_history=args.days_per_history,
        segment_count=args.segment_count,
        step_minutes=args.step_minutes,
        seed=args.seed,
        output_path=args.output,
        metadata_path=args.metadata_output,
    )
    output_path, metadata_path = save_synthetic_histories(config)
    logger.info("Saved synthetic training data to %s", Path(output_path))
    logger.info("Saved event metadata to %s", Path(metadata_path))


if __name__ == "__main__":
    main()
