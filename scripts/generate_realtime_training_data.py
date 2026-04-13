"""Generate long-horizon synthetic SCADA training data for realtime models.

This replaces the old pile-of-short-episodes approach with a hybrid generator:
- stochastic operational histories spanning days or weeks
- leak events shaped by physics-derived priors
- nuisance disturbances and sensor realism layered onto the histories
- blended micro-leak episodes from the legacy simulator for sensitivity
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic.generator import SyntheticHistoryConfig, save_synthetic_histories, generate_synthetic_histories
from src.simulation.core import PipelineTelemetrySimulator, SimulationConfig, make_default_profiles
from src.simulation.scenarios import build_scenarios

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_micro_leak_episodes(
    n_episodes: int,
    steps_per: int,
    segment_count: int,
    seed: int,
) -> pd.DataFrame:
    """Generate short micro-leak episodes using the legacy simulator.

    These episodes preserve the proven micro-leak signal that the old training
    pipeline used to achieve 95% micro-leak sensitivity.
    """
    profiles = make_default_profiles(segment_count)
    segment_ids = [p.segment_id for p in profiles]
    frames: list[pd.DataFrame] = []

    for i in range(n_episodes):
        config = SimulationConfig(segment_profiles=profiles, seed=seed + i)
        scenarios = build_scenarios("micro_leak", segment_ids)
        sim = PipelineTelemetrySimulator(config, scenarios=scenarios)
        sim.start()
        sim.advance_steps(steps_per)
        df = sim.snapshot()
        if df.empty:
            continue
        df["history_id"] = f"micro_leak_episode_{i:03d}"
        df["scenario_id"] = df["history_id"]
        df["operating_regime"] = "nominal"
        df["active_event_id"] = ""
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
    parser.add_argument(
        "--no-micro-leak-blend",
        action="store_true",
        default=False,
        help="Disable blending legacy micro-leak episodes",
    )
    parser.add_argument(
        "--micro-leak-episodes",
        type=int,
        default=20,
        help="Number of legacy micro-leak episodes to blend",
    )
    parser.add_argument(
        "--micro-leak-steps",
        type=int,
        default=120,
        help="Steps per legacy micro-leak episode",
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
    df, metadata = generate_synthetic_histories(config)
    logger.info("Generated %d rows of long-horizon synthetic data", len(df))

    if not args.no_micro_leak_blend:
        micro_df = generate_micro_leak_episodes(
            n_episodes=args.micro_leak_episodes,
            steps_per=args.micro_leak_steps,
            segment_count=args.segment_count,
            seed=args.seed + 10000,
        )
        if not micro_df.empty:
            # Align columns — add any missing columns from synthetic data
            for col in df.columns:
                if col not in micro_df.columns:
                    micro_df[col] = df[col].iloc[0] if col in df.columns else ""
            # Keep only columns present in the synthetic data
            micro_df = micro_df[[c for c in df.columns if c in micro_df.columns]]
            df = pd.concat([df, micro_df], ignore_index=True)
            logger.info("Blended %d legacy micro-leak rows (%d episodes)", len(micro_df), args.micro_leak_episodes)

    output_path = Path(config.output_path)
    metadata_path = Path(config.metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    logger.info("Saved synthetic training data to %s (%d total rows)", output_path, len(df))
    logger.info("Saved event metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
