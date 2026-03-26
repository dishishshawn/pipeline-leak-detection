"""Generate training data from the simulator for realtime models.

Runs multiple scenarios to produce a balanced dataset that matches
the actual value distributions the live view will see.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.simulation import (
    PipelineTelemetrySimulator,
    SimulationConfig,
    build_scenarios,
    make_default_profiles,
)
from src.simulation.scenarios import LeakProgressionScenario, PumpWearScenario

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/sample")
OUTPUT_FILE = OUTPUT_DIR / "realtime_training_data.csv"

PROFILES = make_default_profiles(3)
SEGMENT_IDS = [p.segment_id for p in PROFILES]
STEPS_PER_RUN = 120  # 2 hours of simulated data per scenario


def _run_scenario(preset_key: str, seed: int) -> pd.DataFrame:
    """Run a scenario and return the full history."""
    scenarios = build_scenarios(preset_key, SEGMENT_IDS)
    config = SimulationConfig(segment_profiles=PROFILES, seed=seed)
    sim = PipelineTelemetrySimulator(config, scenarios)
    sim.start()
    df = sim.advance_steps(STEPS_PER_RUN)
    return df


def _run_custom_leak(start_step: int, segment_id: int, seed: int,
                     ramp: int = 10, hold: int = 20, recovery: int = 8,
                     max_severity: float = 1.0) -> pd.DataFrame:
    """Run a custom leak scenario for variety."""
    scenarios = [
        LeakProgressionScenario(
            start_step=start_step, ramp_steps=ramp, hold_steps=hold,
            recovery_steps=recovery, max_severity=max_severity,
            affected_segments=[segment_id],
        )
    ]
    config = SimulationConfig(segment_profiles=PROFILES, seed=seed)
    sim = PipelineTelemetrySimulator(config, scenarios)
    sim.start()
    return sim.advance_steps(STEPS_PER_RUN)


def main():
    frames = []

    # Normal operation (multiple seeds for variety)
    for i, seed in enumerate([42, 99, 137, 200]):
        logger.info("Generating steady_state run %d (seed=%d)", i + 1, seed)
        frames.append(_run_scenario("steady_state", seed))

    # Leak scenarios
    for i, seed in enumerate([10, 55, 88]):
        logger.info("Generating slow_seep run %d (seed=%d)", i + 1, seed)
        frames.append(_run_scenario("slow_seep", seed))

    # Demand shock (non-leak stress — models must learn to NOT flag this)
    for i, seed in enumerate([33, 77]):
        logger.info("Generating demand_shock run %d (seed=%d)", i + 1, seed)
        frames.append(_run_scenario("demand_shock", seed))

    # Compound incident
    logger.info("Generating compound_incident")
    frames.append(_run_scenario("compound_incident", seed=42))

    # Custom leaks at different times/segments for diversity
    for seg_id in SEGMENT_IDS:
        for start in [5, 25, 50]:
            logger.info("Generating custom leak seg=%d start=%d", seg_id, start)
            frames.append(_run_custom_leak(start, seg_id, seed=seg_id * 100 + start))

    # Fast ruptures
    for seg_id in SEGMENT_IDS:
        logger.info("Generating rupture seg=%d", seg_id)
        frames.append(_run_custom_leak(
            10, seg_id, seed=seg_id * 200,
            ramp=4, hold=14, recovery=8, max_severity=1.35,
        ))

    df = pd.concat(frames, ignore_index=True)

    # Drop ground-truth columns that won't be available at inference time
    # but keep 'target' as the label
    df = df.drop(columns=["leak_severity", "pump_efficiency"], errors="ignore")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    n_pos = int(df["target"].sum())
    n_neg = len(df) - n_pos
    logger.info("Saved %d rows to %s (positive=%d, negative=%d, ratio=%.1f%%)",
                len(df), OUTPUT_FILE, n_pos, n_neg, 100 * n_pos / len(df))


if __name__ == "__main__":
    main()
