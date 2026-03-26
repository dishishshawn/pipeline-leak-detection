"""Scenario batch generator for physics-based training data.

Creates diverse normal and leak scenarios with randomised parameters
(boundary pressures, demand amplitude, leak size / position / timing,
sensor noise level).  The resulting list of ``ScenarioSpec`` objects
can be fed directly to ``PipelineSimulator`` for batch dataset generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.physics_sim.config import (
    BoundaryConfig,
    FluidConfig,
    LeakConfig,
    PipeConfig,
    SensorConfig,
    SimConfig,
    TemperatureConfig,
)


@dataclass
class ScenarioSpec:
    """Fully self-contained specification for one simulation run."""

    scenario_id: str
    config: SimConfig
    leak_exists: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_batch(
    n_normal: int = 25,
    n_leak: int = 25,
    pipe: PipeConfig | None = None,
    fluid: FluidConfig | None = None,
    duration: float = 1800.0,
    seed: int = 42,
) -> list[ScenarioSpec]:
    """Return a shuffled list of normal + leak scenario specs.

    Normal scenarios have varied demand / noise.  Leak scenarios
    additionally randomise orifice diameter (3-30 mm), position
    (20-80 % of pipe), onset time (10-50 % of duration), and onset
    type (sudden / ramp / exponential).
    """
    rng = np.random.default_rng(seed)
    pipe = pipe or PipeConfig()
    fluid = fluid or FluidConfig()
    specs: list[ScenarioSpec] = []

    onset_types: list[str] = ["sudden", "ramp", "exponential"]

    for i in range(n_normal):
        cfg = _random_base_config(rng, pipe, fluid, duration)
        specs.append(ScenarioSpec(
            scenario_id=f"normal_{i:04d}",
            config=cfg,
            leak_exists=False,
            notes="Normal operation with demand variation",
        ))

    for i in range(n_leak):
        cfg = _random_base_config(rng, pipe, fluid, duration)

        orifice_d = float(rng.uniform(0.003, 0.030))           # 3-30 mm
        leak_pos = float(rng.uniform(0.2, 0.8)) * pipe.length  # 20-80 %
        leak_start = float(rng.uniform(0.1, 0.5)) * duration   # 10-50 %
        onset = str(rng.choice(onset_types))
        onset_dur = float(rng.uniform(30.0, 300.0)) if onset != "sudden" else 0.0

        cfg.leak = LeakConfig(
            position=leak_pos,
            max_orifice_diameter=orifice_d,
            start_time=leak_start,
            onset_type=onset,  # type: ignore[arg-type]
            onset_duration=onset_dur,
        )

        size_label = "small" if orifice_d < 0.008 else ("medium" if orifice_d < 0.018 else "large")
        specs.append(ScenarioSpec(
            scenario_id=f"leak_{size_label}_{i:04d}",
            config=cfg,
            leak_exists=True,
            notes=(
                f"{size_label} {onset} leak at {leak_pos:.0f} m, "
                f"d={orifice_d * 1000:.1f} mm, start={leak_start:.0f} s"
            ),
        ))

    # Shuffle so normal and leak scenarios are interleaved
    rng.shuffle(specs)  # type: ignore[arg-type]
    return specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_base_config(
    rng: np.random.Generator,
    pipe: PipeConfig,
    fluid: FluidConfig,
    duration: float,
) -> SimConfig:
    """Build a SimConfig with randomised boundary / sensor / seed."""
    bc = BoundaryConfig(
        inlet_pressure=float(5.0e6 + rng.uniform(-0.5e6, 0.5e6)),
        outlet_pressure=float(2.0e6 + rng.uniform(-0.3e6, 0.3e6)),
        demand_amplitude=float(rng.uniform(0.02, 0.08)),
        demand_period=float(rng.uniform(2400.0, 4800.0)),
    )
    sensors = SensorConfig(
        pressure_noise_std=float(rng.uniform(3000.0, 8000.0)),
        flow_noise_std=float(rng.uniform(0.0003, 0.0008)),
        temperature_noise_std=float(rng.uniform(0.05, 0.2)),
    )
    return SimConfig(
        pipe=pipe,
        fluid=fluid,
        boundary=bc,
        sensors=sensors,
        duration=duration,
        seed=int(rng.integers(0, 1_000_000)),
    )
