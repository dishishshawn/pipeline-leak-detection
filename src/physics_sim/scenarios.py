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
    n_normal: int = 250,
    n_leak: int = 250,
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
    """Build a SimConfig sampled from real operating regimes.

    Regime mix calibrated to Petrobras 3W quantile analysis:
      - shut_in   (~12%): near-zero flow, low pressures  (matches real Q=0 mode)
      - low_prod  (~25%): moderate pressure, low flow     (covers real p10-p25)
      - nominal   (~40%): typical subsea production       (covers real p25-p75)
      - high_prod (~23%): high pressure, high flow        (covers real p75-p90)

    P_outlet calibrated lower (1-6 MPa) to match real median ~3 MPa.
    Temperatures shifted up (70-130 C) to match real T_mid median ~100 C.
    """
    regime = rng.choice(
        ["shut_in", "low_prod", "nominal", "high_prod"],
        p=[0.12, 0.25, 0.40, 0.23],
    )

    if regime == "shut_in":
        # Very low pressure differential -> near-zero flow
        p_inlet = float(rng.uniform(1.0e6, 5.0e6))
        p_outlet = float(rng.uniform(0.5e6, p_inlet - 0.2e6))
        t_inlet_C = float(rng.uniform(30.0, 60.0))
    elif regime == "low_prod":
        p_inlet = float(rng.uniform(8.0e6, 18.0e6))
        p_outlet = float(rng.uniform(1.0e6, 4.0e6))
        t_inlet_C = float(rng.uniform(60.0, 100.0))
    elif regime == "nominal":
        p_inlet = float(rng.uniform(15.0e6, 25.0e6))
        p_outlet = float(rng.uniform(1.5e6, 5.0e6))
        t_inlet_C = float(rng.uniform(90.0, 130.0))
    else:  # high_prod
        p_inlet = float(rng.uniform(25.0e6, 38.0e6))
        p_outlet = float(rng.uniform(1.5e6, 5.0e6))
        t_inlet_C = float(rng.uniform(100.0, 140.0))

    # Ensure minimum dP of 0.5 MPa (solver stability)
    p_outlet = min(p_outlet, p_inlet - 0.5e6)

    # T_inlet sensor missing ~30% of time in real data
    mask_t_inlet = bool(rng.random() < 0.30)

    bc = BoundaryConfig(
        inlet_pressure=p_inlet,
        outlet_pressure=p_outlet,
        demand_amplitude=float(rng.uniform(0.02, 0.08)),
        demand_period=float(rng.uniform(2400.0, 4800.0)),
    )
    # Ground temp varies; higher inlet temp -> hotter mid/outlet
    ground_C = float(rng.uniform(5.0, 20.0))
    temp = TemperatureConfig(
        inlet_temperature=t_inlet_C + 273.15,
        ground_temperature=ground_C + 273.15,
    )
    sensors = SensorConfig(
        pressure_noise_std=float(rng.uniform(10_000.0, 40_000.0)),
        flow_noise_std=float(rng.uniform(0.002, 0.01)),
        temperature_noise_std=float(rng.uniform(0.1, 0.3)),
        sampling_interval=10.0,
    )
    cfg = SimConfig(
        pipe=pipe,
        fluid=fluid,
        boundary=bc,
        temperature=temp,
        sensors=sensors,
        duration=duration,
        seed=int(rng.integers(0, 1_000_000)),
    )
    # Stash regime metadata on the config for export-time use
    cfg._regime = regime  # type: ignore[attr-defined]
    cfg._mask_t_inlet = mask_t_inlet  # type: ignore[attr-defined]
    return cfg
