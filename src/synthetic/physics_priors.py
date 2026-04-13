from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.physics_sim.config import (
    BoundaryConfig,
    FluidConfig,
    LeakConfig,
    PipeConfig,
    SensorConfig,
    SimConfig,
    TemperatureConfig,
)
from src.physics_sim.solver import PipelineSimulator


def _fallback_signature(length: int, intensity: float) -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, num=max(length, 2))
    rise = 1.0 / (1.0 + np.exp(-10.0 * (x - 0.25)))
    tail = 0.72 + 0.28 * np.exp(-4.0 * np.maximum(x - 0.7, 0))
    shape = rise * tail
    return pd.DataFrame(
        {
            "pressure_effect": -0.95 * shape * intensity,
            "flow_effect": -0.55 * shape * intensity,
            "temperature_effect": 0.12 * shape * intensity,
            "severity": np.clip(shape * intensity, 0.0, 1.0),
        }
    )


@lru_cache(maxsize=64)
def _paired_physics_signature(seed: int, onset_type: str, diameter_mm: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    duration = 3600.0
    boundary = BoundaryConfig(
        inlet_pressure=float(rng.uniform(12e6, 28e6)),
        outlet_pressure=float(rng.uniform(1.5e6, 6e6)),
        demand_amplitude=float(rng.uniform(0.03, 0.08)),
        demand_period=float(rng.uniform(2400.0, 4200.0)),
    )
    base = SimConfig(
        pipe=PipeConfig(n_cells=10),
        fluid=FluidConfig(),
        boundary=boundary,
        temperature=TemperatureConfig(
            inlet_temperature=float(rng.uniform(330.0, 390.0)),
            ground_temperature=float(rng.uniform(278.0, 295.0)),
        ),
        sensors=SensorConfig(sampling_interval=10.0),
        duration=duration,
        seed=seed,
    )
    leak = SimConfig(
        pipe=base.pipe,
        fluid=base.fluid,
        boundary=base.boundary,
        temperature=base.temperature,
        sensors=base.sensors,
        duration=base.duration,
        seed=base.seed,
        leak=LeakConfig(
            position=float(rng.uniform(0.25, 0.75)) * base.pipe.length,
            max_orifice_diameter=diameter_mm / 1000.0,
            start_time=float(rng.uniform(480.0, 1200.0)),
            onset_type=onset_type,  # type: ignore[arg-type]
            onset_duration=0.0 if onset_type == "sudden" else float(rng.uniform(120.0, 600.0)),
        ),
    )

    base_sim = PipelineSimulator(base)
    leak_sim = PipelineSimulator(leak)
    base_df = base_sim.measure(base_sim.run(), scenario_id="base")
    leak_df = leak_sim.measure(leak_sim.run(), scenario_id="leak")
    merged = pd.DataFrame(
        {
            "minute": np.floor(base_df["time"].to_numpy() / 60.0).astype(int),
            "pressure_effect": leak_df["P_mid"].to_numpy() - base_df["P_mid"].to_numpy(),
            "flow_effect": leak_df["Q_inlet"].to_numpy() - base_df["Q_inlet"].to_numpy(),
            "temperature_effect": leak_df["T_mid"].to_numpy() - base_df["T_mid"].to_numpy(),
            "severity": np.clip(leak_df["leak_rate"].to_numpy() / max(leak_df["leak_rate"].max(), 1e-6), 0.0, 1.0),
        }
    )
    minute_df = merged.groupby("minute", as_index=False).mean(numeric_only=True)
    for col in ("pressure_effect", "flow_effect", "temperature_effect"):
        scale = max(np.abs(minute_df[col]).max(), 1e-6)
        minute_df[col] = minute_df[col] / scale
    return minute_df[["pressure_effect", "flow_effect", "temperature_effect", "severity"]]


def sample_leak_signature(
    *,
    duration_steps: int,
    intensity: float,
    seed: int,
    onset_type: str = "ramp",
) -> pd.DataFrame:
    if duration_steps <= 0:
        return _fallback_signature(2, intensity)

    diameter_mm = int(round(5 + 18 * float(intensity)))
    try:
        base = _paired_physics_signature(seed, onset_type, diameter_mm)
    except Exception:
        return _fallback_signature(duration_steps, intensity)

    if base.empty:
        return _fallback_signature(duration_steps, intensity)

    source_x = np.linspace(0.0, 1.0, num=len(base))
    target_x = np.linspace(0.0, 1.0, num=duration_steps)
    out = {}
    for col in base.columns:
        out[col] = np.interp(target_x, source_x, base[col].to_numpy())
    signature = pd.DataFrame(out)
    signature["pressure_effect"] *= intensity
    signature["flow_effect"] *= intensity
    signature["temperature_effect"] *= intensity
    signature["severity"] = np.clip(signature["severity"] * intensity, 0.0, 1.0)
    return signature
