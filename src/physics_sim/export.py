"""Dataset export: wide-format SCADA CSV, scenario metadata, optional long format."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.physics_sim.scenarios import ScenarioSpec


def export_scada_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Write the wide-format SCADA timeseries to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_scada_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Write the wide-format SCADA timeseries to Parquet (much smaller)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def export_metadata(specs: Sequence[ScenarioSpec], path: Path | str) -> None:
    """Write one-row-per-scenario metadata CSV."""
    rows = []
    for spec in specs:
        leak = spec.config.leak
        rows.append({
            "scenario_id": spec.scenario_id,
            "leak_exists": int(spec.leak_exists),
            "leak_start_time": leak.start_time if leak else None,
            "leak_orifice_d_mm": round(leak.max_orifice_diameter * 1000, 2) if leak else None,
            "leak_position_m": round(leak.position, 1) if leak else None,
            "leak_onset_type": leak.onset_type if leak else None,
            "leak_onset_duration_s": leak.onset_duration if leak else None,
            "pipe_length_m": spec.config.pipe.length,
            "pipe_diameter_m": spec.config.pipe.diameter,
            "pipe_roughness_m": spec.config.pipe.roughness,
            "fluid": spec.config.fluid.name,
            "P_inlet_Pa": spec.config.boundary.inlet_pressure,
            "P_outlet_Pa": spec.config.boundary.outlet_pressure,
            "demand_amplitude": spec.config.boundary.demand_amplitude,
            "sensor_P_noise_Pa": spec.config.sensors.pressure_noise_std,
            "sensor_Q_noise": spec.config.sensors.flow_noise_std,
            "duration_s": spec.config.duration,
            "notes": spec.notes,
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def export_long_format(df: pd.DataFrame, path: Path | str) -> None:
    """Melt the wide SCADA table into long sensor format.

    Output columns: time, scenario_id, sensor_id, variable, value
    """
    id_cols = ["time", "scenario_id"]
    sensor_cols = [c for c in df.columns if c.startswith(("P_", "Q_", "T_"))]
    if not sensor_cols:
        return

    long = df[id_cols + sensor_cols].melt(
        id_vars=id_cols, var_name="raw_col", value_name="value",
    )
    var_map = {"P": "pressure", "Q": "flow", "T": "temperature"}
    split = long["raw_col"].str.split("_", n=1, expand=True)
    long["variable"] = split[0].map(var_map)
    long["sensor_id"] = split[1]
    long = long.drop(columns=["raw_col"])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(path, index=False)
