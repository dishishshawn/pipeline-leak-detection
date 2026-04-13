from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.simulation.core import _append_context


def _inject_stale_blocks(
    values: np.ndarray,
    *,
    block_probability: float,
    max_block_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = values.copy()
    i = 1
    while i < len(out):
        if rng.random() < block_probability:
            block = int(rng.integers(1, max_block_steps + 1))
            out[i : i + block] = out[i - 1]
            i += block
        else:
            i += 1
    return out


def apply_sensor_realism(
    df: pd.DataFrame,
    *,
    step_minutes: int,
    rng: np.random.Generator,
    signal_columns: Sequence[str] = ("pressure", "flow_rate", "temperature", "pump_speed", "energy_consumption"),
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    ordered = out.sort_values(["history_id", "segment_id", "timestamp"]).copy()
    issue_mask = np.zeros(len(ordered), dtype=bool)

    for (_, segment_id), idx in ordered.groupby(["history_id", "segment_id"], sort=False).groups.items():
        loc = np.array(idx)
        local_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)) + int(segment_id))
        for col in signal_columns:
            if col not in ordered.columns:
                continue
            values = ordered.loc[loc, col].to_numpy(dtype=float)
            scale = {
                "pressure": 0.0025,
                "flow_rate": 0.0060,
                "temperature": 0.0015,
                "pump_speed": 0.0018,
                "energy_consumption": 0.0022,
            }.get(col, 0.002)
            drift = np.cumsum(local_rng.normal(0.0, scale * max(np.nanstd(values), 1.0), size=len(values)))
            values = values + drift
            spike_mask = local_rng.random(len(values)) < 0.0015
            if spike_mask.any():
                spike_scale = max(np.nanstd(values), 1.0)
                values[spike_mask] += local_rng.normal(0.0, 3.5 * spike_scale, size=spike_mask.sum())
                issue_mask[loc[spike_mask]] = True
            values = _inject_stale_blocks(
                values,
                block_probability=min(0.01, 0.0025 * step_minutes),
                max_block_steps=max(2, int(30 / step_minutes)),
                rng=local_rng,
            )
            if np.any(np.diff(values) == 0):
                issue_mask[loc[1:][np.diff(values) == 0]] = True
            ordered.loc[loc, col] = values

        pressure_drop = ordered.loc[loc, "pressure"].diff().abs().fillna(0.0).to_numpy()
        if len(loc) > 1 and pressure_drop.mean() > 0:
            dropout_mask = local_rng.random(len(loc)) < 0.001
            if dropout_mask.any():
                held_pressure = ordered.loc[loc, "pressure"].shift(1).bfill()
                held_flow = ordered.loc[loc, "flow_rate"].shift(1).bfill()
                ordered.loc[loc[dropout_mask], "pressure"] = held_pressure.loc[loc[dropout_mask]].to_numpy()
                ordered.loc[loc[dropout_mask], "flow_rate"] = held_flow.loc[loc[dropout_mask]].to_numpy()
                issue_mask[loc[dropout_mask]] = True

    if "scenario_context" in ordered.columns:
        for index in np.flatnonzero(issue_mask):
            ordered.iat[index, ordered.columns.get_loc("scenario_context")] = _append_context(
                str(ordered.iat[index, ordered.columns.get_loc("scenario_context")]),
                "sensor_issue",
            )

    for col in ("pressure", "flow_rate", "temperature", "pump_speed", "energy_consumption"):
        if col in ordered.columns:
            ordered[col] = ordered[col].clip(lower=0 if col != "temperature" else None)

    return ordered.sort_index()
