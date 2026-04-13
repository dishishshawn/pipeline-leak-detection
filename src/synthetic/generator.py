from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.core import make_default_profiles, _append_context
from src.synthetic.physics_priors import sample_leak_signature
from src.synthetic.realism import apply_sensor_realism
from src.synthetic.scenario_engine import (
    EventSpec,
    REGIME_MULTIPLIERS,
    build_event_plan,
    generate_regime_schedule,
)


@dataclass(frozen=True)
class SyntheticHistoryConfig:
    n_histories: int = 8
    days_per_history: int = 7
    segment_count: int = 3
    step_minutes: int = 1
    seed: int = 42
    output_path: str = "data/sample/realtime_training_data.csv"
    metadata_path: str = "data/sample/realtime_training_metadata.csv"


def _baseline_segment_frame(
    *,
    timestamps: pd.DatetimeIndex,
    history_id: str,
    segment_id: int,
    base_pressure: float,
    base_flow: float,
    base_temperature: float,
    base_pump_speed: float,
    regime_schedule: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_steps = len(timestamps)
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    day_of_week = timestamps.dayofweek.to_numpy()
    daily = np.sin(2.0 * np.pi * (minute_of_day / 1440.0 - 0.18))
    shoulder = np.sin(4.0 * np.pi * (minute_of_day / 1440.0 + segment_id * 0.03))
    weekly = np.sin(2.0 * np.pi * (day_of_week / 7.0 + segment_id * 0.09))
    ambient = np.sin(2.0 * np.pi * (minute_of_day / 1440.0 - 0.05))

    demand = 1.0 + 0.11 * daily + 0.03 * shoulder + 0.06 * weekly
    demand += np.cumsum(rng.normal(0.0, 0.002, size=n_steps))
    demand = np.clip(demand, 0.45, 1.55)

    pressure_target = base_pressure * (1.02 - 0.10 * (demand - 1.0))
    flow_target = base_flow * demand
    temperature_target = base_temperature + 1.2 * ambient + 0.6 * weekly
    pump_target = base_pump_speed * (0.95 + 0.12 * demand)

    pressure = np.empty(n_steps)
    flow = np.empty(n_steps)
    temperature = np.empty(n_steps)
    pump = np.empty(n_steps)
    pressure[0] = pressure_target[0]
    flow[0] = flow_target[0]
    temperature[0] = temperature_target[0]
    pump[0] = pump_target[0]

    for i in range(1, n_steps):
        pressure[i] = pressure[i - 1] + 0.26 * (pressure_target[i] - pressure[i - 1]) + rng.normal(0.0, 0.18)
        flow[i] = flow[i - 1] + 0.34 * (flow_target[i] - flow[i - 1]) + rng.normal(0.0, 0.05)
        temperature[i] = temperature[i - 1] + 0.18 * (temperature_target[i] - temperature[i - 1]) + rng.normal(0.0, 0.05)
        pump[i] = pump[i - 1] + 0.30 * (pump_target[i] - pump[i - 1]) + rng.normal(0.0, 5.0)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "history_id": history_id,
            "scenario_id": history_id,
            "segment_id": segment_id,
            "pressure": pressure,
            "flow_rate": np.clip(flow, 0.01, None),
            "temperature": temperature,
            "pump_speed": np.clip(pump, 0.0, None),
            "valve_status": 1,
            "pump_state": 1,
            "compressor_state": 1,
            "alarm_triggered": 0,
            "event_type": "normal",
            "target": 0,
            "scenario_context": "nominal",
            "leak_severity": 0.0,
            "pump_efficiency": 1.0,
            "operating_regime": regime_schedule.astype(str),
            "active_event_id": "",
        }
    )

    multipliers = pd.DataFrame([REGIME_MULTIPLIERS[str(regime)] for regime in regime_schedule])
    frame["flow_rate"] *= multipliers["flow"].to_numpy()
    frame["pressure"] *= multipliers["pressure"].to_numpy()
    frame["pump_speed"] *= multipliers["pump"].to_numpy()
    energy = 36.0 + frame["pump_speed"] * 0.035 + frame["flow_rate"] * 11.0
    frame["energy_consumption"] = energy * multipliers["energy"].to_numpy()

    shutdown_mask = frame["operating_regime"].eq("shutdown")
    frame.loc[shutdown_mask, "pump_state"] = 0
    frame.loc[shutdown_mask, "compressor_state"] = 0
    frame.loc[shutdown_mask, "event_type"] = "maintenance"
    startup_mask = frame["operating_regime"].eq("startup")
    frame.loc[startup_mask, "event_type"] = "warning"
    maintenance_mask = frame["operating_regime"].eq("maintenance_bypass")
    frame.loc[maintenance_mask, "scenario_context"] = "maintenance_bypass"
    frame.loc[maintenance_mask, "event_type"] = "warning"
    return frame


def _apply_leak_event(
    df: pd.DataFrame,
    event: EventSpec,
    *,
    step_minutes: int,
    seed: int,
) -> None:
    onset_type = "sudden" if event.event_type == "abrupt_leak" else "ramp"
    signature = sample_leak_signature(
        duration_steps=event.duration_steps,
        intensity=event.intensity,
        seed=seed,
        onset_type=onset_type,
    )
    idx = (df["segment_id"].isin(event.segment_ids)) & (df["step_index"].between(event.start_step, event.end_step - 1))
    segment_rows = df.loc[idx].copy()
    if segment_rows.empty:
        return
    sig = signature.iloc[: len(segment_rows)].reset_index(drop=True)
    pressure_scale = max(segment_rows["pressure"].median(), 1.0)
    flow_scale = max(segment_rows["flow_rate"].median(), 0.5)
    segment_rows["pressure"] += sig["pressure_effect"].to_numpy() * pressure_scale * 0.12
    segment_rows["flow_rate"] += sig["flow_effect"].to_numpy() * flow_scale * 0.22
    segment_rows["temperature"] += sig["temperature_effect"].to_numpy() * 6.0
    severity = sig["severity"].to_numpy()
    if event.event_type == "borderline_leak":
        severity *= 0.82
    if event.event_type == "slow_leak":
        severity *= 0.94
    segment_rows["leak_severity"] = np.maximum(segment_rows["leak_severity"], severity)
    segment_rows["target"] = np.where(segment_rows["leak_severity"] >= (0.10 if event.event_type == "borderline_leak" else 0.12), 1, segment_rows["target"])
    segment_rows["event_type"] = np.where(
        segment_rows["leak_severity"] >= 0.45,
        "fault",
        np.where(segment_rows["leak_severity"] >= 0.12, "warning", segment_rows["event_type"]),
    )
    segment_rows["alarm_triggered"] = np.where(segment_rows["leak_severity"] >= 0.35, 1, segment_rows["alarm_triggered"])
    segment_rows["scenario_context"] = [
        _append_context(ctx, event.event_type) for ctx in segment_rows["scenario_context"].astype(str)
    ]
    segment_rows["active_event_id"] = event.event_id
    segment_rows["energy_consumption"] += np.clip(severity, 0.0, 1.0) * 8.0
    df.loc[idx, segment_rows.columns] = segment_rows


def _apply_disturbance_event(df: pd.DataFrame, event: EventSpec) -> None:
    idx = (df["segment_id"].isin(event.segment_ids)) & (df["step_index"].between(event.start_step, event.end_step - 1))
    if not idx.any():
        return
    segment_rows = df.loc[idx].copy()
    x = np.linspace(0.0, 1.0, num=len(segment_rows))
    smooth = np.sin(np.pi * np.clip(x, 0.0, 1.0)) ** 2
    if event.event_type == "demand_spike":
        segment_rows["flow_rate"] *= 1.0 + 0.18 * event.intensity * smooth
        segment_rows["pressure"] *= 1.0 - 0.05 * event.intensity * smooth
        segment_rows["pump_speed"] *= 1.0 + 0.07 * event.intensity * smooth
        segment_rows["energy_consumption"] *= 1.0 + 0.10 * event.intensity * smooth
    elif event.event_type == "pump_wear":
        segment_rows["pump_efficiency"] *= 1.0 - 0.18 * event.intensity * smooth
        segment_rows["pressure"] *= 1.0 - 0.06 * event.intensity * smooth
        segment_rows["flow_rate"] *= 1.0 - 0.05 * event.intensity * smooth
        segment_rows["energy_consumption"] *= 1.0 + 0.08 * event.intensity * smooth
    elif event.event_type == "valve_transient":
        oscillation = np.sin(np.linspace(0.0, 3.5 * np.pi, num=len(segment_rows)))
        segment_rows["pressure"] += oscillation * event.intensity * 2.5
        segment_rows["flow_rate"] -= np.maximum(oscillation, 0.0) * event.intensity * 0.35
    elif event.event_type == "sensor_drift":
        segment_rows["pressure"] += np.linspace(0.0, event.intensity * 2.0, num=len(segment_rows))
        segment_rows["flow_rate"] += np.linspace(0.0, -event.intensity * 0.14, num=len(segment_rows))
    segment_rows["event_type"] = np.where(segment_rows["event_type"].eq("normal"), "warning", segment_rows["event_type"])
    segment_rows["scenario_context"] = [
        _append_context(ctx, event.event_type) for ctx in segment_rows["scenario_context"].astype(str)
    ]
    segment_rows["active_event_id"] = np.where(segment_rows["active_event_id"].eq(""), event.event_id, segment_rows["active_event_id"])
    df.loc[idx, segment_rows.columns] = segment_rows


def generate_synthetic_histories(config: SyntheticHistoryConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    profiles = make_default_profiles(config.segment_count)
    all_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    base_start = datetime(2026, 1, 1, 0, 0, 0)
    n_steps = int(config.days_per_history * 24 * 60 / config.step_minutes)

    for history_idx in range(config.n_histories):
        history_id = f"history_{history_idx:03d}"
        history_start = base_start + timedelta(days=history_idx * config.days_per_history)
        timestamps = pd.date_range(history_start, periods=n_steps, freq=f"{config.step_minutes}min")
        regime_schedule = generate_regime_schedule(n_steps, step_minutes=config.step_minutes, rng=np.random.default_rng(int(rng.integers(0, 1_000_000))))
        history_frames = []

        for profile in profiles:
            segment_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)) + profile.segment_id)
            frame = _baseline_segment_frame(
                timestamps=timestamps,
                history_id=history_id,
                segment_id=profile.segment_id,
                base_pressure=profile.base_pressure,
                base_flow=profile.base_flow_rate,
                base_temperature=profile.base_temperature,
                base_pump_speed=profile.base_pump_speed,
                regime_schedule=regime_schedule,
                rng=segment_rng,
            )
            history_frames.append(frame)

        history_df = pd.concat(history_frames, ignore_index=True)
        history_df["step_index"] = history_df.groupby("segment_id").cumcount()
        events = build_event_plan(
            history_id=history_id,
            n_steps=n_steps,
            step_minutes=config.step_minutes,
            segment_ids=[profile.segment_id for profile in profiles],
            rng=np.random.default_rng(int(rng.integers(0, 1_000_000))),
        )

        for event in events:
            if event.metadata.get("kind") == "leak":
                _apply_leak_event(history_df, event, step_minutes=config.step_minutes, seed=config.seed + history_idx * 101 + event.start_step)
            else:
                _apply_disturbance_event(history_df, event)
            metadata_rows.append(
                {
                    "history_id": history_id,
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "segment_ids": ",".join(str(segment_id) for segment_id in event.segment_ids),
                    "start_step": event.start_step,
                    "duration_steps": event.duration_steps,
                    "intensity": event.intensity,
                    **event.metadata,
                }
            )

        history_df = apply_sensor_realism(
            history_df.drop(columns=["step_index"]),
            step_minutes=config.step_minutes,
            rng=np.random.default_rng(int(rng.integers(0, 1_000_000))),
        )
        for col in ("pressure", "flow_rate", "temperature", "pump_speed", "energy_consumption", "leak_severity", "pump_efficiency"):
            history_df[col] = history_df[col].astype(float)
        all_frames.append(history_df)

    combined = pd.concat(all_frames, ignore_index=True).sort_values(["timestamp", "segment_id"]).reset_index(drop=True)
    combined["pressure"] = combined["pressure"].round(3)
    combined["flow_rate"] = combined["flow_rate"].round(3)
    combined["temperature"] = combined["temperature"].round(3)
    combined["pump_speed"] = combined["pump_speed"].round(3)
    combined["energy_consumption"] = combined["energy_consumption"].round(3)
    combined["leak_severity"] = combined["leak_severity"].round(3)
    combined["pump_efficiency"] = combined["pump_efficiency"].round(3)
    metadata = pd.DataFrame(metadata_rows)
    return combined, metadata


def save_synthetic_histories(config: SyntheticHistoryConfig) -> tuple[Path, Path]:
    df, metadata = generate_synthetic_histories(config)
    output_path = Path(config.output_path)
    metadata_path = Path(config.metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    return output_path, metadata_path
