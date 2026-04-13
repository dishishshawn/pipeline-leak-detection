from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


REGIME_MULTIPLIERS: dict[str, dict[str, float]] = {
    "shutdown": {"flow": 0.08, "pressure": 0.72, "pump": 0.18, "energy": 0.28},
    "startup": {"flow": 0.65, "pressure": 0.88, "pump": 0.82, "energy": 0.78},
    "low_demand": {"flow": 0.82, "pressure": 1.05, "pump": 0.92, "energy": 0.87},
    "nominal": {"flow": 1.00, "pressure": 1.00, "pump": 1.00, "energy": 1.00},
    "high_demand": {"flow": 1.18, "pressure": 0.94, "pump": 1.10, "energy": 1.12},
    "maintenance_bypass": {"flow": 0.74, "pressure": 0.90, "pump": 0.75, "energy": 0.72},
}


@dataclass(frozen=True)
class EventSpec:
    history_id: str
    event_id: str
    event_type: str
    start_step: int
    duration_steps: int
    segment_ids: tuple[int, ...]
    intensity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def end_step(self) -> int:
        return self.start_step + self.duration_steps


def generate_regime_schedule(
    n_steps: int,
    *,
    step_minutes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    schedule = np.empty(n_steps, dtype=object)
    i = 0
    while i < n_steps:
        regime = str(
            rng.choice(
                ["low_demand", "nominal", "high_demand", "startup", "maintenance_bypass", "shutdown"],
                p=[0.17, 0.43, 0.18, 0.07, 0.08, 0.07],
            )
        )
        min_hours, max_hours = {
            "shutdown": (2, 10),
            "startup": (1, 4),
            "low_demand": (4, 18),
            "nominal": (8, 36),
            "high_demand": (3, 12),
            "maintenance_bypass": (2, 8),
        }[regime]
        dwell_steps = int(rng.integers(max(1, int(min_hours * 60 / step_minutes)), max(2, int(max_hours * 60 / step_minutes)) + 1))
        end = min(n_steps, i + dwell_steps)
        schedule[i:end] = regime
        i = end
    return schedule


def _bounded_duration(
    n_steps: int,
    step_minutes: int,
    hours_low: float,
    hours_high: float,
    rng: np.random.Generator,
) -> int:
    low = max(1, int(round(hours_low * 60 / step_minutes)))
    high = max(low + 1, int(round(hours_high * 60 / step_minutes)))
    return int(min(n_steps // 2, rng.integers(low, high)))


def _pick_start(
    n_steps: int,
    duration_steps: int,
    rng: np.random.Generator,
    reserved: list[tuple[int, int]],
    min_gap: int,
) -> int:
    if duration_steps >= n_steps:
        return 0
    for _ in range(200):
        start = int(rng.integers(0, max(1, n_steps - duration_steps)))
        end = start + duration_steps
        if all(end + min_gap <= taken_start or start >= taken_end + min_gap for taken_start, taken_end in reserved):
            reserved.append((start, end))
            return start
    start = int(rng.integers(0, max(1, n_steps - duration_steps)))
    reserved.append((start, start + duration_steps))
    return start


def build_event_plan(
    *,
    history_id: str,
    n_steps: int,
    step_minutes: int,
    segment_ids: Sequence[int],
    rng: np.random.Generator,
) -> list[EventSpec]:
    events: list[EventSpec] = []
    reserved_major: list[tuple[int, int]] = []
    horizon_days = (n_steps * step_minutes) / (60 * 24)
    mid_segment = segment_ids[len(segment_ids) // 2]

    leak_counts = {
        "slow_leak": max(1, int(round(horizon_days * 0.45))),
        "borderline_leak": max(1, int(round(horizon_days * 0.35))),
        "abrupt_leak": max(1, int(round(horizon_days * 0.20))),
    }
    nuisance_counts = {
        "demand_spike": max(2, int(round(horizon_days * 1.4))),
        "pump_wear": max(1, int(round(horizon_days * 0.65))),
        "valve_transient": max(2, int(round(horizon_days * 1.2))),
        "sensor_drift": max(1, int(round(horizon_days * 0.75))),
    }

    event_index = 0
    for event_type, count in leak_counts.items():
        for _ in range(count):
            if event_type == "slow_leak":
                duration = _bounded_duration(n_steps, step_minutes, 8, 48, rng)
                intensity = float(rng.uniform(0.32, 0.65))
            elif event_type == "borderline_leak":
                duration = _bounded_duration(n_steps, step_minutes, 10, 40, rng)
                intensity = float(rng.uniform(0.10, 0.24))
            else:
                duration = _bounded_duration(n_steps, step_minutes, 0.8, 6, rng)
                intensity = float(rng.uniform(0.70, 1.0))
            start = _pick_start(n_steps, duration, rng, reserved_major, min_gap=max(3, int(180 / step_minutes)))
            segment_id = int(rng.choice(segment_ids if len(segment_ids) > 1 else [mid_segment]))
            events.append(
                EventSpec(
                    history_id=history_id,
                    event_id=f"{history_id}_event_{event_index:03d}",
                    event_type=event_type,
                    start_step=start,
                    duration_steps=duration,
                    segment_ids=(segment_id,),
                    intensity=intensity,
                    metadata={"kind": "leak"},
                )
            )
            event_index += 1

    for event_type, count in nuisance_counts.items():
        for _ in range(count):
            if event_type == "demand_spike":
                duration = _bounded_duration(n_steps, step_minutes, 0.5, 4, rng)
                segment_choice = tuple(int(seg) for seg in segment_ids)
                intensity = float(rng.uniform(0.35, 0.9))
            elif event_type == "pump_wear":
                duration = _bounded_duration(n_steps, step_minutes, 6, 36, rng)
                segment_choice = (int(rng.choice(segment_ids)),)
                intensity = float(rng.uniform(0.35, 0.8))
            elif event_type == "sensor_drift":
                duration = _bounded_duration(n_steps, step_minutes, 8, 72, rng)
                segment_choice = (int(rng.choice(segment_ids)),)
                intensity = float(rng.uniform(0.25, 0.75))
            else:
                duration = _bounded_duration(n_steps, step_minutes, 0.25, 2, rng)
                segment_choice = (int(rng.choice(segment_ids)),)
                intensity = float(rng.uniform(0.30, 0.8))
            start = _pick_start(n_steps, duration, rng, reserved_major, min_gap=max(1, int(60 / step_minutes)))
            events.append(
                EventSpec(
                    history_id=history_id,
                    event_id=f"{history_id}_event_{event_index:03d}",
                    event_type=event_type,
                    start_step=start,
                    duration_steps=duration,
                    segment_ids=segment_choice,
                    intensity=intensity,
                    metadata={"kind": "disturbance"},
                )
            )
            event_index += 1

    events.sort(key=lambda event: (event.start_step, event.event_type, event.segment_ids))
    return events
