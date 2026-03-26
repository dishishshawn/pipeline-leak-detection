from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import random
from typing import Protocol, Sequence

import pandas as pd


@dataclass(frozen=True)
class SegmentProfile:
    segment_id: int
    base_pressure: float = 70.0
    base_flow_rate: float = 3.0
    base_temperature: float = 25.0
    base_pump_speed: float = 1225.0
    valve_status: int = 1
    pump_state: int = 1
    compressor_state: int = 1
    pressure_noise: float = 0.6
    flow_noise: float = 0.12
    temperature_noise: float = 0.2


@dataclass
class SegmentState:
    segment_id: int
    pressure: float
    flow_rate: float
    temperature: float
    valve_status: int
    pump_state: int
    pump_speed: float
    compressor_state: int
    energy_consumption: float
    alarm_triggered: int
    event_type: str
    target: int
    scenario_context: str = "nominal"
    leak_severity: float = 0.0
    pump_efficiency: float = 1.0

    def to_record(self, timestamp: datetime) -> dict:
        return {
            "timestamp": timestamp,
            "segment_id": self.segment_id,
            "pressure": round(self.pressure, 3),
            "flow_rate": round(self.flow_rate, 3),
            "temperature": round(self.temperature, 3),
            "valve_status": int(self.valve_status),
            "pump_state": int(self.pump_state),
            "pump_speed": round(self.pump_speed, 3),
            "compressor_state": int(self.compressor_state),
            "energy_consumption": round(self.energy_consumption, 3),
            "alarm_triggered": int(self.alarm_triggered),
            "event_type": self.event_type,
            "target": int(self.target),
            "scenario_context": self.scenario_context,
            "leak_severity": round(self.leak_severity, 3),
            "pump_efficiency": round(self.pump_efficiency, 3),
        }


@dataclass(frozen=True)
class SimulationConfig:
    segment_profiles: Sequence[SegmentProfile]
    start_time: datetime = datetime(2026, 1, 1, 6, 0, 0)
    tick_seconds: float = 1.0
    step_minutes: int = 1
    history_limit: int = 720
    seed: int = 42


@dataclass(frozen=True)
class SimulationContext:
    step_index: int
    timestamp: datetime
    config: SimulationConfig
    rng: random.Random


class Scenario(Protocol):
    key: str
    label: str
    description: str

    def apply(self, state: SegmentState, context: SimulationContext) -> SegmentState:
        ...


def make_default_profiles(segment_count: int = 3) -> tuple[SegmentProfile, ...]:
    # Segments are differentiated like real pipeline zones: upstream high-pressure
    # trunk, midstream distribution, downstream low-pressure branches.
    _templates = [
        dict(base_pressure=74.0, base_flow_rate=3.6, base_temperature=22.5,
             base_pump_speed=1340.0, pressure_noise=0.8, flow_noise=0.22),
        dict(base_pressure=67.0, base_flow_rate=2.9, base_temperature=25.0,
             base_pump_speed=1180.0, pressure_noise=0.7, flow_noise=0.28),
        dict(base_pressure=61.0, base_flow_rate=2.3, base_temperature=27.5,
             base_pump_speed=1050.0, pressure_noise=0.9, flow_noise=0.18),
        dict(base_pressure=70.0, base_flow_rate=3.1, base_temperature=24.0,
             base_pump_speed=1220.0, pressure_noise=0.75, flow_noise=0.24),
        dict(base_pressure=64.0, base_flow_rate=2.6, base_temperature=26.0,
             base_pump_speed=1110.0, pressure_noise=0.85, flow_noise=0.20),
        dict(base_pressure=58.0, base_flow_rate=2.0, base_temperature=29.0,
             base_pump_speed=980.0, pressure_noise=1.0, flow_noise=0.16),
    ]
    profiles = []
    for index in range(segment_count):
        tpl = _templates[index % len(_templates)]
        profiles.append(SegmentProfile(segment_id=index + 1, **tpl))
    return tuple(profiles)


def _append_context(current: str, tag: str) -> str:
    if not current or current == "nominal":
        return tag
    parts = [part for part in current.split("|") if part]
    if tag not in parts:
        parts.append(tag)
    return "|".join(parts)


class PipelineTelemetrySimulator:
    """
    Real-time telemetry simulator that emits SCADA-shaped rows on wall-clock ticks.
    """

    def __init__(
        self,
        config: SimulationConfig,
        scenarios: Sequence[Scenario] | None = None,
    ) -> None:
        self.config = config
        self.scenarios = tuple(scenarios or ())
        self._rng = random.Random(config.seed)
        self._step_index = 0
        self._running = False
        self._history: deque[dict] = deque(
            maxlen=max(1, config.history_limit * len(config.segment_profiles))
        )
        self._state_by_segment: dict[int, SegmentState] = {}
        self._next_tick_wall: datetime | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, wall_clock: datetime | None = None) -> None:
        now = wall_clock or datetime.now()
        self._running = True
        if self._next_tick_wall is None:
            self._next_tick_wall = now

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self._step_index = 0
        self._running = False
        self._history.clear()
        self._state_by_segment.clear()
        self._next_tick_wall = None
        self._rng = random.Random(self.config.seed)

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def segment_ids(self) -> list[int]:
        return [profile.segment_id for profile in self.config.segment_profiles]

    def add_scenario(self, scenario: Scenario) -> None:
        self.scenarios = (*self.scenarios, scenario)

    def snapshot(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._history))

    def advance(self, wall_clock: datetime | None = None) -> pd.DataFrame:
        if not self._running:
            return pd.DataFrame()

        now = wall_clock or datetime.now()
        if self._next_tick_wall is None:
            self._next_tick_wall = now

        emitted_rows: list[dict] = []
        while self._next_tick_wall is not None and now >= self._next_tick_wall:
            emitted_rows.extend(self._emit_step())
            self._next_tick_wall += timedelta(seconds=self.config.tick_seconds)

        if emitted_rows:
            self._history.extend(emitted_rows)
        return pd.DataFrame(emitted_rows)

    def advance_steps(self, steps: int) -> pd.DataFrame:
        if not self._running or steps <= 0:
            return pd.DataFrame()

        emitted_rows: list[dict] = []
        for _ in range(steps):
            emitted_rows.extend(self._emit_step())

        if self._next_tick_wall is None:
            self._next_tick_wall = datetime.now() + timedelta(seconds=self.config.tick_seconds)
        else:
            self._next_tick_wall += timedelta(seconds=self.config.tick_seconds * steps)

        if emitted_rows:
            self._history.extend(emitted_rows)
        return pd.DataFrame(emitted_rows)

    def _emit_step(self) -> list[dict]:
        timestamp = self.config.start_time + timedelta(
            minutes=self._step_index * self.config.step_minutes
        )
        context = SimulationContext(
            step_index=self._step_index,
            timestamp=timestamp,
            config=self.config,
            rng=self._rng,
        )

        rows = []
        for profile in self.config.segment_profiles:
            previous = self._state_by_segment.get(profile.segment_id)
            state = self._build_nominal_state(profile, previous, context)
            for scenario in self.scenarios:
                state = scenario.apply(state, context)
            state.energy_consumption = max(state.energy_consumption, 0.0)
            state.pump_speed = max(state.pump_speed, 0.0)
            state.flow_rate = max(state.flow_rate, 0.0)
            self._state_by_segment[profile.segment_id] = state
            rows.append(state.to_record(timestamp))

        self._step_index += 1
        return rows

    def _build_nominal_state(
        self,
        profile: SegmentProfile,
        previous: SegmentState | None,
        context: SimulationContext,
    ) -> SegmentState:
        rng = context.rng
        minute_of_day = context.timestamp.hour * 60 + context.timestamp.minute
        daily_cycle = math.sin(2.0 * math.pi * (minute_of_day / 1440.0 - 0.22))
        shoulder_cycle = math.sin(
            4.0 * math.pi * (minute_of_day / 1440.0 + profile.segment_id * 0.04)
        )
        demand_multiplier = 1.0 + 0.13 * daily_cycle + 0.04 * shoulder_cycle

        pressure_target = profile.base_pressure * (1.0 - 0.08 * (demand_multiplier - 1.0))
        flow_target = profile.base_flow_rate * demand_multiplier
        temperature_target = profile.base_temperature + 1.4 * math.sin(
            2.0 * math.pi * (minute_of_day / 1440.0 - 0.08)
        )
        pump_speed_target = profile.base_pump_speed * (0.96 + 0.1 * demand_multiplier)

        # Occasional sensor spike (~2% chance) — realistic SCADA artifact
        spike = 0.0
        if rng.random() < 0.02:
            spike = rng.choice([-1, 1]) * rng.uniform(1.5, 4.0)

        if previous is None:
            pressure = pressure_target + rng.gauss(0.0, profile.pressure_noise)
            flow_rate = flow_target + rng.gauss(0.0, profile.flow_noise)
            temperature = temperature_target + rng.gauss(0.0, profile.temperature_noise)
            pump_speed = pump_speed_target + rng.gauss(0.0, 15.0)
        else:
            pressure = previous.pressure + 0.45 * (pressure_target - previous.pressure)
            pressure += rng.gauss(0.0, profile.pressure_noise) + spike
            flow_rate = previous.flow_rate + 0.5 * (flow_target - previous.flow_rate)
            flow_rate += rng.gauss(0.0, profile.flow_noise)
            # Correlated flow dip when pressure spikes down
            if spike < -1.0:
                flow_rate -= abs(spike) * 0.08
            temperature = previous.temperature + 0.35 * (temperature_target - previous.temperature)
            temperature += rng.gauss(0.0, profile.temperature_noise)
            pump_speed = previous.pump_speed + 0.5 * (pump_speed_target - previous.pump_speed)
            pump_speed += rng.gauss(0.0, 12.0)

        compressor_state = profile.compressor_state if demand_multiplier < 1.1 else 1
        energy_consumption = 38.0
        energy_consumption += max(pump_speed, 0.0) * 0.038
        energy_consumption += max(flow_rate, 0.0) * 11.5
        energy_consumption += compressor_state * 7.5

        return SegmentState(
            segment_id=profile.segment_id,
            pressure=pressure,
            flow_rate=flow_rate,
            temperature=temperature,
            valve_status=profile.valve_status,
            pump_state=profile.pump_state,
            pump_speed=pump_speed,
            compressor_state=compressor_state,
            energy_consumption=energy_consumption,
            alarm_triggered=0,
            event_type="normal",
            target=0,
            scenario_context="nominal",
            leak_severity=0.0,
            pump_efficiency=1.0,
        )


__all__ = [
    "PipelineTelemetrySimulator",
    "Scenario",
    "SegmentProfile",
    "SegmentState",
    "SimulationConfig",
    "SimulationContext",
    "_append_context",
    "make_default_profiles",
]
