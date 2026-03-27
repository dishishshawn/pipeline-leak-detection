from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from src.simulation.core import Scenario, SegmentState, SimulationContext, _append_context


def _progress(
    step_index: int,
    *,
    start_step: int,
    ramp_steps: int,
    hold_steps: int,
    recovery_steps: int,
    residual: float = 0.0,
) -> float:
    """Compute scenario intensity with optional residual floor after recovery.

    ``residual`` (0.0-1.0) controls how much severity persists after recovery
    (e.g. 0.3 means the system never fully heals — realistic for leaks/wear).
    """
    if step_index < start_step:
        return 0.0

    elapsed = step_index - start_step
    ramp_steps = max(1, ramp_steps)
    recovery_steps = max(1, recovery_steps)

    if elapsed < ramp_steps:
        return min(1.0, (elapsed + 1) / ramp_steps)
    if elapsed < ramp_steps + hold_steps:
        return 1.0
    if elapsed < ramp_steps + hold_steps + recovery_steps:
        recovery_elapsed = elapsed - ramp_steps - hold_steps
        raw = max(0.0, 1.0 - ((recovery_elapsed + 1) / recovery_steps))
        return max(residual, raw)
    return residual


@dataclass(frozen=True)
class DemandSpikeScenario:
    key: str = "demand_spike"
    label: str = "Demand Spike"
    description: str = "Short-lived demand increase that stresses pressure and pumps."
    start_step: int = 8
    ramp_steps: int = 4
    hold_steps: int = 5
    recovery_steps: int = 5
    intensity: float = 1.0
    affected_segments: Sequence[int] | None = None

    def apply(self, state: SegmentState, context: SimulationContext) -> SegmentState:
        if self.affected_segments and state.segment_id not in self.affected_segments:
            return state

        severity = _progress(
            context.step_index,
            start_step=self.start_step,
            ramp_steps=self.ramp_steps,
            hold_steps=self.hold_steps,
            recovery_steps=self.recovery_steps,
        ) * self.intensity
        if severity <= 0.0:
            return state

        state.flow_rate *= 1.0 + 0.22 * severity
        state.pressure -= 5.5 * severity
        state.pump_speed += 120.0 * severity
        state.energy_consumption += 8.5 * severity
        state.scenario_context = _append_context(state.scenario_context, self.key)
        if state.target == 0 and severity >= 0.45:
            state.event_type = "warning"
        return state


@dataclass(frozen=True)
class PumpWearScenario:
    key: str = "pump_wear"
    label: str = "Pump Wear"
    description: str = "Pump efficiency degrades, driving lower head pressure and noisier flow."
    start_step: int = 10
    ramp_steps: int = 10
    hold_steps: int = 16
    recovery_steps: int = 4
    intensity: float = 1.0
    residual: float = 0.2
    affected_segments: Sequence[int] | None = None

    def apply(self, state: SegmentState, context: SimulationContext) -> SegmentState:
        if self.affected_segments and state.segment_id not in self.affected_segments:
            return state

        severity = _progress(
            context.step_index,
            start_step=self.start_step,
            ramp_steps=self.ramp_steps,
            hold_steps=self.hold_steps,
            recovery_steps=self.recovery_steps,
            residual=self.residual,
        ) * self.intensity
        if severity <= 0.0:
            return state

        efficiency = max(0.68, 1.0 - 0.18 * severity)
        state.pump_efficiency = efficiency
        state.pump_speed *= efficiency
        state.pressure -= 6.0 * severity
        state.flow_rate *= max(0.7, 1.0 - 0.1 * severity)
        state.temperature += 0.8 * severity
        state.energy_consumption += 10.0 * severity
        state.scenario_context = _append_context(state.scenario_context, self.key)

        if severity >= 0.45 and state.target == 0:
            state.event_type = "warning"
        if severity >= 0.8:
            state.alarm_triggered = 1
        return state


@dataclass(frozen=True)
class LeakProgressionScenario:
    key: str = "leak_progression"
    label: str = "Leak Progression"
    description: str = "Small seep ramps into a confirmed leak with alarm conditions."
    start_step: int = 12
    ramp_steps: int = 14
    hold_steps: int = 18
    recovery_steps: int = 8
    max_severity: float = 1.0
    residual: float = 0.3
    affected_segments: Sequence[int] | None = None

    def apply(self, state: SegmentState, context: SimulationContext) -> SegmentState:
        if self.affected_segments and state.segment_id not in self.affected_segments:
            return state

        severity = _progress(
            context.step_index,
            start_step=self.start_step,
            ramp_steps=self.ramp_steps,
            hold_steps=self.hold_steps,
            recovery_steps=self.recovery_steps,
            residual=self.residual,
        ) * self.max_severity
        if severity <= 0.0:
            return state

        state.leak_severity = max(state.leak_severity, severity)
        state.pressure -= 14.0 * severity
        state.flow_rate *= max(0.42, 1.0 - 0.24 * severity)
        state.temperature += 2.0 * severity
        state.energy_consumption += 12.0 * severity
        state.target = 1 if severity >= 0.18 else state.target
        state.alarm_triggered = 1 if severity >= 0.55 else state.alarm_triggered
        state.scenario_context = _append_context(state.scenario_context, self.key)

        if severity >= 0.7:
            state.event_type = "fault"
        elif severity >= 0.25:
            state.event_type = "warning"
        return state


@dataclass(frozen=True)
class ScenarioPreset:
    key: str
    label: str
    description: str
    factory: Callable[[Sequence[int]], list[Scenario]]


@dataclass(frozen=True)
class ManualLeakPreset:
    key: str
    label: str
    description: str
    factory: Callable[[int, int], list[Scenario]]


def _primary_segment(segment_ids: Sequence[int]) -> int:
    return segment_ids[len(segment_ids) // 2]


def get_scenario_presets() -> tuple[ScenarioPreset, ...]:
    return (
        ScenarioPreset(
            key="steady_state",
            label="Steady State",
            description="Nominal operating conditions with realistic demand noise and no incident.",
            factory=lambda segment_ids: [],
        ),
        ScenarioPreset(
            key="slow_seep",
            label="Slow Seep",
            description="A realistic leak ramps gradually on one segment before alarms trip.",
            factory=lambda segment_ids: [
                LeakProgressionScenario(affected_segments=[_primary_segment(segment_ids)])
            ],
        ),
        ScenarioPreset(
            key="demand_shock",
            label="Demand Shock",
            description="Demand spike followed by mild pump strain, but no true leak event.",
            factory=lambda segment_ids: [
                DemandSpikeScenario(),
                PumpWearScenario(start_step=18, ramp_steps=8, hold_steps=10, affected_segments=[segment_ids[-1]]),
            ],
        ),
        ScenarioPreset(
            key="compound_incident",
            label="Compound Incident",
            description="Demand stress transitions into pump degradation and then a leak on the trunk line.",
            factory=lambda segment_ids: [
                DemandSpikeScenario(start_step=6, ramp_steps=4, hold_steps=6, recovery_steps=4),
                PumpWearScenario(start_step=14, ramp_steps=8, hold_steps=18, affected_segments=[segment_ids[-1]]),
                LeakProgressionScenario(
                    start_step=20,
                    ramp_steps=10,
                    hold_steps=18,
                    recovery_steps=10,
                    affected_segments=[_primary_segment(segment_ids)],
                ),
            ],
        ),
        ScenarioPreset(
            key="micro_leak",
            label="Micro Leak",
            description="Very subtle leak that barely crosses detection thresholds — tests early detection sensitivity.",
            factory=lambda segment_ids: [
                LeakProgressionScenario(
                    start_step=15,
                    ramp_steps=25,
                    hold_steps=40,
                    recovery_steps=20,
                    max_severity=0.22,
                    residual=0.1,
                    affected_segments=[_primary_segment(segment_ids)],
                ),
            ],
        ),
    )


def build_scenarios(preset_key: str, segment_ids: Sequence[int]) -> list[Scenario]:
    preset_map = {preset.key: preset for preset in get_scenario_presets()}
    if preset_key not in preset_map:
        raise KeyError(f"Unknown scenario preset: {preset_key}")
    return preset_map[preset_key].factory(segment_ids)


def get_manual_leak_presets() -> tuple[ManualLeakPreset, ...]:
    return (
        ManualLeakPreset(
            key="slow_seep_manual",
            label="Slow Seep",
            description="Gradual leak growth with an earlier warning phase before alarms.",
            factory=lambda start_step, segment_id: [
                LeakProgressionScenario(
                    start_step=start_step,
                    ramp_steps=14,
                    hold_steps=18,
                    recovery_steps=8,
                    max_severity=1.0,
                    affected_segments=[segment_id],
                )
            ],
        ),
        ManualLeakPreset(
            key="rupture_manual",
            label="Fast Rupture",
            description="A fast, high-severity leak event that should trigger a rapid score jump.",
            factory=lambda start_step, segment_id: [
                LeakProgressionScenario(
                    start_step=start_step,
                    ramp_steps=4,
                    hold_steps=14,
                    recovery_steps=8,
                    max_severity=1.35,
                    affected_segments=[segment_id],
                )
            ],
        ),
        ManualLeakPreset(
            key="pump_assisted_leak_manual",
            label="Pump-Assisted Leak",
            description="Pump degradation appears first, then transitions into a leak on the same segment.",
            factory=lambda start_step, segment_id: [
                PumpWearScenario(
                    start_step=start_step,
                    ramp_steps=6,
                    hold_steps=12,
                    recovery_steps=6,
                    intensity=1.0,
                    affected_segments=[segment_id],
                ),
                LeakProgressionScenario(
                    start_step=start_step + 5,
                    ramp_steps=10,
                    hold_steps=18,
                    recovery_steps=8,
                    max_severity=1.0,
                    affected_segments=[segment_id],
                ),
            ],
        ),
    )


def build_manual_leak_scenarios(preset_key: str, *, start_step: int, segment_id: int) -> list[Scenario]:
    preset_map = {preset.key: preset for preset in get_manual_leak_presets()}
    if preset_key not in preset_map:
        raise KeyError(f"Unknown manual leak preset: {preset_key}")
    return preset_map[preset_key].factory(start_step, segment_id)


__all__ = [
    "DemandSpikeScenario",
    "LeakProgressionScenario",
    "ManualLeakPreset",
    "PumpWearScenario",
    "ScenarioPreset",
    "build_manual_leak_scenarios",
    "build_scenarios",
    "get_manual_leak_presets",
    "get_scenario_presets",
]
