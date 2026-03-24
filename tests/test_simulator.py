from datetime import datetime, timedelta

from src.simulation import (
    PipelineTelemetrySimulator,
    SimulationConfig,
    build_scenarios,
    make_default_profiles,
)


REQUIRED_COLUMNS = {
    "timestamp",
    "segment_id",
    "pressure",
    "flow_rate",
    "temperature",
    "valve_status",
    "pump_state",
    "pump_speed",
    "compressor_state",
    "energy_consumption",
    "alarm_triggered",
    "event_type",
    "target",
    "scenario_context",
}


def _make_simulator(preset_key: str = "steady_state", segment_count: int = 3):
    profiles = make_default_profiles(segment_count)
    config = SimulationConfig(
        segment_profiles=profiles,
        start_time=datetime(2026, 1, 1, 6, 0, 0),
        tick_seconds=1,
        step_minutes=1,
        history_limit=60,
        seed=7,
    )
    scenarios = build_scenarios(preset_key, [profile.segment_id for profile in profiles])
    return PipelineTelemetrySimulator(config=config, scenarios=scenarios)


def test_default_profiles_have_unique_segment_ids():
    profiles = make_default_profiles(4)
    assert len(profiles) == 4
    assert len({profile.segment_id for profile in profiles}) == 4


def test_simulator_emits_required_schema():
    simulator = _make_simulator()
    now = datetime(2026, 1, 1, 12, 0, 0)
    simulator.start(wall_clock=now)

    batch = simulator.advance(wall_clock=now)

    assert len(batch) == 3
    assert REQUIRED_COLUMNS.issubset(batch.columns)
    assert set(batch["event_type"]) == {"normal"}
    assert batch["target"].sum() == 0


def test_simulator_advances_on_wall_clock_ticks():
    simulator = _make_simulator(segment_count=2)
    now = datetime(2026, 1, 1, 12, 0, 0)
    simulator.start(wall_clock=now)

    first = simulator.advance(wall_clock=now)
    second = simulator.advance(wall_clock=now + timedelta(seconds=1))

    assert len(first) == 2
    assert len(second) == 2
    assert second["timestamp"].min() > first["timestamp"].min()


def test_slow_seep_preset_eventually_produces_leak_rows():
    simulator = _make_simulator(preset_key="slow_seep", segment_count=3)
    now = datetime(2026, 1, 1, 12, 0, 0)
    simulator.start(wall_clock=now)

    for step in range(35):
        simulator.advance(wall_clock=now + timedelta(seconds=step))

    history = simulator.snapshot()

    assert not history.empty
    assert (history["target"] == 1).any()
    assert (history["alarm_triggered"] == 1).any()
    assert history["scenario_context"].str.contains("leak_progression").any()


def test_unknown_preset_raises_key_error():
    try:
        build_scenarios("does_not_exist", [1, 2, 3])
    except KeyError:
        return
    assert False, "Expected build_scenarios to raise KeyError for an unknown preset"
