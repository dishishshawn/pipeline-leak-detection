from __future__ import annotations

import pandas as pd

from src.synthetic.generator import SyntheticHistoryConfig, generate_synthetic_histories
from src.synthetic.scenario_engine import build_event_plan, generate_regime_schedule


def test_regime_schedule_covers_full_horizon():
    schedule = generate_regime_schedule(96, step_minutes=15, rng=__import__("numpy").random.default_rng(7))
    assert len(schedule) == 96
    assert set(schedule).issubset({"shutdown", "startup", "low_demand", "nominal", "high_demand", "maintenance_bypass"})


def test_build_event_plan_contains_leaks_and_disturbances():
    rng = __import__("numpy").random.default_rng(11)
    events = build_event_plan(
        history_id="history_000",
        n_steps=24 * 12,
        step_minutes=5,
        segment_ids=[1, 2, 3],
        rng=rng,
    )
    event_types = {event.event_type for event in events}
    assert {"slow_leak", "borderline_leak", "abrupt_leak"}.issubset(event_types)
    assert {"demand_spike", "pump_wear", "valve_transient", "sensor_drift"}.issubset(event_types)


def test_generate_synthetic_histories_preserves_training_schema(monkeypatch):
    def _fake_signature(*, duration_steps: int, intensity: float, seed: int, onset_type: str = "ramp"):
        x = pd.Series(range(duration_steps), dtype=float) / max(duration_steps - 1, 1)
        return pd.DataFrame(
            {
                "pressure_effect": -0.8 * x * intensity,
                "flow_effect": -0.4 * x * intensity,
                "temperature_effect": 0.1 * x * intensity,
                "severity": (x * intensity).clip(upper=1.0),
            }
        )

    monkeypatch.setattr("src.synthetic.generator.sample_leak_signature", _fake_signature)

    config = SyntheticHistoryConfig(
        n_histories=2,
        days_per_history=1,
        segment_count=3,
        step_minutes=15,
        seed=9,
    )
    df, metadata = generate_synthetic_histories(config)

    required = {
        "timestamp",
        "history_id",
        "scenario_id",
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
        "leak_severity",
        "pump_efficiency",
    }
    assert required.issubset(df.columns)
    assert df["history_id"].nunique() == 2
    assert (df.groupby("history_id")["timestamp"].max() > df.groupby("history_id")["timestamp"].min()).all()
    assert df["target"].sum() > 0
    assert metadata["event_type"].nunique() >= 4
