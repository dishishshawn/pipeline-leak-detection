from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_synthetic_data import compute_validation_summary, infer_history_id


def test_infer_history_id_detects_timestamp_resets_per_segment():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
            ],
            "segment_id": [1, 1, 1, 1],
            "pressure": [10, 11, 10, 11],
            "flow_rate": [2, 2.1, 2, 2.1],
            "temperature": [30, 30, 30, 30],
            "valve_status": [1, 1, 1, 1],
            "pump_state": [1, 1, 1, 1],
            "pump_speed": [100, 100, 100, 100],
            "compressor_state": [1, 1, 1, 1],
            "energy_consumption": [50, 50, 50, 50],
            "alarm_triggered": [0, 0, 0, 0],
            "event_type": ["normal", "normal", "normal", "normal"],
            "target": [0, 0, 0, 0],
        }
    )

    histories = infer_history_id(df)

    assert histories.nunique() == 2


def test_validation_summary_flags_short_easy_dataset():
    rows = []
    for run in range(6):
        for step in range(6):
            leak = int(run % 2 == 1 and step >= 3)
            rows.append(
                {
                    "timestamp": f"2026-01-01 00:0{step}:00",
                    "segment_id": 1,
                    "pressure": 70 - 8 * leak - 0.1 * step,
                    "flow_rate": 3 - 0.7 * leak,
                    "temperature": 25 + 0.2 * leak,
                    "valve_status": 1,
                    "pump_state": 1,
                    "pump_speed": 1200 + 5 * step,
                    "compressor_state": 1,
                    "energy_consumption": 55 + 1.5 * step,
                    "alarm_triggered": leak,
                    "event_type": "warning" if leak else "normal",
                    "target": leak,
                    "scenario_context": "leak_progression" if leak else "nominal",
                    "leak_severity": 0.2 if leak else 0.0,
                    "history_id": f"run_{run}",
                }
            )

    df = pd.DataFrame(rows)
    summary = compute_validation_summary(df)

    assert summary.median_history_hours < 24
    assert any("Raw sensors alone separate leaks too easily" in warning for warning in summary.warnings)
