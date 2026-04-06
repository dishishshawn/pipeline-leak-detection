from pathlib import Path

import joblib
import pandas as pd

from src.data.dataset_adapters import apply_column_map
from src.data.robust_corpus import (
    load_dataset_for_live_training,
    load_scada_like_csv,
    rebalance_binary_frame,
    to_scada_training_frame,
)
from src.models.artifacts import discover_model_artifacts
from scripts.build_robust_training_data import DEFAULT_DATASETS


class ConstantModel:
    def predict(self, X):
        return [0] * len(X)


def test_apply_column_map_accepts_alias_lists():
    df = pd.DataFrame(
        {
            "Pressure (bar)": [1.0, 2.0],
            "Flow Rate (L/s)": [3.0, 4.0],
        }
    )

    mapped = apply_column_map(
        df.copy(),
        {
            "pressure": ["pressure", "Pressure (bar)"],
            "flow_rate": ["flow_rate", "Flow Rate (L/s)"],
        },
    )

    assert "pressure" in mapped.columns
    assert "flow_rate" in mapped.columns


def test_to_scada_training_frame_builds_required_schema():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:01:00"]),
            "sensor_id": ["A", "A"],
            "pressure": [4.2, 4.1],
            "flow_rate": [12.0, 11.8],
            "temperature": [21.0, 21.1],
            "leak_label": [0, 1],
        }
    )

    frame = to_scada_training_frame(df, "water_leak")

    assert set(frame.columns).issuperset(
        {
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
            "source_dataset",
        }
    )
    assert frame["target"].tolist() == [0, 1]


def test_rebalance_binary_frame_caps_negative_ratio():
    df = pd.DataFrame(
        {
            "pressure": range(20),
            "flow_rate": range(20),
            "target": [1] * 4 + [0] * 16,
        }
    )

    rebalanced = rebalance_binary_frame(df, max_negative_ratio=2.0, random_state=42)

    positives = int((rebalanced["target"] == 1).sum())
    negatives = int((rebalanced["target"] == 0).sum())
    assert positives == 4
    assert negatives <= 8


def test_load_scada_like_csv_backfills_missing_scada_columns(tmp_path):
    csv_path = tmp_path / "petrobras.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:01:00"],
            "scenario_id": ["well-a", "well-a"],
            "pressure": [10.0, 10.2],
            "flow_rate": [5.0, 5.1],
            "temperature": [30.0, 30.2],
            "target": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    frame = load_scada_like_csv(csv_path, "petrobras_3w")

    assert set(frame.columns).issuperset(
        {
            "segment_id",
            "valve_status",
            "pump_state",
            "pump_speed",
            "compressor_state",
            "energy_consumption",
            "alarm_triggered",
            "event_type",
        }
    )
    assert frame["segment_id"].nunique() == 1
    assert frame["event_type"].tolist() == ["normal", "leak"]


def test_load_dataset_for_live_training_prefers_processed_petrobras_csv(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed" / "petrobras_3w"
    processed_dir.mkdir(parents=True)
    processed_csv = processed_dir / "petrobras_3w_scada.csv"
    processed_csv.write_text("timestamp,segment_id,pressure,flow_rate,temperature,target\n")

    sentinel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00:00"]),
            "segment_id": [1],
            "pressure": [10.0],
            "flow_rate": [5.0],
            "temperature": [30.0],
            "target": [1],
            "source_dataset": ["petrobras_3w"],
        }
    )

    monkeypatch.setattr(
        "src.data.robust_corpus.get_dataset",
        lambda name: {
            "telemetry_compatible": True,
            "live_training_eligible": True,
            "local_processed": str(processed_dir),
        },
    )
    monkeypatch.setattr("src.data.robust_corpus.load_scada_like_csv", lambda path, source_name: sentinel)
    monkeypatch.setattr(
        "src.data.robust_corpus.normalize",
        lambda name, nrows=None: (_ for _ in ()).throw(AssertionError("normalize should not be called")),
    )

    frame = load_dataset_for_live_training("petrobras_3w")

    assert frame.equals(sentinel)


def test_robust_corpus_defaults_use_petrobras_instead_of_water_leak():
    assert "petrobras_3w" in DEFAULT_DATASETS
    assert "water_leak" not in DEFAULT_DATASETS


def test_discover_model_artifacts_prefers_robust_models(tmp_path):
    models_root = tmp_path / "models"
    robust_dir = models_root / "robust"
    realtime_dir = models_root / "realtime"
    robust_dir.mkdir(parents=True)
    realtime_dir.mkdir(parents=True)

    joblib.dump(ConstantModel(), realtime_dir / "scada_pipeline_realtime_random_forest.joblib")
    joblib.dump(ConstantModel(), robust_dir / "robust_realtime_realtime_random_forest.joblib")

    artifacts = discover_model_artifacts(models_root, "scada")

    assert Path(artifacts["Robust Random Forest"].path).parent.name == "robust"


def test_discover_model_artifacts_skips_physics_scaler_files(tmp_path):
    models_root = tmp_path / "models"
    physics_dir = models_root / "physics_sim"
    physics_dir.mkdir(parents=True)

    joblib.dump(ConstantModel(), physics_dir / "physics_sim_random_forest.joblib")
    joblib.dump(ConstantModel(), physics_dir / "physics_sim_logistic_regression_scaler.joblib")

    artifacts = discover_model_artifacts(models_root, "scada")

    assert "Physics Sim Logistic Regression Scaler" not in artifacts
    assert "Physics Sim Random Forest" in artifacts
