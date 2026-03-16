import pytest
import pandas as pd
from src.data.loader import load_and_prepare, validate_schema, basic_cleaning

SAMPLE_CSV = "data/sample/scada_sample.csv"


def test_load_and_prepare_returns_dataframe():
    df = load_and_prepare(SAMPLE_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_and_prepare_has_required_columns():
    df = load_and_prepare(SAMPLE_CSV)
    expected = ["timestamp", "pressure", "flow_rate", "temperature", "target"]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


def test_validate_schema_raises_on_missing_column():
    df = pd.DataFrame({"timestamp": [], "pressure": []})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_basic_cleaning_removes_duplicates():
    row = {
        "timestamp": "2024-01-01",
        "segment_id": 1,
        "pressure": 70.0,
        "flow_rate": 3.0,
        "temperature": 25.0,
        "valve_status": 1,
        "pump_state": 1,
        "pump_speed": 1200.0,
        "compressor_state": 1,
        "energy_consumption": 100.0,
        "alarm_triggered": 0,
        "event_type": "normal",
        "target": 0,
    }
    df = pd.DataFrame([row, row])
    cleaned = basic_cleaning(df)
    assert len(cleaned) == 1


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_and_prepare("data/does_not_exist.csv")
