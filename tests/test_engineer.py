import pandas as pd
from src.features.engineer import (
    add_pressure_delta,
    add_flow_rate_delta,
    add_rolling_pressure_features,
    build_features,
)


def _make_df(n=6):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
        "segment_id": [1] * n,
        "pressure": [70.0, 69.5, 69.0, 68.0, 68.5, 67.0],
        "flow_rate": [3.0, 2.9, 2.8, 2.7, 2.6, 2.5],
        "event_type": ["normal"] * n,
    })


def test_pressure_delta_first_row_is_zero():
    df = add_pressure_delta(_make_df())
    assert df["pressure_delta"].iloc[0] == 0.0


def test_pressure_delta_correct_diff():
    df = add_pressure_delta(_make_df())
    assert abs(df["pressure_delta"].iloc[1] - (-0.5)) < 1e-6


def test_flow_rate_delta_column_created():
    df = add_flow_rate_delta(_make_df())
    assert "flow_rate_delta" in df.columns


def test_rolling_pressure_features_columns_created():
    df = add_rolling_pressure_features(_make_df())
    assert "pressure_roll_mean" in df.columns
    assert "pressure_roll_std" in df.columns


def test_build_features_adds_all_columns():
    df = build_features(_make_df())
    for col in ["pressure_delta", "flow_rate_delta", "pressure_roll_mean",
                "flow_roll_mean", "event_type_encoded"]:
        assert col in df.columns, f"Missing: {col}"


def test_build_features_preserves_row_count():
    df_in = _make_df()
    df_out = build_features(df_in)
    assert len(df_out) == len(df_in)
