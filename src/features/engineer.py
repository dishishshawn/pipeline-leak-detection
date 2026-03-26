import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_pressure_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pressure change between consecutive rows within each segment.
    """
    df = df.copy()

    df = df.sort_values(["segment_id", "timestamp"])
    df["pressure_delta"] = (
        df.groupby("segment_id")["pressure"].diff().fillna(0)
    )

    return df


def add_flow_rate_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add flow rate change between consecutive rows within each segment.
    """
    df = df.copy()

    df = df.sort_values(["segment_id", "timestamp"])
    df["flow_rate_delta"] = (
        df.groupby("segment_id")["flow_rate"].diff().fillna(0)
    )

    return df


def add_relative_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percent-change style features that are less sensitive to dataset scale.
    """
    df = df.copy()
    df = df.sort_values(["segment_id", "timestamp"])

    prev_pressure = df.groupby("segment_id")["pressure"].shift(1).replace(0, np.nan)
    prev_flow = df.groupby("segment_id")["flow_rate"].shift(1).replace(0, np.nan)

    df["pressure_pct_delta"] = (df["pressure_delta"] / prev_pressure.abs()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["flow_pct_delta"] = (df["flow_rate_delta"] / prev_flow.abs()).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    safe_flow = df["flow_rate"].replace(0, np.nan)
    df["pressure_flow_ratio"] = (df["pressure"] / safe_flow).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def add_rolling_pressure_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling mean and rolling std for pressure within each segment.
    """
    df = df.copy()
    df = df.sort_values(["segment_id", "timestamp"])

    df["pressure_roll_mean"] = (
        df.groupby("segment_id")["pressure"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )

    df["pressure_roll_std"] = (
        df.groupby("segment_id")["pressure"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).std())
        .fillna(0)
    )

    return df


def add_rolling_flow_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling mean and rolling std for flow rate within each segment.
    """
    df = df.copy()
    df = df.sort_values(["segment_id", "timestamp"])

    df["flow_roll_mean"] = (
        df.groupby("segment_id")["flow_rate"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )

    df["flow_roll_std"] = (
        df.groupby("segment_id")["flow_rate"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).std())
        .fillna(0)
    )

    return df


def add_rolling_zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling z-scores for pressure and flow relative to recent history.
    """
    df = df.copy()
    pressure_denom = df["pressure_roll_std"].replace(0, np.nan)
    flow_denom = df["flow_roll_std"].replace(0, np.nan)

    df["pressure_roll_z"] = ((df["pressure"] - df["pressure_roll_mean"]) / pressure_denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["flow_roll_z"] = ((df["flow_rate"] - df["flow_roll_mean"]) / flow_denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def encode_event_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert event_type to numeric codes.
    """
    df = df.copy()

    if "event_type" in df.columns:
        df["event_type_encoded"] = df["event_type"].astype("category").cat.codes

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    """
    df = df.copy()

    df = add_pressure_delta(df)
    df = add_flow_rate_delta(df)
    df = add_relative_change_features(df)
    df = add_rolling_pressure_features(df, window=5)
    df = add_rolling_flow_features(df, window=5)
    df = add_rolling_zscore_features(df)
    df = encode_event_type(df)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


if __name__ == "__main__":
    sample_data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01 00:00",
            "2024-01-01 00:01",
            "2024-01-01 00:02",
            "2024-01-01 00:03",
        ]),
        "segment_id": [1, 1, 1, 1],
        "pressure": [71.6, 71.2, 70.8, 70.1],
        "flow_rate": [3.1, 3.0, 2.9, 2.5],
        "event_type": ["normal", "normal", "warning", "fault"],
    })

    featured_df = build_features(sample_data)
    print(featured_df)
