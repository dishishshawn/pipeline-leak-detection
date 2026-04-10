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


def add_pressure_drop_consistency(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Deviation of pressure delta from its rolling mean — persistent negative = leak."""
    df = df.copy()
    df = df.sort_values(["segment_id", "timestamp"])
    if "pressure_delta" not in df.columns:
        return df
    roll_mean = (
        df.groupby("segment_id")["pressure_delta"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    df["pressure_delta_deviation"] = df["pressure_delta"] - roll_mean
    return df


def add_flow_imbalance_persistence(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling count of consecutive negative flow deltas — sustained drop = leak."""
    df = df.copy()
    df = df.sort_values(["segment_id", "timestamp"])
    if "flow_rate_delta" not in df.columns:
        return df
    negative_flag = (df["flow_rate_delta"] < -0.01).astype(int)
    df["flow_neg_streak"] = (
        negative_flag.groupby(df["segment_id"])
        .transform(lambda s: s.rolling(window=window, min_periods=1).sum())
    )
    return df


def add_segment_relative_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Segment pressure/flow vs cross-segment mean at each timestamp — leak affects one segment more."""
    df = df.copy()
    if "timestamp" not in df.columns:
        return df
    ts_mean_pressure = df.groupby("timestamp")["pressure"].transform("mean")
    df["pressure_segment_deviation"] = df["pressure"] - ts_mean_pressure
    ts_mean_flow = df.groupby("timestamp")["flow_rate"].transform("mean")
    df["flow_segment_deviation"] = df["flow_rate"] - ts_mean_flow
    return df


def add_change_over_baseline(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Deviation from long-window baseline — captures slow drift from normal."""
    df = df.copy()
    df = df.sort_values(["segment_id", "timestamp"])
    for col, out_col in [("pressure", "pressure_baseline_dev"), ("flow_rate", "flow_baseline_dev")]:
        if col not in df.columns:
            continue
        baseline = (
            df.groupby("segment_id")[col]
            .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        )
        df[out_col] = df[col] - baseline
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering pipeline — optimised single-pass implementation.

    Compared with chaining the individual add_* helpers this avoids:
    - 11 redundant df.copy() calls
    - 9 redundant sort_values() calls
    - slow groupby().transform(lambda s: s.rolling()...) per-group Python lambdas,
      replaced by the native groupby().rolling() path
    All intermediate groupby objects reuse the same sorted DataFrame.
    """
    df = df.copy()
    df.sort_values(["segment_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    gb = df.groupby("segment_id", sort=False)

    # -- Diffs ----------------------------------------------------------------
    df["pressure_delta"] = gb["pressure"].diff().fillna(0.0)
    df["flow_rate_delta"] = gb["flow_rate"].diff().fillna(0.0)

    # -- Relative change features ---------------------------------------------
    prev_pressure = gb["pressure"].shift(1).replace(0, np.nan)
    prev_flow = gb["flow_rate"].shift(1).replace(0, np.nan)
    df["pressure_pct_delta"] = (
        (df["pressure_delta"] / prev_pressure.abs())
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    df["flow_pct_delta"] = (
        (df["flow_rate_delta"] / prev_flow.abs())
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    safe_flow = df["flow_rate"].replace(0, np.nan)
    df["pressure_flow_ratio"] = (
        (df["pressure"] / safe_flow)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # -- Window-5 rolling stats (native groupby.rolling — no Python lambda) --
    roll5_p = gb["pressure"].rolling(5, min_periods=1)
    df["pressure_roll_mean"] = roll5_p.mean().reset_index(level=0, drop=True)
    df["pressure_roll_std"] = roll5_p.std().reset_index(level=0, drop=True).fillna(0.0)

    roll5_f = gb["flow_rate"].rolling(5, min_periods=1)
    df["flow_roll_mean"] = roll5_f.mean().reset_index(level=0, drop=True)
    df["flow_roll_std"] = roll5_f.std().reset_index(level=0, drop=True).fillna(0.0)

    # -- Z-scores (inline, no copy) ------------------------------------------
    p_denom = df["pressure_roll_std"].replace(0, np.nan)
    f_denom = df["flow_roll_std"].replace(0, np.nan)
    df["pressure_roll_z"] = (
        ((df["pressure"] - df["pressure_roll_mean"]) / p_denom)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    df["flow_roll_z"] = (
        ((df["flow_rate"] - df["flow_roll_mean"]) / f_denom)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # -- Encode event type ---------------------------------------------------
    if "event_type" in df.columns:
        df["event_type_encoded"] = df["event_type"].astype("category").cat.codes

    # -- Pressure delta deviation (window=5) ---------------------------------
    pd_roll_mean = (
        gb["pressure_delta"].rolling(5, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df["pressure_delta_deviation"] = df["pressure_delta"] - pd_roll_mean

    # -- Flow neg streak (window=10) -----------------------------------------
    neg_flag = (df["flow_rate_delta"] < -0.01).astype(float)
    df["flow_neg_streak"] = (
        neg_flag.groupby(df["segment_id"])
        .rolling(10, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # -- Segment deviation (cross-segment at each timestamp) -----------------
    if "timestamp" in df.columns:
        ts_mean_p = df.groupby("timestamp")["pressure"].transform("mean")
        df["pressure_segment_deviation"] = df["pressure"] - ts_mean_p
        ts_mean_f = df.groupby("timestamp")["flow_rate"].transform("mean")
        df["flow_segment_deviation"] = df["flow_rate"] - ts_mean_f

    # -- Long-window baseline deviation (window=20) --------------------------
    roll20_p = gb["pressure"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    df["pressure_baseline_dev"] = df["pressure"] - roll20_p
    roll20_f = gb["flow_rate"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    df["flow_baseline_dev"] = df["flow_rate"] - roll20_f

    # -- Micro-leak detection features --------------------------------------
    # Long-window (30-step) rolling stats capture slow persistent drift that
    # short windows miss.  Micro-leaks change pressure/flow by only ~3-5%
    # over many steps — a 5-step window cannot accumulate enough signal.
    roll30_p = gb["pressure"].rolling(30, min_periods=1)
    roll30_f = gb["flow_rate"].rolling(30, min_periods=1)
    pressure_roll30_mean = roll30_p.mean().reset_index(level=0, drop=True)
    flow_roll30_mean = roll30_f.mean().reset_index(level=0, drop=True)
    df["pressure_roll30_std"] = roll30_p.std().reset_index(level=0, drop=True).fillna(0.0)
    df["flow_roll30_std"] = roll30_f.std().reset_index(level=0, drop=True).fillna(0.0)

    # CUSUM-style cumulative pressure drop: sum of negative pressure deltas
    # over a 30-step window.  A micro-leak accumulates small negatives;
    # normal operation has symmetric noise that cancels out.
    neg_p_delta = df["pressure_delta"].clip(upper=0.0)
    df["pressure_cusum_neg30"] = (
        neg_p_delta.groupby(df["segment_id"])
        .rolling(30, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # Pressure-flow divergence: pressure drops but flow stays stable = leak.
    # Normalised difference between pressure trend and flow trend relative
    # to their respective long-window baselines.
    p_dev_norm = (df["pressure"] - pressure_roll30_mean) / pressure_roll30_mean.abs().replace(0, np.nan)
    f_dev_norm = (df["flow_rate"] - flow_roll30_mean) / flow_roll30_mean.abs().replace(0, np.nan)
    df["pressure_flow_divergence"] = (
        (p_dev_norm.fillna(0.0) - f_dev_norm.fillna(0.0))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Sustained negative pressure streak over 15 steps — a run of small
    # drops is the fingerprint of a micro-leak.
    neg_p_flag = (df["pressure_delta"] < -0.05).astype(float)
    df["pressure_neg_streak15"] = (
        neg_p_flag.groupby(df["segment_id"])
        .rolling(15, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # -- EMA-based drift detector (span=20) -----------------------------------
    # Exponential moving average responds faster to persistent drift than
    # simple rolling mean.  The gap between EMA and the long baseline
    # amplifies micro-leak signals while suppressing symmetric noise.
    ema_p = gb["pressure"].transform(lambda s: s.ewm(span=20, min_periods=1).mean())
    df["pressure_ema_dev"] = df["pressure"] - ema_p
    ema_f = gb["flow_rate"].transform(lambda s: s.ewm(span=20, min_periods=1).mean())
    df["flow_ema_dev"] = df["flow_rate"] - ema_f

    # -- Pressure acceleration (second derivative) ----------------------------
    # Micro-leaks produce consistently negative first derivative (delta).
    # The second derivative being near zero while delta is negative
    # distinguishes a sustained drift from a one-off spike.
    df["pressure_accel"] = gb["pressure_delta"].diff().fillna(0.0)

    # -- Ultra-long-window baseline (90 steps) --------------------------------
    # A 90-step window adapts very slowly, so even after 40+ steps of a
    # micro-leak the baseline still remembers pre-leak levels.  This keeps
    # the deviation signal alive during extended leak hold periods where the
    # 20-step baseline has already adapted.
    roll90_p = gb["pressure"].rolling(90, min_periods=1).mean().reset_index(level=0, drop=True)
    roll90_f = gb["flow_rate"].rolling(90, min_periods=1).mean().reset_index(level=0, drop=True)
    df["pressure_baseline60_dev"] = df["pressure"] - roll90_p
    df["flow_baseline60_dev"] = df["flow_rate"] - roll90_f

    # -- Expanding cumulative stats -------------------------------------------
    # Unlike rolling windows, expanding stats never forget the start of the
    # run.  The deviation from the expanding mean grows monotonically during
    # a sustained leak, providing signal that survives indefinitely.
    exp_mean_p = gb["pressure"].expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    exp_mean_f = gb["flow_rate"].expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    df["pressure_expanding_dev"] = df["pressure"] - exp_mean_p
    df["flow_expanding_dev"] = df["flow_rate"] - exp_mean_f

    # -- Cumulative flow deficit (30-step window) -----------------------------
    # Same idea as pressure CUSUM but for flow: a micro-leak drains flow
    # persistently.  Sum of negative flow deltas accumulates signal.
    neg_f_delta = df["flow_rate_delta"].clip(upper=0.0)
    df["flow_cusum_neg30"] = (
        neg_f_delta.groupby(df["segment_id"])
        .rolling(30, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # -- Pressure-to-baseline ratio -------------------------------------------
    # A normalised version of baseline deviation: how far below baseline as
    # a fraction of baseline.  More informative than absolute deviation for
    # different operating points.
    safe_baseline_p = roll20_p.replace(0, np.nan)
    df["pressure_baseline_pct"] = (
        ((df["pressure"] - roll20_p) / safe_baseline_p.abs())
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

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
