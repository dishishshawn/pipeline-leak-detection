import numpy as np
import pandas as pd

from scripts.download_petrobras_3w import normalize_event_class, sanitize_sensor_columns
from scripts.train_petrobras_models import engineer_features


def test_normalize_event_class_handles_100_series_labels():
    assert normalize_event_class(0) == 0
    assert normalize_event_class(3) == 3
    assert normalize_event_class(101) == 1
    assert normalize_event_class(109) == 9


def test_sanitize_sensor_columns_replaces_extreme_sentinels_with_nan():
    df = pd.DataFrame(
        {
            "P_inlet": [-1.180116e42, 2.5e7],
            "T_inlet": [-1.711613e38, 45.0],
            "Q_inlet": [0.2, 0.3],
        }
    )

    cleaned = sanitize_sensor_columns(df)

    assert np.isnan(cleaned.loc[0, "P_inlet"])
    assert np.isnan(cleaned.loc[0, "T_inlet"])
    assert cleaned.loc[1, "P_inlet"] == df.loc[1, "P_inlet"]
    assert cleaned.loc[1, "T_inlet"] == df.loc[1, "T_inlet"]


def test_engineer_features_clips_ratio_features_to_finite_values():
    df = pd.DataFrame(
        {
            "scenario_id": ["a", "a", "a"],
            "target": [0, 1, 1],
            "P_inlet": [5.0e7, 5.0e7, 5.0e7],
            "P_mid": [4.5e7, 4.5e7, 4.5e7],
            "P_outlet": [0.0, 1.0e-12, 1.0],
            "Q_inlet": [1.0, 1.0, 1.0],
            "Q_outlet": [0.0, 1.0e-12, 1.0],
            "T_inlet": [45.0, 45.0, 45.0],
            "T_mid": [40.0, 40.0, 40.0],
            "T_outlet": [35.0, 35.0, 35.0],
        }
    )

    features, labels = engineer_features(df)

    assert labels.tolist() == [0, 1, 1]
    assert np.isfinite(features.to_numpy()).all()
    assert float(features["P_ratio_in_out"].abs().max()) <= 1000.0
    assert float(features["Q_ratio"].abs().max()) <= 1000.0
