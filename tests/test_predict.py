import pytest
import numpy as np
import pandas as pd
from src.models.predict import load_model, predict, predict_proba, predict_leak_score


LR_PATH = "models/logistic_regression.joblib"
RF_PATH = "models/random_forest.joblib"

FEATURE_COLS = [
    "segment_id", "pressure", "flow_rate", "temperature",
    "valve_status", "pump_state", "pump_speed", "compressor_state",
    "energy_consumption", "pressure_delta", "flow_rate_delta",
    "pressure_roll_mean", "pressure_roll_std",
    "flow_roll_mean", "flow_roll_std", "event_type_encoded",
]


def _sample_X(n=10):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random((n, len(FEATURE_COLS))), columns=FEATURE_COLS)


def test_load_model_logistic_regression():
    model = load_model(LR_PATH)
    assert model is not None


def test_load_model_random_forest():
    model = load_model(RF_PATH)
    assert model is not None


def test_load_model_missing_file():
    with pytest.raises(FileNotFoundError):
        load_model("models/nonexistent.joblib")


def test_predict_returns_array():
    model = load_model(LR_PATH)
    X = _sample_X()
    preds = predict(model, X)
    assert len(preds) == len(X)


def test_predict_proba_shape():
    model = load_model(RF_PATH)
    X = _sample_X()
    proba = predict_proba(model, X)
    assert proba.shape == (len(X), 2)


def test_predict_proba_sums_to_one():
    model = load_model(RF_PATH)
    X = _sample_X()
    proba = predict_proba(model, X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_leak_score_between_zero_and_one():
    model = load_model(RF_PATH)
    X = _sample_X()
    scores = predict_leak_score(model, X)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
