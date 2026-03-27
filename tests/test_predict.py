import pytest
import numpy as np
import pandas as pd
from src.models.predict import (
    load_model,
    predict,
    predict_proba,
    predict_leak_score,
    supports_leak_score,
    supports_probability_scores,
)
from src.models.physics_wrapper import PhysicsModelWrapper


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


class IdentityScaler:
    def __init__(self):
        self.seen = None

    def transform(self, df):
        self.seen = df.copy()
        return df.to_numpy() + 1.0


class ThresholdClassifier:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return (np.asarray(X)[:, 0] > 1.5).astype(int)


class DecisionFunctionClassifier(ThresholdClassifier):
    def decision_function(self, X):
        return np.asarray(X)[:, 0] - 1.5


class ScoreSamplesClassifier(ThresholdClassifier):
    def score_samples(self, X):
        return np.asarray(X)[:, 0]


class PredictOnlyClassifier:
    def predict(self, X):
        return np.array([0, 1, 1])


class CaptureShapeClassifier:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.last_shape = None

    def predict(self, X):
        self.last_shape = np.asarray(X).shape
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        self.last_shape = np.asarray(X).shape
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class NegativeOneAnomalyClassifier:
    def predict(self, X):
        return np.array([-1, 1, -1])


def test_predict_handles_legacy_tuple_model():
    model = (IdentityScaler(), ThresholdClassifier())
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    preds = predict(model, X)
    np.testing.assert_array_equal(preds, np.array([0, 1, 1]))


class NamedThresholdClassifier(ThresholdClassifier):
    feature_names_in_ = np.array(["a", "b"])


def test_predict_reindexes_dataframe_to_model_feature_schema():
    model = NamedThresholdClassifier()
    X = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0],
            "b": [0.0, 0.0, 0.0],
            "pressure_pct_delta": [9.0, 9.0, 9.0],
            "flow_roll_z": [8.0, 8.0, 8.0],
        }
    )
    preds = predict(model, X)
    np.testing.assert_array_equal(preds, np.array([0, 0, 1]))


def test_predict_proba_uses_decision_function_when_available():
    model = DecisionFunctionClassifier()
    X = pd.DataFrame({"a": [0.0, 1.5, 3.0]})
    proba = predict_proba(model, X)
    assert proba.shape == (3, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert proba[0, 1] < 0.5
    assert proba[2, 1] > 0.5


def test_predict_leak_score_uses_score_samples_for_anomaly_models():
    model = ScoreSamplesClassifier()
    X = pd.DataFrame({"a": [3.0, 2.0, 1.0]})
    scores = predict_leak_score(model, X)
    assert scores.shape == (3,)
    assert scores[0] < scores[-1]
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_predict_proba_falls_back_to_hard_predictions():
    model = PredictOnlyClassifier()
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    proba = predict_proba(model, X)
    expected = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(proba, expected)


def test_physics_wrapper_emits_full_training_feature_schema_from_live_scada():
    inner = CaptureShapeClassifier()
    wrapper = PhysicsModelWrapper(inner)
    X = pd.DataFrame(
        {
            "pressure": [70.0, 69.8, 69.5],
            "flow_rate": [3.1, 3.0, 2.9],
            "temperature": [24.0, 24.1, 24.2],
        }
    )

    preds = predict(wrapper, X)

    assert len(preds) == len(X)
    assert inner.last_shape == (3, 27)


def test_physics_wrapper_maps_isolation_style_predictions_to_binary_labels():
    wrapper = PhysicsModelWrapper(NegativeOneAnomalyClassifier())
    X = pd.DataFrame(
        {
            "pressure": [70.0, 69.8, 69.5],
            "flow_rate": [3.1, 3.0, 2.9],
            "temperature": [24.0, 24.1, 24.2],
        }
    )

    preds = predict(wrapper, X)

    np.testing.assert_array_equal(preds, np.array([1, 0, 1]))


def test_support_helpers_distinguish_native_scores_from_fallbacks():
    assert supports_probability_scores((IdentityScaler(), DecisionFunctionClassifier()))
    assert not supports_probability_scores(PredictOnlyClassifier())
    assert supports_leak_score(PredictOnlyClassifier())
