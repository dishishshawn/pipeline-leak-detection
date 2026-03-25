from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.artifacts import discover_model_artifacts
from src.models.realtime import (
    EnsembleMember,
    FeatureSubsetModel,
    IsolationForestLeakDetector,
    WeightedLeakEnsemble,
)


def _sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(40, 4)),
        columns=["pressure", "flow_rate", "temperature", "pump_speed"],
    )


def test_isolation_forest_leak_detector_outputs_binary_probabilities():
    X = _sample_frame()
    y = pd.Series([0] * 30 + [1] * 10)

    model = IsolationForestLeakDetector(random_state=42)
    model.fit(X, y)

    probabilities = model.predict_proba(X.iloc[:5])
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


class ConstantProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability

    def predict_proba(self, X):
        positive = np.full(len(X), self.positive_probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_weighted_leak_ensemble_combines_member_probabilities():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    ensemble = WeightedLeakEnsemble(
        members=[
            EnsembleMember("low", ConstantProbabilityModel(0.2), 1.0),
            EnsembleMember("high", ConstantProbabilityModel(0.8), 3.0),
        ]
    )

    probabilities = ensemble.predict_proba(X)
    expected_positive = np.full(len(X), 0.65)
    np.testing.assert_allclose(probabilities[:, 1], expected_positive, atol=1e-6)
    np.testing.assert_array_equal(ensemble.predict(X), np.array([1, 1, 1]))


def test_feature_subset_model_reindexes_columns_and_drops_extras():
    base = ConstantProbabilityModel(0.7)
    wrapped = FeatureSubsetModel(base, ["flow_rate", "pressure"])
    X = pd.DataFrame(
        {
            "pressure": [1.0, 2.0],
            "temperature": [9.0, 9.0],
            "flow_rate": [3.0, 4.0],
        }
    )

    probabilities = wrapped.predict_proba(X)

    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities[:, 1], np.array([0.7, 0.7]), atol=1e-6)


def test_discover_model_artifacts_includes_realtime_models(tmp_path):
    models_root = tmp_path / "models"
    realtime_dir = models_root / "realtime"
    realtime_dir.mkdir(parents=True)

    joblib.dump(ConstantProbabilityModel(0.7), realtime_dir / "scada_pipeline_realtime_hybrid_ensemble.joblib")

    artifacts = discover_model_artifacts(models_root, "scada")

    assert "Realtime Hybrid Ensemble" in artifacts
    assert Path(artifacts["Realtime Hybrid Ensemble"].path).parent.name == "realtime"
