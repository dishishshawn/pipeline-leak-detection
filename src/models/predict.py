from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.physics_wrapper import PhysicsModelWrapper


def _patch_legacy_model(model):
    if type(model).__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return model


def _split_model(model):
    if isinstance(model, tuple) and len(model) == 2:
        scaler, estimator = model
        return scaler, _patch_legacy_model(estimator)
    return None, _patch_legacy_model(model)


def _expected_feature_columns_from_parts(scaler, estimator) -> tuple[str, ...] | None:
    feature_columns = getattr(estimator, "feature_columns", None)
    if feature_columns is not None:
        return tuple(str(name) for name in feature_columns)

    names_in = getattr(estimator, "feature_names_in_", None)
    if names_in is not None:
        return tuple(str(name) for name in names_in)

    inner_model = getattr(estimator, "model", None)
    if inner_model is not None:
        inner_names = getattr(inner_model, "feature_names_in_", None)
        if inner_names is not None:
            return tuple(str(name) for name in inner_names)

    if scaler is not None:
        scaler_names = getattr(scaler, "feature_names_in_", None)
        if scaler_names is not None:
            return tuple(str(name) for name in scaler_names)

    return None


def _align_feature_frame(scaler, estimator, df):
    if not isinstance(df, pd.DataFrame):
        return df

    feature_columns = _expected_feature_columns_from_parts(scaler, estimator)
    if feature_columns:
        return df.reindex(columns=list(feature_columns), fill_value=0.0)
    return df


def _prepare_features(model, df: pd.DataFrame):
    scaler, estimator = _split_model(model)
    df = _align_feature_frame(scaler, estimator, df)
    if scaler is not None:
        return estimator, scaler.transform(df)
    return estimator, df


def _binary_probabilities(pos_scores) -> np.ndarray:
    pos = np.asarray(pos_scores, dtype=float).reshape(-1)
    pos = np.clip(pos, 0.0, 1.0)
    return np.column_stack([1.0 - pos, pos])


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    denom = exp_scores.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp_scores / denom


def _decision_function_to_proba(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        pos = 1.0 / (1.0 + np.exp(-scores))
        return _binary_probabilities(pos)
    return _softmax(scores)


def _score_samples_to_proba(scores) -> np.ndarray:
    anomaly_score = -np.asarray(scores, dtype=float).reshape(-1)
    if anomaly_score.size == 0:
        return np.empty((0, 2))
    min_score = anomaly_score.min()
    max_score = anomaly_score.max()
    if np.isfinite(min_score) and np.isfinite(max_score) and max_score > min_score:
        pos = (anomaly_score - min_score) / (max_score - min_score)
    else:
        pos = np.zeros_like(anomaly_score, dtype=float)
    return _binary_probabilities(pos)


def _prediction_to_proba(predictions) -> np.ndarray:
    preds = np.asarray(predictions).reshape(-1)
    return _binary_probabilities(preds.astype(float, copy=False))


def _positive_class_index(estimator, proba: np.ndarray) -> int:
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        classes = list(classes)
        if 1 in classes:
            return classes.index(1)
        if True in classes:
            return classes.index(True)
        if "1" in classes:
            return classes.index("1")
    if proba.ndim == 2 and proba.shape[1] > 1:
        return proba.shape[1] - 1
    return 0


def load_model(path: str):
    """
    Load a trained model from disk and patch older sklearn artifacts as needed.

    For physics_sim models, wrap them with PhysicsModelWrapper to engineer
    features on-the-fly from raw SCADA data.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(model_path)
    scaler, estimator = _split_model(model)

    # Wrap physics_sim and petrobras models to auto-engineer features from raw SCADA
    if "physics_sim" in str(model_path) or "petrobras" in str(model_path):
        if scaler is None:
            scaler_path = model_path.with_name(f"{model_path.stem}_scaler{model_path.suffix}")
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
        return PhysicsModelWrapper(estimator, scaler=scaler)

    if scaler is not None:
        return scaler, estimator
    return estimator


def supports_probability_scores(model) -> bool:
    """
    Return True when the model exposes native score/probability methods.
    """
    _, estimator = _split_model(model)
    return any(
        hasattr(estimator, attr)
        for attr in ("predict_proba", "decision_function", "score_samples")
    )


def supports_leak_score(model) -> bool:
    """
    Return True when the compatibility layer can derive a leak score.
    """
    _, estimator = _split_model(model)
    return hasattr(estimator, "predict")


def expected_feature_columns(model) -> tuple[str, ...] | None:
    """
    Return the feature schema a model expects, when available.
    """
    scaler, estimator = _split_model(model)
    return _expected_feature_columns_from_parts(scaler, estimator)


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """
    Return class predictions for a feature DataFrame.
    """
    # PhysicsModelWrapper handles its own feature engineering
    if isinstance(model, PhysicsModelWrapper):
        return model.predict(df)
    estimator, X = _prepare_features(model, df)
    return estimator.predict(X)


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    """
    Return class probabilities or compatibility-layer approximations.
    """
    # PhysicsModelWrapper handles its own feature engineering
    if isinstance(model, PhysicsModelWrapper):
        return model.predict_proba(df)
    estimator, X = _prepare_features(model, df)

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)

    if hasattr(estimator, "decision_function"):
        return _decision_function_to_proba(estimator.decision_function(X))

    if hasattr(estimator, "score_samples"):
        return _score_samples_to_proba(estimator.score_samples(X))

    if hasattr(estimator, "predict"):
        return _prediction_to_proba(estimator.predict(X))

    raise AttributeError(
        f"{type(estimator).__name__} does not support predictions or probability estimates."
    )


def predict_leak_score(model, df: pd.DataFrame) -> np.ndarray:
    """
    Return the positive-class leak score for each sample.
    """
    # Extract estimator without re-preparing features (predict_proba does that)
    _, estimator = _split_model(model)
    proba = predict_proba(model, df)
    if proba.ndim == 1:
        return proba
    return proba[:, _positive_class_index(estimator, proba)]
