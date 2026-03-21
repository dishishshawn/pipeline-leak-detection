from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model(path: str):
    """
    Load a trained model from disk.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(model_path)


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """
    Return binary class predictions for a feature DataFrame.
    """
    if isinstance(model, tuple) and len(model) == 2:
        # Handle scaled models (scaler, clf)
        scaler, clf = model
        X_scaled = scaler.transform(df)
        return clf.predict(X_scaled)
    else:
        return model.predict(df)


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    """
    Return class probability estimates.
    Returns an array of shape (n_samples, n_classes).
    """
    if isinstance(model, tuple) and len(model) == 2:
        # Handle scaled models (scaler, clf)
        scaler, clf = model
        if not hasattr(clf, "predict_proba"):
            raise AttributeError(
                f"{type(clf).__name__} does not support probability estimates."
            )
        X_scaled = scaler.transform(df)
        return clf.predict_proba(X_scaled)
    else:
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"{type(model).__name__} does not support probability estimates."
            )
        return model.predict_proba(df)


def predict_leak_score(model, df: pd.DataFrame) -> np.ndarray:
    """
    Return the probability of the positive (leak) class for each sample.
    Assumes class order [0=normal, 1=leak].
    """
    proba = predict_proba(model, df)
    return proba[:, 1]
