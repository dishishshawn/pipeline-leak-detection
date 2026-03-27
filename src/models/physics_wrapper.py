"""Wrapper for physics-trained models to work with live simulator data.

Physics models are trained on engineered features (deltas, ratios, rolling stats)
but live data is raw SCADA sensors. This wrapper engineers features on-the-fly
and applies the model, maintaining compatibility with the dashboard interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_binary_predictions(raw_pred) -> np.ndarray:
    raw_pred = np.asarray(raw_pred)
    unique = set(np.unique(raw_pred).tolist())
    if unique.issubset({-1, 1}):
        return (raw_pred == -1).astype(int)
    return raw_pred.astype(int)


class PhysicsModelWrapper:
    """Wraps a physics-trained model to engineer features from raw SCADA data.

    Usage::

        wrapper = PhysicsModelWrapper(model, scaler=None)
        df = pd.DataFrame({...raw SCADA columns...})
        scores = wrapper.predict_proba(df)  # returns (n, 2) array
    """

    def __init__(self, model: object, scaler: object | None = None) -> None:
        self.model = model
        self.scaler = scaler
        self.classes_ = np.array([0, 1])  # binary classification

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features matching training pipeline."""
        df = df.copy()
        features = pd.DataFrame(index=df.index)

        pressure_fallback = pd.to_numeric(df["pressure"], errors="coerce") if "pressure" in df.columns else None
        flow_fallback = pd.to_numeric(df["flow_rate"], errors="coerce") if "flow_rate" in df.columns else None
        temp_fallback = pd.to_numeric(df["temperature"], errors="coerce") if "temperature" in df.columns else None

        # Raw sensors
        pressure_cols = ["P_inlet", "P_mid", "P_outlet"]
        flow_cols = ["Q_inlet", "Q_outlet"]
        temp_cols = ["T_inlet", "T_mid", "T_outlet"]

        for col in pressure_cols:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            elif pressure_fallback is not None:
                features[col] = pressure_fallback.fillna(0.0)
            else:
                features[col] = 0.0

        for col in flow_cols:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            elif flow_fallback is not None:
                features[col] = flow_fallback.fillna(0.0)
            else:
                features[col] = 0.0

        for col in temp_cols:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            elif temp_fallback is not None:
                features[col] = temp_fallback.fillna(0.0)
            else:
                features[col] = 0.0

        # Pressure deltas (Pa) and ratios
        features["P_drop_inlet_to_mid"] = features["P_inlet"] - features["P_mid"]
        features["P_drop_mid_to_outlet"] = features["P_mid"] - features["P_outlet"]
        features["P_drop_total"] = features["P_inlet"] - features["P_outlet"]
        features["P_ratio_in_out"] = features["P_inlet"] / (features["P_outlet"] + 1e-6)

        # Flow balance (leak signature)
        features["Q_imbalance"] = features["Q_inlet"] - features["Q_outlet"]
        features["Q_imbalance_pct"] = (
            100 * features["Q_imbalance"] / (features["Q_inlet"] + 1e-6)
        )
        features["Q_ratio"] = features["Q_inlet"] / (features["Q_outlet"] + 1e-6)

        # Temperature gradient
        features["T_drop_inlet_to_outlet"] = features["T_inlet"] - features["T_outlet"]
        features["T_gradient_per_km"] = features["T_drop_inlet_to_outlet"] * 0.1

        # Rolling stats computed over the live history seen so far. Even when the
        # live dashboard only has coarse SCADA fields, the fallback-mapped series
        # above ensure we still emit the full training schema.
        for col in ["P_inlet", "P_mid", "P_outlet", "Q_inlet", "Q_outlet"]:
            series = features[col]
            features[f"{col}_rolling_mean_5"] = series.rolling(window=5, min_periods=1).mean()
            features[f"{col}_rolling_std_5"] = series.rolling(window=5, min_periods=1).std().fillna(0)

        # Fill any remaining NaN
        features = features.bfill().ffill().fillna(0.0)

        return features

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict binary leak label (0 or 1)."""
        if isinstance(X, pd.DataFrame):
            X_feat = self._engineer_features(X)
        else:
            # If raw numpy array, try to reconstruct DataFrame columns
            return _to_binary_predictions(self.model.predict(X))

        if self.scaler:
            X_feat = self.scaler.transform(X_feat)

        return _to_binary_predictions(self.model.predict(X_feat))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probability of each class.

        Returns shape (n_samples, 2) where column 1 is P(leak).
        """
        if isinstance(X, pd.DataFrame):
            X_feat = self._engineer_features(X)
        else:
            return self.model.predict_proba(X)

        if self.scaler:
            X_feat = self.scaler.transform(X_feat)

        # Get raw model output
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_feat)
        elif hasattr(self.model, "score_samples"):
            # Isolation Forest case
            scores = -self.model.score_samples(X_feat)
            proba_leak = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - proba_leak, proba_leak])
        else:
            # Fallback: use predict and return hard labels
            pred = _to_binary_predictions(self.model.predict(X_feat))
            proba = np.zeros((len(pred), 2))
            proba[np.arange(len(pred)), pred] = 1.0
            return proba
