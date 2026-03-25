from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.models.predict import predict_proba


def _binary_probabilities(pos_scores: np.ndarray) -> np.ndarray:
    positive = np.asarray(pos_scores, dtype=float).reshape(-1)
    positive = np.clip(positive, 0.0, 1.0)
    return np.column_stack([1.0 - positive, positive])


@dataclass(frozen=True)
class EnsembleMember:
    name: str
    model: object
    weight: float


class FeatureSubsetModel:
    """
    Wrap a model with the exact feature columns it was trained on so live scoring
    cannot accidentally reintroduce leaking or mismatched columns.
    """

    def __init__(self, model, feature_columns: Sequence[str]) -> None:
        self.model = model
        self.feature_columns = tuple(feature_columns)
        classes = getattr(model, "classes_", None)
        if classes is not None:
            self.classes_ = classes

    def _select(self, X):
        if isinstance(X, pd.DataFrame):
            frame = X.reindex(columns=self.feature_columns, fill_value=0.0)
            return frame
        return X

    def predict(self, X):
        return self.model.predict(self._select(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._select(X))

    def decision_function(self, X):
        return self.model.decision_function(self._select(X))

    def score_samples(self, X):
        return self.model.score_samples(self._select(X))


class IsolationForestLeakDetector:
    """
    Isolation forest wrapper that learns normal operating behavior and emits leak-like
    probabilities relative to the normal score distribution seen during training.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        contamination: float = 0.08,
        random_state: int = 42,
        n_jobs: int = 1,
        alert_threshold: float = 0.5,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.alert_threshold = alert_threshold
        self.scaler = StandardScaler()
        self.estimator = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.score_center_: float = 0.0
        self.score_scale_: float = 1.0
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        X_fit = X
        if y is not None:
            normal_mask = np.asarray(y) == 0
            if normal_mask.any():
                X_fit = X.loc[normal_mask] if hasattr(X, "loc") else np.asarray(X)[normal_mask]

        X_scaled = self.scaler.fit_transform(X_fit)
        self.estimator.fit(X_scaled)

        train_scores = self.estimator.score_samples(X_scaled)
        self.score_center_ = float(np.median(train_scores))
        score_std = float(np.std(train_scores))
        self.score_scale_ = score_std if score_std > 1e-6 else 1.0
        return self

    def score_samples(self, X) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.estimator.score_samples(X_scaled)

    def predict_proba(self, X) -> np.ndarray:
        scores = self.score_samples(X)
        # Lower isolation-forest scores are more anomalous.
        normalized = (self.score_center_ - scores) / self.score_scale_
        positive = 1.0 / (1.0 + np.exp(-normalized))
        return _binary_probabilities(positive)

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.alert_threshold).astype(int)


class WeightedLeakEnsemble:
    """
    Weighted probability ensemble over multiple leak models.
    """

    def __init__(
        self,
        members: Sequence[EnsembleMember],
        *,
        threshold: float = 0.5,
    ) -> None:
        if not members:
            raise ValueError("WeightedLeakEnsemble requires at least one member")
        self.members = tuple(members)
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        positive_parts = []
        weights = []
        for member in self.members:
            member_proba = predict_proba(member.model, X)
            positive_parts.append(member_proba[:, 1])
            weights.append(member.weight)

        stacked = np.vstack(positive_parts)
        weight_array = np.asarray(weights, dtype=float)
        denom = weight_array.sum()
        if denom <= 0.0:
            weight_array = np.ones_like(weight_array)
            denom = weight_array.sum()

        positive = np.average(stacked, axis=0, weights=weight_array)
        return _binary_probabilities(positive)

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)


def normalize_member_weights(scores: Iterable[tuple[str, float]]) -> dict[str, float]:
    positive_scores = {name: max(float(score), 0.0) for name, score in scores}
    total = sum(positive_scores.values())
    if total <= 0.0:
        count = max(len(positive_scores), 1)
        return {name: 1.0 / count for name in positive_scores}
    return {name: score / total for name, score in positive_scores.items()}


__all__ = [
    "EnsembleMember",
    "FeatureSubsetModel",
    "IsolationForestLeakDetector",
    "WeightedLeakEnsemble",
    "normalize_member_weights",
]
