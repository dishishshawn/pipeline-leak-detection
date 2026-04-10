"""ATLAS Model Serving Layer -- deployment infrastructure for the winning model.

Provides:
- Automatic best-model selection with configurable override
- LRU prediction cache to avoid recomputation
- Batch prediction support optimized for simulator throughput
- Latency monitoring with per-call and rolling statistics
- Fallback/ensemble safety net (top-2 candidate cascade)
- Demo metadata (version, provenance, model card)

Usage::

    from src.models.atlas import AtlasServer

    server = AtlasServer.from_model_dir("models")
    scores = server.predict_leak_score(df)       # cached, monitored
    meta   = server.metadata()                   # model card dict
    stats  = server.latency_stats()              # timing breakdown
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.models.predict import (
    load_model,
    predict_leak_score,
    predict_proba,
    predict as predict_class,
    supports_leak_score,
)
from src.models.artifacts import discover_model_artifacts, ModelArtifact

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version & metadata
# ---------------------------------------------------------------------------

ATLAS_VERSION = "1.0.0"
ATLAS_CODENAME = "ATLAS"

# Default fallback ordering: if the primary is not set, try these in order.
# The colleague will override the primary once the winning model is identified.
DEFAULT_CANDIDATE_ORDER = [
    "Realtime Xgboost",
    "Realtime Random Forest",
    "Realtime Lightgbm",
    "Realtime Hybrid Ensemble",
    "Petrobras Lightgbm",
    "Petrobras Xgboost",
    "Petrobras Random Forest",
]


@dataclass
class ModelProvenance:
    """Provenance record for a loaded model."""
    label: str
    artifact_path: str
    source_dir: str
    roc_auc: float = 0.0
    f1: float = 0.0
    calibrated_threshold: float = 0.5
    trained_on: str = "unknown"
    notes: str = ""


@dataclass
class LatencyRecord:
    """Single timing measurement."""
    operation: str
    duration_ms: float
    batch_size: int
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------

def _dataframe_hash(df: pd.DataFrame) -> str:
    """Fast content hash for cache key. Uses shape + first/last values + raw bytes sample."""
    n = len(df)
    if n == 0:
        return "empty"
    # Use a combination of shape, dtypes, and a sample of values
    parts = [
        str(df.shape),
        str(list(df.columns)),
    ]
    # Sample up to 500 bytes from the underlying data for speed
    try:
        raw = df.values.tobytes()[:2048]
        parts.append(hashlib.md5(raw).hexdigest())
    except (TypeError, ValueError):
        # Mixed dtypes -- fall back to string hash of head/tail
        parts.append(str(df.iloc[0].values.tolist()))
        if n > 1:
            parts.append(str(df.iloc[-1].values.tolist()))
    return hashlib.md5("|".join(parts).encode()).hexdigest()


class PredictionCache:
    """Bounded LRU cache for prediction results keyed on input data hash."""

    def __init__(self, maxsize: int = 128):
        self._maxsize = maxsize
        self._cache: dict[str, np.ndarray] = {}
        self._order: deque[str] = deque()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        if key in self._cache:
            return
        if len(self._cache) >= self._maxsize:
            oldest = self._order.popleft()
            self._cache.pop(oldest, None)
        self._cache[key] = value
        self._order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Latency monitor
# ---------------------------------------------------------------------------

class LatencyMonitor:
    """Tracks per-operation latency for diagnostics."""

    def __init__(self, window_size: int = 200):
        self._window_size = window_size
        self._records: deque[LatencyRecord] = deque(maxlen=window_size)

    def record(self, operation: str, duration_ms: float, batch_size: int = 1) -> None:
        self._records.append(LatencyRecord(
            operation=operation,
            duration_ms=duration_ms,
            batch_size=batch_size,
        ))
        if duration_ms > 200:
            logger.warning(
                "ATLAS latency warning: %s took %.1fms (batch=%d)",
                operation, duration_ms, batch_size,
            )

    def stats(self, operation: str | None = None) -> dict[str, Any]:
        """Return rolling latency statistics."""
        records = [r for r in self._records if operation is None or r.operation == operation]
        if not records:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        durations = [r.duration_ms for r in records]
        return {
            "count": len(durations),
            "mean_ms": round(float(np.mean(durations)), 2),
            "p50_ms": round(float(np.median(durations)), 2),
            "p95_ms": round(float(np.percentile(durations, 95)), 2),
            "max_ms": round(float(np.max(durations)), 2),
            "total_batches": sum(r.batch_size for r in records),
        }

    def all_stats(self) -> dict[str, dict[str, Any]]:
        """Return stats grouped by operation."""
        ops = {r.operation for r in self._records}
        result = {op: self.stats(op) for op in sorted(ops)}
        result["_overall"] = self.stats()
        return result

    def reset(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# ATLAS Server
# ---------------------------------------------------------------------------

class AtlasServer:
    """Model serving layer for the ATLAS deployment.

    Wraps model loading, prediction caching, fallback logic, and latency
    monitoring into a single interface that the dashboard and evaluation
    harness can call.
    """

    def __init__(
        self,
        primary_model: object | None = None,
        primary_provenance: ModelProvenance | None = None,
        fallback_models: list[tuple[object, ModelProvenance]] | None = None,
        cache_size: int = 128,
    ):
        self._primary = primary_model
        self._primary_provenance = primary_provenance
        self._fallbacks: list[tuple[object, ModelProvenance]] = fallback_models or []
        self._cache = PredictionCache(maxsize=cache_size)
        self._latency = LatencyMonitor()
        self._active_model: object | None = primary_model
        self._active_provenance: ModelProvenance | None = primary_provenance
        self._using_fallback = False

    # -- Factory ----------------------------------------------------------

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        primary_label: str | None = None,
        candidate_order: Sequence[str] | None = None,
        dataset_type: str = "scada",
        cache_size: int = 128,
    ) -> "AtlasServer":
        """Build an AtlasServer by scanning the model directory.

        Parameters
        ----------
        model_dir : path to the models/ directory
        primary_label : artifact label for the winning model (set by colleague).
            If None, uses the first available from candidate_order.
        candidate_order : ordered list of model labels to try.
        dataset_type : artifact dataset type filter.
        cache_size : LRU cache capacity.
        """
        model_dir = Path(model_dir)
        candidates = candidate_order or DEFAULT_CANDIDATE_ORDER

        # Discover artifacts
        artifacts = discover_model_artifacts(model_dir, dataset_type)

        # Load metrics from evaluation results + training summaries
        metrics = _load_all_metrics(model_dir)

        # Resolve primary
        if primary_label and primary_label in artifacts:
            ordered_labels = [primary_label] + [c for c in candidates if c != primary_label]
        else:
            ordered_labels = list(candidates)

        loaded: list[tuple[object, ModelProvenance]] = []
        for label in ordered_labels:
            art = artifacts.get(label)
            if art is None:
                continue
            try:
                model = load_model(art.path)
            except Exception as exc:
                logger.warning("ATLAS: failed to load %s: %s", label, exc)
                continue
            prov = ModelProvenance(
                label=label,
                artifact_path=art.path,
                source_dir=art.source,
                roc_auc=metrics.get(label, {}).get("roc_auc", 0.0),
                f1=metrics.get(label, {}).get("f1", 0.0),
                calibrated_threshold=metrics.get(label, {}).get("calibrated_threshold", 0.5),
                trained_on=art.dataset_type or "unknown",
            )
            loaded.append((model, prov))

        if not loaded:
            logger.error("ATLAS: no models could be loaded from %s", model_dir)
            return cls(primary_model=None, primary_provenance=None, cache_size=cache_size)

        primary_model, primary_prov = loaded[0]
        fallbacks = loaded[1:]
        logger.info(
            "ATLAS: primary=%s (AUC=%.3f, F1=%.3f), %d fallback(s)",
            primary_prov.label, primary_prov.roc_auc, primary_prov.f1, len(fallbacks),
        )
        return cls(
            primary_model=primary_model,
            primary_provenance=primary_prov,
            fallback_models=fallbacks,
            cache_size=cache_size,
        )

    # -- Core prediction API ----------------------------------------------

    def predict_leak_score(self, df: pd.DataFrame, use_cache: bool = True) -> np.ndarray:
        """Return per-row leak scores with caching and fallback.

        This is the main entry point for the dashboard live view and batch
        evaluation. Returns a 1-D array of floats in [0, 1].
        """
        if df.empty:
            return np.array([], dtype=float)

        cache_key = _dataframe_hash(df) if use_cache else None

        if use_cache and cache_key:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        t0 = time.perf_counter()
        scores = self._predict_with_fallback(df)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latency.record("predict_leak_score", elapsed_ms, batch_size=len(df))

        if use_cache and cache_key:
            self._cache.put(cache_key, scores)

        return scores

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return full probability matrix (n_samples, 2)."""
        if df.empty:
            return np.empty((0, 2), dtype=float)
        t0 = time.perf_counter()
        result = self._predict_proba_with_fallback(df)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latency.record("predict_proba", elapsed_ms, batch_size=len(df))
        return result

    def predict_class(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary class predictions."""
        if df.empty:
            return np.array([], dtype=int)
        t0 = time.perf_counter()
        result = self._predict_class_with_fallback(df)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latency.record("predict_class", elapsed_ms, batch_size=len(df))
        return result

    def predict_batch(
        self,
        frames: Sequence[pd.DataFrame],
    ) -> list[np.ndarray]:
        """Score multiple DataFrames efficiently. Returns list of score arrays."""
        t0 = time.perf_counter()
        # Concatenate for vectorized scoring, then split back
        if not frames:
            return []
        lengths = [len(f) for f in frames]
        combined = pd.concat(frames, ignore_index=True)
        all_scores = self.predict_leak_score(combined, use_cache=False)
        results = []
        offset = 0
        for length in lengths:
            results.append(all_scores[offset:offset + length])
            offset += length
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latency.record("predict_batch", elapsed_ms, batch_size=sum(lengths))
        return results

    # -- Fallback logic ---------------------------------------------------

    def _predict_with_fallback(self, df: pd.DataFrame) -> np.ndarray:
        """Try primary model, then cascade through fallbacks."""
        # Try the currently active model first
        if self._active_model is not None:
            try:
                scores = predict_leak_score(self._active_model, df)
                return np.asarray(scores, dtype=float)
            except Exception as exc:
                logger.warning(
                    "ATLAS: active model %s failed: %s -- trying fallback",
                    self._active_provenance.label if self._active_provenance else "unknown",
                    exc,
                )

        # Cascade through fallbacks
        for model, prov in self._fallbacks:
            try:
                scores = predict_leak_score(model, df)
                logger.info("ATLAS: fell back to %s", prov.label)
                self._active_model = model
                self._active_provenance = prov
                self._using_fallback = True
                return np.asarray(scores, dtype=float)
            except Exception as exc:
                logger.warning("ATLAS: fallback %s also failed: %s", prov.label, exc)
                continue

        # All models failed -- return safe zeros
        logger.error("ATLAS: all models failed, returning zeros")
        return np.zeros(len(df), dtype=float)

    def _predict_proba_with_fallback(self, df: pd.DataFrame) -> np.ndarray:
        """predict_proba with fallback cascade."""
        if self._active_model is not None:
            try:
                return predict_proba(self._active_model, df)
            except Exception:
                pass
        for model, prov in self._fallbacks:
            try:
                result = predict_proba(model, df)
                self._active_model = model
                self._active_provenance = prov
                self._using_fallback = True
                return result
            except Exception:
                continue
        return np.column_stack([np.ones(len(df)), np.zeros(len(df))])

    def _predict_class_with_fallback(self, df: pd.DataFrame) -> np.ndarray:
        """predict with fallback cascade."""
        if self._active_model is not None:
            try:
                return predict_class(self._active_model, df)
            except Exception:
                pass
        for model, prov in self._fallbacks:
            try:
                result = predict_class(model, df)
                self._active_model = model
                self._active_provenance = prov
                self._using_fallback = True
                return result
            except Exception:
                continue
        return np.zeros(len(df), dtype=int)

    # -- Metadata & diagnostics -------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if at least one model is loaded and functional."""
        return self._active_model is not None or len(self._fallbacks) > 0

    @property
    def is_using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def active_model_label(self) -> str:
        if self._active_provenance:
            return self._active_provenance.label
        return "none"

    def metadata(self) -> dict[str, Any]:
        """Return model card metadata for dashboard display.

        Example output::

            {
                "display": "ATLAS v1.0.0 - ROC-AUC 0.95 - F1 0.93",
                "version": "1.0.0",
                "codename": "ATLAS",
                "primary_model": "Realtime Random Forest",
                ...
            }
        """
        prov = self._active_provenance
        if prov is None:
            return {
                "display": f"{ATLAS_CODENAME} v{ATLAS_VERSION} -- no model loaded",
                "version": ATLAS_VERSION,
                "codename": ATLAS_CODENAME,
                "primary_model": "none",
                "status": "unavailable",
            }
        display_parts = [f"{ATLAS_CODENAME} v{ATLAS_VERSION}"]
        if prov.roc_auc > 0:
            display_parts.append(f"ROC-AUC {prov.roc_auc:.2f}")
        if prov.f1 > 0:
            display_parts.append(f"F1 {prov.f1:.2f}")
        return {
            "display": " | ".join(display_parts),
            "version": ATLAS_VERSION,
            "codename": ATLAS_CODENAME,
            "primary_model": prov.label,
            "artifact_path": prov.artifact_path,
            "source_dir": prov.source_dir,
            "roc_auc": prov.roc_auc,
            "f1": prov.f1,
            "calibrated_threshold": prov.calibrated_threshold,
            "trained_on": prov.trained_on,
            "using_fallback": self._using_fallback,
            "fallback_count": len(self._fallbacks),
            "status": "fallback" if self._using_fallback else "primary",
        }

    def model_card_display(self) -> str:
        """One-line string for dashboard badge.

        Example: "ATLAS v1.0.0 | ROC-AUC 0.95 | F1 0.93"
        """
        return self.metadata()["display"]

    def latency_stats(self, operation: str | None = None) -> dict[str, Any]:
        """Return latency statistics for monitoring."""
        return self._latency.all_stats() if operation is None else self._latency.stats(operation)

    def cache_stats(self) -> dict[str, Any]:
        """Return prediction cache statistics."""
        return {
            "size": self._cache.size,
            "hits": self._cache.hits,
            "misses": self._cache.misses,
            "hit_rate": round(self._cache.hit_rate, 3),
        }

    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self._cache.clear()

    def reset_latency(self) -> None:
        """Reset latency monitoring counters."""
        self._latency.reset()

    def swap_primary(self, label: str, model: object, provenance: ModelProvenance) -> None:
        """Hot-swap the primary model without restarting.

        This is the 1-minute swap path: colleague identifies the winner,
        calls server.swap_primary(...) and the dashboard picks it up.
        """
        # Demote current primary to first fallback position
        if self._primary is not None and self._primary_provenance is not None:
            self._fallbacks.insert(0, (self._primary, self._primary_provenance))
        self._primary = model
        self._primary_provenance = provenance
        self._active_model = model
        self._active_provenance = provenance
        self._using_fallback = False
        self._cache.clear()
        logger.info("ATLAS: swapped primary to %s", label)

    def set_primary_by_label(
        self,
        label: str,
        model_dir: str | Path = Path("models"),
        dataset_type: str = "scada",
    ) -> bool:
        """Set primary model by label, loading from disk if needed.

        Returns True if successful. This is the simplest swap path.
        """
        artifacts = discover_model_artifacts(Path(model_dir), dataset_type)
        art = artifacts.get(label)
        if art is None:
            logger.warning("ATLAS: label %s not found in artifacts", label)
            return False
        try:
            model = load_model(art.path)
        except Exception as exc:
            logger.warning("ATLAS: failed to load %s: %s", label, exc)
            return False
        metrics = _load_all_metrics(Path(model_dir))
        prov = ModelProvenance(
            label=label,
            artifact_path=art.path,
            source_dir=art.source,
            roc_auc=metrics.get(label, {}).get("roc_auc", 0.0),
            f1=metrics.get(label, {}).get("f1", 0.0),
            calibrated_threshold=metrics.get(label, {}).get("calibrated_threshold", 0.5),
            trained_on=art.dataset_type or "unknown",
        )
        self.swap_primary(label, model, prov)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_metrics(model_dir: Path) -> dict[str, dict[str, float]]:
    """Load metrics from evaluation results and training summaries."""
    from src.models.artifacts import _normalize_label

    metrics: dict[str, dict[str, float]] = {}

    # 1) Evaluation results (calibrated thresholds + eval metrics)
    eval_path = Path("reports/evaluation_results.json")
    if eval_path.exists():
        try:
            data = json.loads(eval_path.read_text())
            thresholds = data.get("calibrated_thresholds", {})
            for entry in data.get("models", []):
                name = entry["model_name"]
                metrics.setdefault(name, {})
                metrics[name]["calibrated_threshold"] = entry.get("calibrated_threshold", 0.5)
                metrics[name]["sensitivity_micro_leak"] = entry.get("sensitivity_micro_leak", 0.0)
        except Exception:
            pass

    # 2) Training summaries (ROC-AUC, F1)
    for summary_path in sorted(model_dir.rglob("*summary*.json")):
        try:
            with open(summary_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        models_list = data if isinstance(data, list) else data.get("models", [data])
        for entry in models_list:
            stem = entry.get("model_file", entry.get("model_name", ""))
            if isinstance(stem, str) and stem.endswith(".joblib"):
                stem = stem.replace(".joblib", "")
            label = _normalize_label(stem) if stem else None
            if not label:
                continue
            roc = entry.get("roc_auc", entry.get("test_roc_auc", 0.0))
            f1 = entry.get("f1", entry.get("test_f1", 0.0))
            if roc or f1:
                metrics.setdefault(label, {})
                metrics[label]["roc_auc"] = roc
                metrics[label]["f1"] = f1

    return metrics
