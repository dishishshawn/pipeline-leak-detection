from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.data.loader import load_and_prepare
from src.features.engineer import build_features
from src.models.artifacts import save_model_artifact
from src.models.metrics import compute_classification_metrics
from src.models.predict import predict, predict_proba
from src.models.realtime import (
    EnsembleMember,
    FeatureSubsetModel,
    IsolationForestLeakDetector,
    WeightedLeakEnsemble,
    normalize_member_weights,
)
from src.models.train import prepare_training_data


logger = logging.getLogger(__name__)


def load_realtime_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _stratify_labels(y: pd.Series):
    return y if y.nunique() > 1 else None


def _xgboost_scale_pos_weight(y: pd.Series) -> float:
    positive = max(int(y.sum()), 1)
    negative = max(len(y) - positive, 1)
    return negative / positive


def _prepare_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    data_path = config["data"]["path"]
    df = load_and_prepare(data_path)
    sample_size = config["data"].get("sample_size")
    if sample_size:
        df = df.sample(n=sample_size, random_state=config["training"]["random_state"])

    featured = build_features(df)
    X, y = prepare_training_data(featured)
    include_columns = config.get("feature_selection", {}).get("include_columns")
    if include_columns:
        X = X.reindex(columns=include_columns, fill_value=0.0)
    exclude_columns = set(config.get("feature_selection", {}).get("exclude_columns", []))
    if exclude_columns:
        X = X.drop(columns=sorted(exclude_columns), errors="ignore")
    return featured, X, y


def _derive_scenario_groups(featured: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """Build integer group IDs from contiguous (segment_id, scenario_context) blocks."""
    group_cols = []
    for col in ("segment_id", "scenario_context", "scenario_id"):
        if col in featured.columns:
            group_cols.append(col)
    if not group_cols:
        return np.arange(len(X))

    aligned = featured.loc[X.index, group_cols].astype(str)
    combined = aligned.agg("|".join, axis=1)
    groups = combined.ne(combined.shift()).cumsum().values
    return groups


# Deprecated: replaced by scenario-stratified k-fold CV
def _split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    validation_size: float,
    random_state: int,
):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_labels(y),
    )

    validation_share = validation_size / max(1.0 - test_size, 1e-6)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_share,
        random_state=random_state,
        stratify=_stratify_labels(y_train_val),
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def _safe_metric(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    predictions = predict(model, X)
    probabilities = predict_proba(model, X)
    metrics = compute_classification_metrics(y, predictions, probabilities)
    metrics["positive_rate_predicted"] = float(predictions.mean())
    return metrics


_CV_SCALAR_KEYS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")


def _cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    config: dict[str, Any],
    n_folds: int,
    random_state: int,
) -> dict[str, dict[str, float]]:
    """Run scenario-stratified k-fold CV and return aggregated metrics per model."""
    group_series = pd.Series(groups, index=X.index)
    group_labels = y.groupby(group_series).max().reindex(group_series.values).values

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    rf_params = dict(config["models"]["random_forest"]["params"])
    xgb_params = dict(config["models"]["xgboost"]["params"])
    lgb_params = dict(config["models"]["lightgbm"]["params"])
    iso_params = dict(config["models"]["isolation_forest"]["params"])

    fold_metrics: dict[str, list[dict[str, float]]] = {}

    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, group_labels, groups)):
        logger.info("Fold %d/%d", fold_i + 1, n_folds)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        xgb_fold = dict(xgb_params)
        xgb_fold.setdefault("scale_pos_weight", _xgboost_scale_pos_weight(y_train))
        lgb_fold = dict(lgb_params)
        lgb_fold.setdefault("class_weight", "balanced")

        builders = {
            "realtime_random_forest": lambda: _fit_random_forest(X_train, y_train, rf_params),
            "realtime_xgboost": lambda: _fit_xgboost(X_train, y_train, xgb_fold),
            "realtime_lightgbm": lambda: _fit_lightgbm(X_train, y_train, lgb_fold),
            "realtime_isolation_forest": lambda: _fit_isolation_forest(X_train, y_train, iso_params),
        }

        for name, builder in builders.items():
            model = FeatureSubsetModel(builder(), X.columns.tolist())
            metrics = _evaluate_model(model, X_test, y_test)
            scalars = {k: float(metrics.get(k, 0)) for k in _CV_SCALAR_KEYS}
            fold_metrics.setdefault(name, []).append(scalars)

    cv_metrics: dict[str, dict[str, float]] = {}
    for name, folds in fold_metrics.items():
        agg: dict[str, float] = {}
        for key in _CV_SCALAR_KEYS:
            vals = np.array([f[key] for f in folds])
            agg[key] = float(vals.mean())
            agg[f"{key}_std"] = float(vals.std())
        cv_metrics[name] = agg
        logger.info(
            "  %s  ROC-AUC %.4f (+/- %.4f)  F1 %.4f (+/- %.4f)",
            name, agg["roc_auc"], agg["roc_auc_std"], agg["f1_score"], agg["f1_score_std"],
        )

    return cv_metrics


def _fit_random_forest(X_train, y_train, params: dict[str, Any]):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def _fit_xgboost(X_train, y_train, params: dict[str, Any]):
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is not installed in the active environment")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        **params,
    )
    model.fit(X_train, y_train)
    return model


def _fit_lightgbm(X_train, y_train, params: dict[str, Any]):
    if not LGBM_AVAILABLE:
        raise ImportError("lightgbm is not installed in the active environment")

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def _fit_isolation_forest(X_train, y_train, params: dict[str, Any]):
    model = IsolationForestLeakDetector(**params)
    model.fit(X_train, y_train)
    return model


def _build_ensemble(
    trained_models: dict[str, object],
    validation_metrics: dict[str, dict[str, Any]],
    include_models: list[str],
    threshold: float,
) -> WeightedLeakEnsemble:
    included = []
    for name in include_models:
        model = trained_models.get(name)
        if model is None:
            continue
        metrics = validation_metrics.get(name, {})
        score = 0.7 * _safe_metric(metrics, "roc_auc") + 0.3 * _safe_metric(metrics, "f1_score")
        included.append((name, score))

    weights = normalize_member_weights(included)
    members = [
        EnsembleMember(name=name, model=trained_models[name], weight=weights[name])
        for name in include_models
        if name in trained_models and name in weights
    ]
    return WeightedLeakEnsemble(members, threshold=threshold)


def _json_ready_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    ready = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            ready[key] = value
        elif isinstance(value, list):
            ready[key] = value
        elif value is None:
            ready[key] = None
        else:
            ready[key] = float(value) if isinstance(value, (int, float)) else value
    return ready


def _write_summary(
    output_dir: Path,
    *,
    dataset_name: str,
    data_path: str,
    feature_columns: list[str],
    cv_metrics: dict[str, dict[str, float]],
    n_folds: int,
) -> Path:
    summary = {
        "dataset_name": dataset_name,
        "data_path": data_path,
        "feature_columns": feature_columns,
        "n_folds": n_folds,
        "cv_metrics": {
            name: _json_ready_metrics(metrics)
            for name, metrics in cv_metrics.items()
        },
    }
    output_path = output_dir / f"{dataset_name}_realtime_training_summary.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_path


def train_realtime_models(config_path: str) -> dict[str, Any]:
    config = load_realtime_config(config_path)
    logging.basicConfig(level=getattr(logging, config["output"].get("log_level", "INFO")))

    featured, X, y = _prepare_dataset(config)
    training_cfg = config["training"]
    random_state = training_cfg["random_state"]
    n_folds = training_cfg.get("n_folds", 5)

    # --- Scenario-stratified k-fold CV for metric estimation ---
    groups = _derive_scenario_groups(featured, X)
    logger.info("Scenario groups: %d unique groups for %d-fold CV", len(np.unique(groups)), n_folds)

    cv_metrics = _cross_validate_models(
        X, y, groups, config, n_folds=n_folds, random_state=random_state,
    )

    # --- Retrain final models on ALL data for deployment ---
    logger.info("Retraining final models on full dataset (%d rows)", len(X))
    random_forest_params = dict(config["models"]["random_forest"]["params"])
    xgboost_params = dict(config["models"]["xgboost"]["params"])
    lightgbm_params = dict(config["models"]["lightgbm"]["params"])
    anomaly_params = dict(config["models"]["isolation_forest"]["params"])

    xgboost_params.setdefault("scale_pos_weight", _xgboost_scale_pos_weight(y))
    lightgbm_params.setdefault("class_weight", "balanced")

    model_builders = {
        "realtime_random_forest": lambda: _fit_random_forest(X, y, random_forest_params),
        "realtime_xgboost": lambda: _fit_xgboost(X, y, xgboost_params),
        "realtime_lightgbm": lambda: _fit_lightgbm(X, y, lightgbm_params),
        "realtime_isolation_forest": lambda: _fit_isolation_forest(X, y, anomaly_params),
    }

    trained_models: dict[str, object] = {}
    for model_name, builder in model_builders.items():
        logger.info("Training final %s", model_name)
        trained_models[model_name] = FeatureSubsetModel(builder(), X.columns.tolist())

    # Ensemble weights from CV mean metrics
    synthetic_validation = {
        name: {"roc_auc": m["roc_auc"], "f1_score": m["f1_score"]}
        for name, m in cv_metrics.items()
    }
    ensemble_cfg = config["models"]["hybrid_ensemble"]
    ensemble = _build_ensemble(
        trained_models=trained_models,
        validation_metrics=synthetic_validation,
        include_models=ensemble_cfg["include_models"],
        threshold=ensemble_cfg["threshold"],
    )
    ensemble_name = "realtime_hybrid_ensemble"
    trained_models[ensemble_name] = ensemble

    output_dir = Path(config["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = config["data"]["dataset_name"]

    saved_paths: dict[str, str] = {}
    for model_name, model in trained_models.items():
        saved_paths[model_name] = save_model_artifact(
            model=model,
            model_dir=output_dir,
            model_name=model_name,
            dataset_name=dataset_name,
        )

    summary_path = _write_summary(
        output_dir,
        dataset_name=dataset_name,
        data_path=config["data"]["path"],
        feature_columns=X.columns.tolist(),
        cv_metrics=cv_metrics,
        n_folds=n_folds,
    )

    logger.info("Saved realtime models to %s", output_dir)
    logger.info("Training summary written to %s", summary_path)

    return {
        "saved_paths": saved_paths,
        "summary_path": str(summary_path),
        "cv_metrics": cv_metrics,
        "feature_columns": X.columns.tolist(),
    }


__all__ = ["train_realtime_models", "load_realtime_config"]
