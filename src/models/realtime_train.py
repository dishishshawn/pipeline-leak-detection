from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
    validation_metrics: dict[str, dict[str, Any]],
    test_metrics: dict[str, dict[str, Any]],
) -> Path:
    summary = {
        "dataset_name": dataset_name,
        "data_path": data_path,
        "feature_columns": feature_columns,
        "validation_metrics": {
            name: _json_ready_metrics(metrics)
            for name, metrics in validation_metrics.items()
        },
        "test_metrics": {
            name: _json_ready_metrics(metrics)
            for name, metrics in test_metrics.items()
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
    X_train, X_valid, X_test, y_train, y_valid, y_test = _split_dataset(
        X,
        y,
        test_size=training_cfg["test_size"],
        validation_size=training_cfg["validation_size"],
        random_state=training_cfg["random_state"],
    )

    random_forest_params = dict(config["models"]["random_forest"]["params"])
    xgboost_params = dict(config["models"]["xgboost"]["params"])
    lightgbm_params = dict(config["models"]["lightgbm"]["params"])
    anomaly_params = dict(config["models"]["isolation_forest"]["params"])

    if random_forest_params.get("class_weight") == "balanced_subsample":
        pass
    elif random_forest_params.get("class_weight") == "balanced":
        pass

    xgboost_params.setdefault("scale_pos_weight", _xgboost_scale_pos_weight(y_train))
    lightgbm_params.setdefault("class_weight", "balanced")

    model_builders = {
        "realtime_random_forest": lambda: _fit_random_forest(X_train, y_train, random_forest_params),
        "realtime_xgboost": lambda: _fit_xgboost(X_train, y_train, xgboost_params),
        "realtime_lightgbm": lambda: _fit_lightgbm(X_train, y_train, lightgbm_params),
        "realtime_isolation_forest": lambda: _fit_isolation_forest(X_train, y_train, anomaly_params),
    }

    trained_models: dict[str, object] = {}
    validation_metrics: dict[str, dict[str, Any]] = {}
    test_metrics: dict[str, dict[str, Any]] = {}

    for model_name, builder in model_builders.items():
        logger.info("Training %s", model_name)
        base_model = builder()
        model = FeatureSubsetModel(base_model, X.columns.tolist())
        trained_models[model_name] = model
        validation_metrics[model_name] = _evaluate_model(model, X_valid, y_valid)
        test_metrics[model_name] = _evaluate_model(model, X_test, y_test)

    ensemble_cfg = config["models"]["hybrid_ensemble"]
    ensemble = _build_ensemble(
        trained_models=trained_models,
        validation_metrics=validation_metrics,
        include_models=ensemble_cfg["include_models"],
        threshold=ensemble_cfg["threshold"],
    )
    ensemble_name = "realtime_hybrid_ensemble"
    trained_models[ensemble_name] = ensemble
    validation_metrics[ensemble_name] = _evaluate_model(ensemble, X_valid, y_valid)
    test_metrics[ensemble_name] = _evaluate_model(ensemble, X_test, y_test)

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
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )

    logger.info("Saved realtime models to %s", output_dir)
    logger.info("Training summary written to %s", summary_path)

    return {
        "saved_paths": saved_paths,
        "summary_path": str(summary_path),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_columns": X.columns.tolist(),
    }


__all__ = ["train_realtime_models", "load_realtime_config"]
