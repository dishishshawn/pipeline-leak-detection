"""
Metrics logging and dataset versioning for training reproducibility.
Handles comprehensive metrics calculation and MLflow experiment tracking.
"""
import logging
from hashlib import sha256
from pathlib import Path
from typing import Dict, Any, Optional
import json

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def compute_dataset_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of dataset file for versioning.
    
    Args:
        file_path: Path to CSV file
        chunk_size: Chunk size for reading large files
        
    Returns:
        SHA256 hash hex string
    """
    hasher = sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Could not compute hash for {file_path}: {e}")
        return ""


def get_dataset_version_info(df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
    """
    Generate comprehensive dataset version and metadata.
    
    Args:
        df: Loaded DataFrame
        file_path: Path to source CSV file
        
    Returns:
        Dictionary with dataset version info
    """
    file_path_obj = Path(file_path)
    
    version_info = {
        "dataset_file": str(file_path_obj.name),
        "file_hash": compute_dataset_hash(file_path),
        "file_size_bytes": file_path_obj.stat().st_size if file_path_obj.exists() else 0,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 ** 2),
        "missing_values": df.isnull().sum().to_dict(),
    }
    
    return version_info


def compute_classification_metrics(y_true, y_pred, y_score=None) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Prediction probabilities (for ROC-AUC, optional)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    
    # ROC-AUC (requires probability scores)
    if y_score is not None:
        try:
            # Handle binary and multiclass
            if len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted'))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics["roc_auc"] = None
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = cr
    
    return metrics


def log_metrics_to_mlflow(
    metrics: Dict[str, Any],
    dataset_version: Dict[str, Any],
    model_name: str,
    dataset_name: Optional[str] = None,
    prefix: str = "",
) -> None:
    """
    Log metrics and dataset version to MLflow.
    
    Args:
        metrics: Dictionary of computed metrics
        dataset_version: Dataset version and metadata
        model_name: Name of the model
        dataset_name: Name of the dataset
        prefix: Prefix for metric names (e.g., 'test_' or 'train_')
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping metric logging")
        return
    
    try:
        # Log dataset version info as parameters
        mlflow.log_param(f"{prefix}dataset_file", dataset_version.get("dataset_file", "unknown"))
        mlflow.log_param(f"{prefix}dataset_name", dataset_name or "unknown")
        mlflow.log_param(f"{prefix}num_rows", dataset_version.get("num_rows", 0))
        mlflow.log_param(f"{prefix}num_columns", dataset_version.get("num_columns", 0))
        
        if dataset_version.get("file_hash"):
            mlflow.log_param(f"{prefix}dataset_hash", dataset_version["file_hash"])
        
        # Log scalar metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"{prefix}{metric_name}", metric_value)
        
        logger.info(f"Logged {len(metrics)} metrics to MLflow for {model_name}")
        
    except Exception as e:
        logger.warning(f"Error logging metrics to MLflow: {e}")


def save_metrics_artifact(
    metrics: Dict[str, Any],
    dataset_version: Dict[str, Any],
    model_name: str,
    output_dir: Path,
    dataset_name: Optional[str] = None,
) -> Path:
    """
    Save metrics and dataset version to JSON artifact.
    
    Args:
        metrics: Dictionary of computed metrics
        dataset_version: Dataset version and metadata
        model_name: Name of the model
        output_dir: Directory to save artifact
        dataset_name: Name of the dataset
        
    Returns:
        Path to saved artifact
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
    }
    
    # Also include detailed dicts but convert to JSON-serializable format
    if "confusion_matrix" in metrics:
        artifact["confusion_matrix"] = metrics["confusion_matrix"]
    
    if "classification_report" in metrics:
        artifact["classification_report"] = metrics["classification_report"]
    
    # Save to JSON
    artifact_path = output_dir / f"{model_name}_metrics.json"
    with open(artifact_path, 'w') as f:
        json.dump(artifact, f, indent=2)
    
    logger.info(f"Saved metrics artifact to {artifact_path}")
    return artifact_path


def log_training_run(
    model,
    model_name: str,
    dataset_name: Optional[str],
    file_path: str,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    train_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    output_dir: Path = None,
) -> None:
    """
    Complete logging of a training run with metrics and dataset versioning.
    Logs to MLflow and saves artifacts.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        dataset_name: Name of the dataset
        file_path: Path to dataset CSV
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        train_metrics: Training metrics dictionary
        test_metrics: Test metrics dictionary
        output_dir: Directory to save metric artifacts
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping experiment tracking")
        return
    
    try:
        # Compute dataset version
        train_df = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
        test_df = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
        
        train_version = get_dataset_version_info(train_df, file_path)
        test_version = get_dataset_version_info(test_df, file_path)
        
        # Log metrics
        log_metrics_to_mlflow(train_metrics, train_version, model_name, dataset_name, prefix="train_")
        log_metrics_to_mlflow(test_metrics, test_version, model_name, dataset_name, prefix="test_")
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        # Save metric artifacts
        if output_dir:
            save_metrics_artifact(train_metrics, train_version, f"{model_name}_train", output_dir, dataset_name)
            save_metrics_artifact(test_metrics, test_version, f"{model_name}_test", output_dir, dataset_name)
        
        logger.info(f"Completed training run logging for {model_name}")
        
    except Exception as e:
        logger.error(f"Error during training run logging: {e}", exc_info=True)
