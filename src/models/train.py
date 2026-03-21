import logging
from pathlib import Path
from typing import Dict, Any, Optional, Iterator
import yaml

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Advanced training dependencies (optional)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from src.data.loader import load_and_prepare
from src.features.engineer import build_features


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"


def prepare_training_data(df: pd.DataFrame):
    """
    Build feature matrix X and target vector y.
    Excludes metadata columns (timestamp, alarm_triggered) and the target column.
    """
    df = df.copy()

    # Columns to exclude: target, timestamp, and alarm_triggered
    # These are either metadata or not available at prediction time
    exclude_cols = {TARGET_COLUMN, "timestamp", "alarm_triggered"}

    # Select only numeric columns that aren't excluded
    X = df.select_dtypes(include=["number"]).drop(columns=exclude_cols, errors="ignore")
    y = df[TARGET_COLUMN]

    return X, y


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Print evaluation results.
    """
    predictions = model.predict(X_test)

    logger.info("=== %s Evaluation ===", model_name)
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, predictions))
    logger.info("Classification Report:\n%s", classification_report(y_test, predictions))


def save_model(model, output_path: str):
    """
    Save trained model to disk.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    logger.info("Model saved to: %s", output_path)


# Advanced Training Framework
def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def chunked_data_loader(file_path: str, chunk_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
    """Load data in chunks for out-of-core processing."""
    if chunk_size is None:
        # Load all at once
        df = load_and_prepare(file_path)
        df = build_features(df)
        yield df
    else:
        # Load in chunks (simplified - would need adaptation for preprocessing)
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Note: load_and_prepare and build_features need to be adapted for chunks
            yield chunk


def prepare_chunked_training_data(df: pd.DataFrame) -> tuple:
    """Prepare features and target from a data chunk."""
    exclude_cols = {"target", "timestamp", "alarm_triggered"}
    X = df.select_dtypes(include=["number"]).drop(columns=exclude_cols, errors="ignore")
    y = df["target"]
    return X, y


def train_xgboost(X_train, y_train, **kwargs):
    """Train XGBoost model."""
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, **kwargs):
    """Train LightGBM model."""
    if not LGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

    model = lgb.LGBMClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def optimize_hyperparameters(model_func, X_train, y_train, param_space: Dict, method: str = "optuna", **kwargs):
    """Optimize hyperparameters using Optuna or scikit-learn."""
    if method == "optuna" and OPTUNA_AVAILABLE:
        def objective(trial):
            params = {}
            for param_name, values in param_space.items():
                if isinstance(values[0], int):
                    params[param_name] = trial.suggest_int(param_name, min(values), max(values))
                elif isinstance(values[0], float):
                    params[param_name] = trial.suggest_float(param_name, min(values), max(values))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, values)

            model = model_func(X_train, y_train, **params)
            # Simple evaluation - in practice, you'd use cross-validation
            score = model.score(X_train, y_train)  # This is a placeholder
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=kwargs.get("n_trials", 50))
        return study.best_params

    else:
        # Fallback to grid search (simplified)
        from sklearn.model_selection import GridSearchCV

        param_grid = param_space
        # This is a simplified implementation - would need proper model instantiation
        logger.warning("Grid search not fully implemented for advanced models")
        return {}


def setup_experiment_tracking(config: Dict):
    """Initialize experiment tracking."""
    if not config["tracking"]["enabled"]:
        return

    if config["tracking"]["backend"] == "mlflow" and MLFLOW_AVAILABLE:
        if config["tracking"]["tracking_uri"]:
            mlflow.set_tracking_uri(config["tracking"]["tracking_uri"])
        mlflow.set_experiment(config["tracking"]["experiment_name"])
        logger.info("MLflow tracking enabled")
    else:
        logger.warning("Experiment tracking backend not available or not configured")


def advanced_train(config_path: str):
    """Advanced training pipeline with scalability and experiment tracking."""
    config = load_config(config_path)
    setup_experiment_tracking(config)

    logger.info("Starting advanced training with config: %s", config_path)

    # Load and prepare data
    if config["data"]["chunk_size"] is None:
        # In-memory processing
        df = load_and_prepare(config["data"]["path"])
        if config["data"]["sample_size"]:
            df = df.sample(n=config["data"]["sample_size"], random_state=42)
        df = build_features(df)
        X, y = prepare_chunked_training_data(df)
    else:
        # Out-of-core processing (placeholder - needs full implementation)
        logger.warning("Chunked processing not fully implemented yet - using in-memory")
        df = load_and_prepare(config["data"]["path"])
        df = build_features(df)
        X, y = prepare_chunked_training_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    # Train models
    trained_models = {}

    # Logistic Regression
    if config["models"]["logistic_regression"]["enabled"]:
        logger.info("Training Logistic Regression...")
        model = train_logistic_regression(X_train, y_train)
        trained_models["logistic_regression"] = model

    # Random Forest
    if config["models"]["random_forest"]["enabled"]:
        logger.info("Training Random Forest...")
        model = train_random_forest(X_train, y_train)
        trained_models["random_forest"] = model

    # XGBoost
    if config["models"]["xgboost"]["enabled"] and XGB_AVAILABLE:
        logger.info("Training XGBoost...")
        model = train_xgboost(X_train, y_train)
        trained_models["xgboost"] = model

    # LightGBM
    if config["models"]["lightgbm"]["enabled"] and LGBM_AVAILABLE:
        logger.info("Training LightGBM...")
        model = train_lightgbm(X_train, y_train)
        trained_models["lightgbm"] = model

    # Save models
    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in trained_models.items():
        model_path = model_dir / f"{name}.joblib"
        save_model(model, str(model_path))

        # Log to experiment tracking
        if config["tracking"]["enabled"] and MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"{name}_training"):
                # Log basic info - expand as needed
                mlflow.log_param("model_type", name)
                mlflow.sklearn.log_model(model, name)

    logger.info("Advanced training completed. Models saved to: %s", model_dir)


def main():
    data_path = "data/raw/scada_pipeline.csv"

    # Load data
    df = load_and_prepare(data_path)

    # Build features
    df = build_features(df)

    # Prepare training inputs
    X, y = prepare_training_data(df)

    logger.info("Feature columns: %s", X.columns.tolist())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train models
    logistic_model = train_logistic_regression(X_train, y_train)
    random_forest_model = train_random_forest(X_train, y_train)

    # Evaluate models
    evaluate_model(logistic_model, X_test, y_test, "Logistic Regression")
    evaluate_model(random_forest_model, X_test, y_test, "Random Forest")

    # Save models
    save_model(logistic_model, "models/trained/logistic_regression.joblib")
    save_model(random_forest_model, "models/trained/random_forest.joblib")


# python -m src.models.train
if __name__ == "__main__":
    main()