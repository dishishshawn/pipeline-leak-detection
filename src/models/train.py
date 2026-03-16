from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data.loader import load_and_prepare
from src.features.engineer import build_features


TARGET_COLUMN = "target"


def prepare_training_data(df: pd.DataFrame):
    """
    Build feature matrix X and target vector y.
    """
    df = df.copy()

    # Columns we do not want to use directly as model inputs
    drop_cols = [col for col in df.columns if col not in ["temperature", "pressure", "flow_rate", TARGET_COLUMN]]

    feature_df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric columns
    X = feature_df.select_dtypes(include=["number"])
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

    print(f"\n=== {model_name} Evaluation ===")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def save_model(model, output_path: str):
    """
    Save trained model to disk.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Model saved to: {output_path}")


def main():
    data_path = "data/raw/scada_pipeline.csv"

    # Load data
    df = load_and_prepare(data_path)

    # Build features
    df = build_features(df)

    # Prepare training inputs
    X, y = prepare_training_data(df)

    print("Feature columns used for training:")
    print(X.columns.tolist())

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