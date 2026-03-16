import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = [
    "timestamp",
    "segment_id",
    "pressure",
    "flow_rate",
    "temperature",
    "valve_status",
    "pump_state",
    "pump_speed",
    "compressor_state",
    "energy_consumption",
    "alarm_triggered",
    "event_type",
    "target",
]


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), file_path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Check that all required columns exist.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        logger.error("Schema validation failed, missing columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")
    logger.debug("Schema validation passed")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform simple cleaning:
    - remove duplicates
    - parse timestamp
    - drop rows missing important values
    """
    df = df.copy()

    df = df.drop_duplicates()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    important_cols = [
        "timestamp",
        "pressure",
        "flow_rate",
        "temperature",
        "target",
    ]

    df = df.dropna(subset=important_cols)
    logger.info("After cleaning: %d rows remain", len(df))

    return df


def load_and_prepare(file_path: str) -> pd.DataFrame:
    """
    Full loading pipeline:
    1. Load CSV
    2. Validate schema
    3. Apply basic cleaning
    """
    df = load_csv(file_path)
    validate_schema(df)
    df = basic_cleaning(df)
    return df


if __name__ == "__main__":
    sample_path = "./data/raw/scada_pipeline.csv"

    try:
        df = load_and_prepare(sample_path)
        print("Data loaded successfully.")
        print(df.head())
        print("\nColumns:")
        print(df.columns.tolist())
    except Exception as exc:
        print(f"Error: {exc}")