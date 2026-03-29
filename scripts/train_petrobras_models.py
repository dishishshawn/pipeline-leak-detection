#!/usr/bin/env python
"""Train leak detection models on the Petrobras 3W real oil well dataset.

Uses the processed 3W CSV (output of download_petrobras_3w.py) which has
real oil well sensor columns mapped to our SCADA schema:
  P_inlet, P_mid, P_outlet, Q_inlet, Q_outlet, T_inlet, T_mid, T_outlet

Trains 5 model types using physics-style 27-feature engineering, then saves
them to models/petrobras/. These models are automatically discovered by the
dashboard and wrapped in PhysicsModelWrapper for live scoring.

Usage:
    python scripts/train_petrobras_models.py
    python scripts/train_petrobras_models.py --input data/processed/petrobras_3w/petrobras_3w_scada.csv
    python scripts/train_petrobras_models.py --test-split 0.25 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "processed" / "petrobras_3w" / "petrobras_3w_scada.csv"
DEFAULT_OUTPUT = ROOT / "models" / "petrobras"
DEFAULT_SUMMARY_NAME = "petrobras_training_summary.json"
DEFAULT_METRICS_NAME = "petrobras_model_metrics.csv"
FEATURE_CLIP_LIMIT = 1e9
RATIO_CLIP_LIMIT = 1e3
PERCENT_CLIP_LIMIT = 1e4
SENSOR_BOUNDS = {
    "P_": (-1e6, 1e9),
    "T_": (-100.0, 500.0),
    "Q_": (-1e3, 1e3),
}


def sanitize_sensor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize impossible raw sensor readings before feature engineering."""
    df = df.copy()
    for prefix, (lower, upper) in SENSOR_BOUNDS.items():
        for col in [c for c in df.columns if c.startswith(prefix)]:
            values = pd.to_numeric(df[col], errors="coerce")
            values = values.mask(~np.isfinite(values))
            values = values.mask((values < lower) | (values > upper))
            df[col] = values
    return df


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    epsilon: float = 1e-3,
    clip: float | None = None,
) -> pd.Series:
    """Divide with stable denominators and optional clipping."""
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0)

    stabilized = den.where(den.abs() >= epsilon, np.where(den < 0, -epsilon, epsilon))
    result = num / stabilized
    result = result.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if clip is not None:
        result = result.clip(-clip, clip)
    return result


def sanitize_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Make the feature matrix safe for large scikit-learn tree training runs."""
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.bfill().ffill().fillna(0.0)
    features = features.clip(lower=-FEATURE_CLIP_LIMIT, upper=FEATURE_CLIP_LIMIT)
    return features.astype(np.float32)


def resolve_output_paths(
    output_dir: Path,
    summary_json: str | None,
    metrics_csv: str | None,
) -> tuple[Path, Path]:
    """Resolve per-run summary and metrics paths."""
    summary_path = Path(summary_json).expanduser() if summary_json else output_dir / DEFAULT_SUMMARY_NAME
    metrics_path = Path(metrics_csv).expanduser() if metrics_csv else output_dir / DEFAULT_METRICS_NAME
    return summary_path, metrics_path


def engineer_features(df: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    """Engineer 27 physics-style features from Petrobras 3W SCADA data.

    Matches the same feature schema used by train_physics_models.py and
    PhysicsModelWrapper, so trained models are compatible with live scoring.

    Returns:
        (features_df, labels) where features_df has 27 columns and labels are 0/1
    """
    df = sanitize_sensor_frame(df)

    # 3W uses 'target' column (0=normal, 1=anomaly)
    if "target" not in df.columns:
        raise ValueError("Dataset must have 'target' column (0=normal, 1=anomaly)")

    # Drop rows with NaN target
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    features = pd.DataFrame(index=df.index)

    # -- Raw sensors (8 columns) --
    pressure_cols = ["P_inlet", "P_mid", "P_outlet"]
    flow_cols = ["Q_inlet", "Q_outlet"]
    temp_cols = ["T_inlet", "T_mid", "T_outlet"]

    for col in pressure_cols + flow_cols + temp_cols:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            # Fallback from derived columns
            if col.startswith("P_") and "pressure" in df.columns:
                features[col] = pd.to_numeric(df["pressure"], errors="coerce").fillna(0.0)
            elif col.startswith("Q_") and "flow_rate" in df.columns:
                features[col] = pd.to_numeric(df["flow_rate"], errors="coerce").fillna(0.0)
            elif col.startswith("T_") and "temperature" in df.columns:
                features[col] = pd.to_numeric(df["temperature"], errors="coerce").fillna(0.0)
            else:
                features[col] = 0.0

    # -- Pressure deltas and ratios (4 columns) --
    features["P_drop_inlet_to_mid"] = features["P_inlet"] - features["P_mid"]
    features["P_drop_mid_to_outlet"] = features["P_mid"] - features["P_outlet"]
    features["P_drop_total"] = features["P_inlet"] - features["P_outlet"]
    features["P_ratio_in_out"] = safe_divide(
        features["P_inlet"],
        features["P_outlet"],
        clip=RATIO_CLIP_LIMIT,
    )

    # -- Flow balance (3 columns) --
    features["Q_imbalance"] = features["Q_inlet"] - features["Q_outlet"]
    features["Q_imbalance_pct"] = safe_divide(
        100 * features["Q_imbalance"],
        features["Q_inlet"],
        clip=PERCENT_CLIP_LIMIT,
    )
    features["Q_ratio"] = safe_divide(
        features["Q_inlet"],
        features["Q_outlet"],
        clip=RATIO_CLIP_LIMIT,
    )

    # -- Temperature gradient (2 columns) --
    features["T_drop_inlet_to_outlet"] = features["T_inlet"] - features["T_outlet"]
    features["T_gradient_per_km"] = features["T_drop_inlet_to_outlet"] * 0.1

    # -- Rolling stats per scenario (10 columns) --
    # Group by scenario_id (or well_id) for rolling calculations
    group_col = "scenario_id" if "scenario_id" in df.columns else "well_id"
    rolling_cols = ["P_inlet", "P_mid", "P_outlet", "Q_inlet", "Q_outlet"]

    for col in rolling_cols:
        mean_col = f"{col}_rolling_mean_{window}"
        std_col = f"{col}_rolling_std_{window}"
        features[mean_col] = 0.0
        features[std_col] = 0.0

        if group_col in df.columns:
            for group_id in df[group_col].unique():
                mask = df[group_col] == group_id
                idx = df[mask].index
                if len(idx) >= window:
                    rolling_mean = features.loc[idx, col].rolling(window=window, min_periods=1).mean()
                    rolling_std = features.loc[idx, col].rolling(window=window, min_periods=1).std()
                    features.loc[idx, mean_col] = rolling_mean
                    features.loc[idx, std_col] = rolling_std.fillna(0)
        else:
            features[mean_col] = features[col].rolling(window=window, min_periods=1).mean()
            features[std_col] = features[col].rolling(window=window, min_periods=1).std().fillna(0)

    features = sanitize_feature_matrix(features)

    labels = df["target"].values.astype(int)
    return features, labels


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Train all available model types and return results."""
    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    log.info("Training set: %d samples (%d normal, %d anomaly, %.1f%% anomaly)",
             len(y_train), n_neg, n_pos, 100 * n_pos / len(y_train))

    # 1. Logistic Regression (needs scaling)
    log.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_score_lr = lr.predict_proba(X_test_scaled)[:, 1]
    results["logistic_regression"] = {
        "model": lr,
        "scaler": scaler,
        "y_pred": y_pred_lr,
        "y_score": y_score_lr,
        "roc_auc": roc_auc_score(y_test, y_score_lr),
    }

    # 2. Random Forest
    log.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_score_rf = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = {
        "model": rf,
        "scaler": None,
        "y_pred": y_pred_rf,
        "y_score": y_score_rf,
        "roc_auc": roc_auc_score(y_test, y_score_rf),
    }

    # 3. XGBoost
    if HAS_XGBOOST:
        log.info("Training XGBoost...")
        scale_pos_weight = max(n_neg, 1) / max(n_pos, 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        y_score_xgb = xgb_model.predict_proba(X_test)[:, 1]
        results["xgboost"] = {
            "model": xgb_model,
            "scaler": None,
            "y_pred": y_pred_xgb,
            "y_score": y_score_xgb,
            "roc_auc": roc_auc_score(y_test, y_score_xgb),
        }
    else:
        log.warning("XGBoost not installed, skipping.")

    # 4. LightGBM
    if HAS_LIGHTGBM:
        log.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        y_score_lgb = lgb_model.predict_proba(X_test)[:, 1]
        results["lightgbm"] = {
            "model": lgb_model,
            "scaler": None,
            "y_pred": y_pred_lgb,
            "y_score": y_score_lgb,
            "roc_auc": roc_auc_score(y_test, y_score_lgb),
        }
    else:
        log.warning("LightGBM not installed, skipping.")

    # 5. Isolation Forest (unsupervised)
    log.info("Training Isolation Forest...")
    contamination = min(0.5, max(0.01, n_pos / len(y_train)))
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)
    y_score_iso = 1 / (1 + np.exp(-iso.score_samples(X_test)))
    y_pred_iso = (y_score_iso >= 0.5).astype(int)
    results["isolation_forest"] = {
        "model": iso,
        "scaler": None,
        "y_pred": y_pred_iso,
        "y_score": y_score_iso,
        "roc_auc": roc_auc_score(y_test, y_score_iso),
    }

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Train models on Petrobras 3W real oil well data")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                    help="Processed Petrobras 3W CSV path")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                    help="Output directory for trained models")
    ap.add_argument("--summary-json", type=str, default=None,
                    help="Optional JSON training summary path")
    ap.add_argument("--metrics-csv", type=str, default=None,
                    help="Optional CSV metrics output path")
    ap.add_argument("--test-split", type=float, default=0.2,
                    help="Fraction of data for test set")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Limit total rows (for quick testing)")
    ap.add_argument("--run-label", type=str, default=None,
                    help="Optional run label for notebook/cloud tracking")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path, metrics_path = resolve_output_paths(output_dir, args.summary_json, args.metrics_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    run_label = args.run_label or output_dir.name

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        log.error("Run 'python tasks.py petrobras-download' first to download and process the 3W dataset.")
        sys.exit(1)

    # -- Load data --
    t0 = time.time()
    log.info("Loading %s...", input_path)
    df = pd.read_csv(input_path)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    if "well_id" in df.columns:
        log.info("Wells: %d", df["well_id"].nunique())
    if "scenario_id" in df.columns:
        log.info("Scenarios: %d", df["scenario_id"].nunique())
    if "event_type" in df.columns:
        log.info("Event distribution:")
        for event, count in df["event_type"].value_counts().items():
            log.info("  %s: %d (%.1f%%)", event, count, 100 * count / len(df))

    # Optional row limit for testing
    if args.max_rows and len(df) > args.max_rows:
        # Stratified sample to preserve label distribution
        df_normal = df[df["target"] == 0]
        df_anomaly = df[df["target"] == 1]
        n_anomaly = min(len(df_anomaly), args.max_rows // 3)
        n_normal = min(len(df_normal), args.max_rows - n_anomaly)
        df = pd.concat([
            df_normal.sample(n=n_normal, random_state=args.seed),
            df_anomaly.sample(n=n_anomaly, random_state=args.seed),
        ]).reset_index(drop=True)
        log.info("Sampled %d rows (%d normal, %d anomaly)", len(df), n_normal, n_anomaly)

    # -- Feature engineering --
    log.info("Engineering 27 physics-style features...")
    X, y = engineer_features(df, window=5)
    log.info("Features: %d columns, %d rows", X.shape[1], X.shape[0])
    log.info("Label distribution: %s", dict(zip(*np.unique(y, return_counts=True))))

    # -- Train/test split --
    # Split by scenario to prevent data leakage (same well's data in both train/test)
    train_scenario_count = None
    test_scenario_count = None
    if "scenario_id" in df.columns:
        scenarios = df.loc[X.index, "scenario_id"].unique()
        scenario_labels = df.loc[X.index].groupby("scenario_id")["target"].max()  # 1 if any anomaly
        train_scenarios, test_scenarios = train_test_split(
            scenarios,
            test_size=args.test_split,
            random_state=args.seed,
            stratify=scenario_labels.loc[scenarios].values if scenario_labels.nunique() > 1 else None,
        )
        train_mask = df.loc[X.index, "scenario_id"].isin(train_scenarios)
        test_mask = df.loc[X.index, "scenario_id"].isin(test_scenarios)
        X_train, X_test = X.loc[train_mask].values, X.loc[test_mask].values
        y_train, y_test = y[train_mask.values], y[test_mask.values]
        train_scenario_count = len(train_scenarios)
        test_scenario_count = len(test_scenarios)
        log.info("Split by scenario: %d train scenarios, %d test scenarios",
                 len(train_scenarios), len(test_scenarios))
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=args.test_split, random_state=args.seed, stratify=y,
        )

    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # -- Train --
    log.info("Training models...")
    t_train = time.time()
    results = train_models(X_train, y_train, X_test, y_test, list(X.columns))
    log.info("Training complete in %.1fs", time.time() - t_train)

    # -- Save models and metrics --
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "dataset": str(input_path),
        "dataset_type": "petrobras_3w",
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "metrics_csv": str(metrics_path),
        "n_total": len(df),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_train_scenarios": train_scenario_count,
        "n_test_scenarios": test_scenario_count,
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "label_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "test_split": args.test_split,
        "seed": args.seed,
        "models": {},
    }
    metrics_rows: list[dict[str, object]] = []

    for name, result in results.items():
        model_path = output_dir / f"petrobras_{name}.joblib"
        joblib.dump(result["model"], model_path)
        log.info("Saved %s -> %s", name, model_path)

        if result["scaler"]:
            scaler_path = output_dir / f"petrobras_{name}_scaler.joblib"
            joblib.dump(result["scaler"], scaler_path)
            log.info("  Scaler -> %s", scaler_path)

        # Compute metrics
        y_pred = result["y_pred"]
        y_score = result["y_score"]
        roc_auc = result["roc_auc"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        summary["models"][name] = {
            "roc_auc": round(float(roc_auc), 4),
            "accuracy": round(float((y_pred == y_test).mean()), 4),
            "precision": round(float(report.get("1", {}).get("precision", 0)), 4),
            "recall": round(float(report.get("1", {}).get("recall", 0)), 4),
            "f1": round(float(report.get("1", {}).get("f1-score", 0)), 4),
            "confusion_matrix": cm.tolist(),
        }
        metrics_rows.append(
            {
                "run_label": run_label,
                "model": name,
                "roc_auc": summary["models"][name]["roc_auc"],
                "accuracy": summary["models"][name]["accuracy"],
                "precision": summary["models"][name]["precision"],
                "recall": summary["models"][name]["recall"],
                "f1": summary["models"][name]["f1"],
                "n_total": int(len(df)),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "dataset": str(input_path),
                "output_dir": str(output_dir),
            }
        )

        log.info("  ROC-AUC: %.4f | Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f",
                 roc_auc,
                 (y_pred == y_test).mean(),
                 report.get("1", {}).get("precision", 0),
                 report.get("1", {}).get("recall", 0),
                 report.get("1", {}).get("f1-score", 0))

    best_name, best_info = max(summary["models"].items(), key=lambda x: x[1]["roc_auc"])
    summary["best_model"] = {"name": best_name, **best_info}

    # Save summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary -> %s", summary_path)
    pd.DataFrame(metrics_rows).sort_values(["roc_auc", "f1"], ascending=[False, False]).to_csv(
        metrics_path,
        index=False,
    )
    log.info("Metrics CSV -> %s", metrics_path)

    # -- Final report --
    elapsed = time.time() - t0
    log.info("=== PETROBRAS 3W TRAINING COMPLETE (%.1fs) ===", elapsed)
    log.info("  Models trained: %s", list(results.keys()))
    log.info("  Best model: %s (ROC-AUC: %.4f, F1: %.4f)", best_name, best_info["roc_auc"], best_info["f1"])
    log.info("  Output dir: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
