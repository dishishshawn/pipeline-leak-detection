#!/usr/bin/env python
"""Train leak detection models on physics-simulated pipeline data.

Reads the physics simulator output, engineers features, trains 5 model types
(Logistic Regression, Random Forest, XGBoost, LightGBM, Isolation Forest),
and saves them to models/physics_sim/ alongside a summary JSON.

Usage:
    python scripts/train_physics_models.py
    python scripts/train_physics_models.py --input data/physics_sim/scada_timeseries.csv
    python scripts/train_physics_models.py --test-split 0.3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
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


def engineer_features(df: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    """Engineer features from raw physics SCADA data.

    For each row, compute:
    - Raw sensors (P_inlet, P_mid, P_outlet, Q_inlet, Q_outlet, T_inlet, T_mid, T_outlet)
    - Deltas (P drop, Q change, T gradient)
    - Ratios (P_in/P_out, Q_in/Q_out)
    - Rolling stats (rolling mean/std of pressure and flow, per scenario)

    Returns:
        (features_df, labels) where features_df is a DataFrame and labels are 0/1
    """
    df = df.copy()

    # Drop NaN leak_position (normal scenarios have NaN)
    df = df.dropna(subset=["leak_label"])

    features = pd.DataFrame(index=df.index)

    # Raw sensors
    pressure_cols = ["P_inlet", "P_mid", "P_outlet"]
    flow_cols = ["Q_inlet", "Q_outlet"]
    temp_cols = ["T_inlet", "T_mid", "T_outlet"]

    for col in pressure_cols + flow_cols + temp_cols:
        features[col] = df[col]

    # Pressure deltas (Pa) and ratios
    features["P_drop_inlet_to_mid"] = df["P_inlet"] - df["P_mid"]
    features["P_drop_mid_to_outlet"] = df["P_mid"] - df["P_outlet"]
    features["P_drop_total"] = df["P_inlet"] - df["P_outlet"]
    features["P_ratio_in_out"] = df["P_inlet"] / (df["P_outlet"] + 1e-6)

    # Flow balance (leak signature: Q_in should roughly equal Q_out in normal case)
    features["Q_imbalance"] = df["Q_inlet"] - df["Q_outlet"]
    features["Q_imbalance_pct"] = (
        100 * features["Q_imbalance"] / (df["Q_inlet"] + 1e-6)
    )
    features["Q_ratio"] = df["Q_inlet"] / (df["Q_outlet"] + 1e-6)

    # Temperature gradient (should decrease along pipe)
    features["T_drop_inlet_to_outlet"] = df["T_inlet"] - df["T_outlet"]
    features["T_gradient_per_km"] = (
        features["T_drop_inlet_to_outlet"] * 1000 / 10000
    )

    # Rolling stats per scenario (within-scenario dynamics)
    rolling_cols = ["P_inlet", "P_mid", "P_outlet", "Q_inlet", "Q_outlet"]
    for col in rolling_cols:
        for scenario_id in df["scenario_id"].unique():
            mask = df["scenario_id"] == scenario_id
            idx = df[mask].index
            if len(idx) >= window:
                rolling_mean = df.loc[idx, col].rolling(window=window, min_periods=1).mean()
                rolling_std = df.loc[idx, col].rolling(window=window, min_periods=1).std()
                features.loc[idx, f"{col}_rolling_mean_{window}"] = rolling_mean
                features.loc[idx, f"{col}_rolling_std_{window}"] = rolling_std.fillna(0)

    # Fill NaN from rolling calcs
    features = features.bfill().ffill().fillna(0)

    labels = df["leak_label"].values.astype(int)
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

    # 1. Logistic Regression
    log.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
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
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
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

    # 3. XGBoost (if available)
    if HAS_XGBOOST:
        log.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
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

    # 4. LightGBM (if available)
    if HAS_LIGHTGBM:
        log.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
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

    # 5. Isolation Forest (anomaly detection)
    log.info("Training Isolation Forest...")
    iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    y_score_iso = 1 / (1 + np.exp(-iso.score_samples(X_test)))  # sigmoid transform to [0,1]
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
    ap = argparse.ArgumentParser(description="Train physics-based leak detection models")
    ap.add_argument(
        "--input",
        type=str,
        default="data/physics_sim/scada_timeseries.csv",
        help="Physics dataset CSV",
    )
    ap.add_argument("--output", type=str, default="models/physics_sim", help="Output directory")
    ap.add_argument("--test-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and feature engineer
    log.info(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df)} rows, {df.scenario_id.nunique()} scenarios")

    log.info("Engineering features...")
    X, y = engineer_features(df, window=5)
    log.info(f"Features shape: {X.shape}")
    log.info(f"Leak distribution: {np.bincount(y)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=args.test_split, random_state=args.seed, stratify=y
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Train all models
    log.info("Training models...")
    results = train_models(X_train, y_train, X_test, y_test, list(X.columns))

    # Save models and metrics
    summary = {
        "dataset": str(input_path),
        "n_total": len(df),
        "n_scenarios": df.scenario_id.nunique(),
        "leak_label_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "n_features": X.shape[1],
        "test_split": args.test_split,
        "seed": args.seed,
        "models": {},
    }

    for name, result in results.items():
        log.info(f"Saving {name}...")
        model_path = output_dir / f"physics_sim_{name}.joblib"
        scaler_path = output_dir / f"physics_sim_{name}_scaler.joblib" if result["scaler"] else None

        joblib.dump(result["model"], model_path)
        if result["scaler"]:
            joblib.dump(result["scaler"], scaler_path)

        # Metrics
        y_pred = result["y_pred"]
        y_score = result["y_score"]
        roc_auc = result["roc_auc"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        summary["models"][name] = {
            "roc_auc": round(float(roc_auc), 4),
            "accuracy": round(float((y_pred == y_test).mean()), 4),
            "precision": round(float(report["1"]["precision"]), 4),
            "recall": round(float(report["1"]["recall"]), 4),
            "f1": round(float(report["1"]["f1-score"]), 4),
            "confusion_matrix": cm.tolist(),
        }

        log.info(f"  ROC-AUC: {roc_auc:.4f} | Acc: {(y_pred == y_test).mean():.4f} | "
                 f"Prec: {report['1']['precision']:.4f} | Rec: {report['1']['recall']:.4f}")

    # Save summary
    summary_path = output_dir / "physics_sim_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary → {summary_path}")

    log.info("=== TRAINING COMPLETE ===")
    log.info(f"Models: {list(results.keys())}")
    log.info(f"Best: {max(summary['models'].items(), key=lambda x: x[1]['roc_auc'])[0]}")


if __name__ == "__main__":
    main()
