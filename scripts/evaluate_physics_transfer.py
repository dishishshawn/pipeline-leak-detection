#!/usr/bin/env python
"""Evaluate how physics-sim-trained models transfer to Petrobras 3W data.

This script loads the saved `models/physics_sim/*.joblib` artifacts, engineers
the same 27-feature schema used for Petrobras training, and scores held-out
Petrobras scenarios via scenario-stratified k-fold evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_petrobras_models import engineer_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "processed" / "petrobras_3w" / "petrobras_3w_scada.csv"
DEFAULT_MODEL_DIR = ROOT / "models" / "physics_sim"
DEFAULT_SUMMARY = ROOT / "reports" / "physics_to_petrobras_transfer_summary.json"
DEFAULT_METRICS_CSV = ROOT / "reports" / "physics_to_petrobras_transfer_metrics.csv"
DEFAULT_BASELINE_SUMMARY = DEFAULT_MODEL_DIR / "physics_sim_training_summary.json"
CV_SCALAR_KEYS = ("roc_auc", "accuracy", "precision", "recall", "f1")


@dataclass(frozen=True)
class ModelBundle:
    name: str
    label: str
    model: object
    scaler: object | None


def _format_model_label(name: str) -> str:
    return name.replace("_", " ").title()


def load_baseline_roc_aucs(summary_path: Path) -> dict[str, float]:
    """Load source-domain ROC-AUC values for gap reporting when available."""
    if not summary_path.exists():
        return {}

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    baselines: dict[str, float] = {}
    for model_name, metrics in summary.get("models", {}).items():
        roc_auc = metrics.get("roc_auc")
        if roc_auc is not None:
            baselines[model_name] = float(roc_auc)
    return baselines


def load_physics_models(model_dir: Path) -> dict[str, ModelBundle]:
    """Load saved physics model artifacts and optional scalers."""
    bundles: dict[str, ModelBundle] = {}
    for model_path in sorted(model_dir.glob("physics_sim_*.joblib")):
        if model_path.stem.endswith("_scaler"):
            continue

        name = model_path.stem.removeprefix("physics_sim_")
        scaler_path = model_path.with_name(f"{model_path.stem}_scaler{model_path.suffix}")
        bundles[name] = ModelBundle(
            name=name,
            label=_format_model_label(name),
            model=joblib.load(model_path),
            scaler=joblib.load(scaler_path) if scaler_path.exists() else None,
        )
    return bundles


def sample_scenarios(df: pd.DataFrame, max_scenarios: int | None, seed: int) -> pd.DataFrame:
    """Limit evaluation size by sampling whole scenarios with label stratification."""
    if max_scenarios is None or "scenario_id" not in df.columns:
        return df

    scenario_labels = df.groupby("scenario_id")["target"].max()
    if len(scenario_labels) <= max_scenarios:
        return df

    rng = np.random.default_rng(seed)
    anomaly_ids = scenario_labels[scenario_labels == 1].index.to_numpy()
    normal_ids = scenario_labels[scenario_labels == 0].index.to_numpy()

    anomaly_quota = int(round(max_scenarios * len(anomaly_ids) / len(scenario_labels)))
    if len(anomaly_ids):
        anomaly_quota = max(1, anomaly_quota)
    anomaly_quota = min(len(anomaly_ids), anomaly_quota)
    normal_quota = min(len(normal_ids), max_scenarios - anomaly_quota)
    anomaly_quota = min(len(anomaly_ids), max_scenarios - normal_quota)

    selected: list[str] = []
    if anomaly_quota:
        selected.extend(rng.choice(anomaly_ids, size=anomaly_quota, replace=False).tolist())
    if normal_quota:
        selected.extend(rng.choice(normal_ids, size=normal_quota, replace=False).tolist())

    sampled = df[df["scenario_id"].isin(selected)].copy()
    return sampled.sort_values(["scenario_id", "timestamp"] if "timestamp" in sampled.columns else ["scenario_id"])


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float | list[list[int]] | None]:
    """Build a consistent binary-classification metric bundle."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics: dict[str, float | list[list[int]] | None] = {
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "precision": round(float(report.get("1", {}).get("precision", 0)), 4),
        "recall": round(float(report.get("1", {}).get("recall", 0)), 4),
        "f1": round(float(report.get("1", {}).get("f1-score", 0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def aggregate_fold_metrics(fold_metrics: dict[str, list[dict[str, float | None]]]) -> dict[str, dict[str, float | None]]:
    """Aggregate fold metrics into mean/std values without losing overall scores."""
    summary: dict[str, dict[str, float | None]] = {}
    for model_name, metrics_list in fold_metrics.items():
        agg: dict[str, float | None] = {}
        for key in CV_SCALAR_KEYS:
            values = [metrics[key] for metrics in metrics_list if metrics.get(key) is not None]
            if values:
                series = np.array(values, dtype=float)
                agg[f"{key}_cv_mean"] = round(float(series.mean()), 4)
                agg[f"{key}_cv_std"] = round(float(series.std()), 4)
            else:
                agg[f"{key}_cv_mean"] = None
                agg[f"{key}_cv_std"] = None
        summary[model_name] = agg
    return summary


def predict_bundle(bundle: ModelBundle, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run inference for a saved physics model bundle on engineered features."""
    X_eval = X.values
    if bundle.scaler is not None:
        X_eval = bundle.scaler.transform(X_eval)

    if hasattr(bundle.model, "predict_proba"):
        y_pred = bundle.model.predict(X_eval)
        y_score = bundle.model.predict_proba(X_eval)[:, 1]
        return np.asarray(y_pred, dtype=int), np.asarray(y_score, dtype=float)

    if hasattr(bundle.model, "score_samples"):
        y_score = 1.0 / (1.0 + np.exp(-bundle.model.score_samples(X_eval)))
        y_pred = (y_score >= 0.5).astype(int)
        return np.asarray(y_pred, dtype=int), np.asarray(y_score, dtype=float)

    y_pred = np.asarray(bundle.model.predict(X_eval), dtype=int)
    return y_pred, y_pred.astype(float)


def evaluate_transfer(
    X: pd.DataFrame,
    y: np.ndarray,
    scenario_ids: np.ndarray,
    bundles: dict[str, ModelBundle],
    *,
    n_folds: int,
    seed: int,
    baseline_rocs: dict[str, float] | None = None,
) -> tuple[dict[str, dict[str, float | list[list[int]] | None]], dict[str, int]]:
    """Evaluate fixed sim-trained models over scenario-stratified Petrobras folds."""
    baseline_rocs = baseline_rocs or {}
    group_ids = pd.factorize(scenario_ids)[0]
    group_labels = pd.Series(y, index=X.index).groupby(group_ids).transform("max").values
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics: dict[str, list[dict[str, float | None]]] = {name: [] for name in bundles}
    oof_predictions = {name: np.zeros(len(X), dtype=int) for name in bundles}
    oof_scores = {name: np.zeros(len(X), dtype=float) for name in bundles}
    train_rows: list[int] = []
    test_rows: list[int] = []
    train_groups: list[int] = []
    test_groups: list[int] = []

    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X.values, group_labels, group_ids)):
        log.info("Fold %d/%d: test rows=%d, test scenarios=%d",
                 fold_i + 1, n_folds, len(test_idx), len(np.unique(group_ids[test_idx])))
        train_rows.append(int(len(train_idx)))
        test_rows.append(int(len(test_idx)))
        train_groups.append(int(len(np.unique(group_ids[train_idx]))))
        test_groups.append(int(len(np.unique(group_ids[test_idx]))))

        X_test = X.iloc[test_idx]
        y_test = y[test_idx]
        for model_name, bundle in bundles.items():
            y_pred, y_score = predict_bundle(bundle, X_test)
            fold_metrics[model_name].append(summarize_metrics(y_test, y_pred, y_score))
            oof_predictions[model_name][test_idx] = y_pred
            oof_scores[model_name][test_idx] = y_score

    summary = aggregate_fold_metrics(fold_metrics)
    for model_name in bundles:
        overall = summarize_metrics(y, oof_predictions[model_name], oof_scores[model_name])
        summary[model_name].update(overall)
        if model_name in baseline_rocs and summary[model_name]["roc_auc"] is not None:
            source_roc_auc = round(float(baseline_rocs[model_name]), 4)
            summary[model_name]["source_roc_auc"] = source_roc_auc
            summary[model_name]["roc_auc_gap_vs_source"] = round(
                float(summary[model_name]["roc_auc"]) - source_roc_auc,
                4,
            )

    fold_sizes = {
        "avg_train_rows": int(round(float(np.mean(train_rows)))) if train_rows else 0,
        "avg_test_rows": int(round(float(np.mean(test_rows)))) if test_rows else 0,
        "avg_train_scenarios": int(round(float(np.mean(train_groups)))) if train_groups else 0,
        "avg_test_scenarios": int(round(float(np.mean(test_groups)))) if test_groups else 0,
    }
    return summary, fold_sizes


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate physics-trained models on Petrobras data")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                    help="Processed Petrobras 3W CSV path")
    ap.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR),
                    help="Directory containing physics_sim model artifacts")
    ap.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY),
                    help="Output JSON summary path")
    ap.add_argument("--metrics-csv", type=str, default=str(DEFAULT_METRICS_CSV),
                    help="Output CSV metrics path")
    ap.add_argument("--baseline-summary", type=str, default=str(DEFAULT_BASELINE_SUMMARY),
                    help="Optional physics training summary for source-domain comparison")
    ap.add_argument("--n-folds", type=int, default=5,
                    help="Scenario-stratified Petrobras evaluation folds")
    ap.add_argument("--max-scenarios", type=int, default=None,
                    help="Optional cap on Petrobras scenarios for a faster transfer readout")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    summary_path = Path(args.summary_json)
    metrics_path = Path(args.metrics_csv)
    baseline_summary_path = Path(args.baseline_summary)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    log.info("Loading Petrobras data from %s...", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "scenario_id" not in df.columns:
        raise SystemExit("Processed Petrobras CSV must include scenario_id for scenario-stratified evaluation.")

    df = sample_scenarios(df, args.max_scenarios, args.seed).reset_index(drop=True)
    log.info("Evaluating %d rows across %d scenarios", len(df), df["scenario_id"].nunique())

    log.info("Engineering Petrobras feature schema...")
    X, y = engineer_features(df, window=5)
    scenario_ids = df.loc[X.index, "scenario_id"].astype(str).values
    log.info("Feature matrix: %d rows x %d columns", X.shape[0], X.shape[1])

    bundles = load_physics_models(model_dir)
    if not bundles:
        raise SystemExit(f"No physics_sim model artifacts found under {model_dir}")
    log.info("Loaded %d physics models", len(bundles))

    actual_folds = min(args.n_folds, len(np.unique(scenario_ids)))
    if actual_folds < 2:
        raise SystemExit("Need at least 2 scenarios to evaluate transfer.")
    baseline_rocs = load_baseline_roc_aucs(baseline_summary_path)

    summary_models, fold_sizes = evaluate_transfer(
        X,
        y,
        scenario_ids,
        bundles,
        n_folds=actual_folds,
        seed=args.seed,
        baseline_rocs=baseline_rocs,
    )

    best_model_name, best_model_info = max(
        summary_models.items(),
        key=lambda item: item[1]["roc_auc"] if item[1]["roc_auc"] is not None else float("-inf"),
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(input_path),
        "model_dir": str(model_dir),
        "baseline_summary": str(baseline_summary_path) if baseline_summary_path.exists() else None,
        "n_total": int(len(df)),
        "n_scenarios": int(df["scenario_id"].nunique()),
        "n_features": int(X.shape[1]),
        "n_folds": int(actual_folds),
        "max_scenarios": args.max_scenarios,
        "seed": args.seed,
        "avg_train_rows": fold_sizes["avg_train_rows"],
        "avg_test_rows": fold_sizes["avg_test_rows"],
        "avg_train_scenarios": fold_sizes["avg_train_scenarios"],
        "avg_test_scenarios": fold_sizes["avg_test_scenarios"],
        "models": summary_models,
        "best_transfer_model": {"name": best_model_name, **best_model_info},
    }

    metrics_rows = []
    for model_name, metrics in summary_models.items():
        metrics_rows.append(
            {
                "model": model_name,
                "label": _format_model_label(model_name),
                "roc_auc": metrics.get("roc_auc"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc_cv_mean": metrics.get("roc_auc_cv_mean"),
                "roc_auc_cv_std": metrics.get("roc_auc_cv_std"),
                "source_roc_auc": metrics.get("source_roc_auc"),
                "roc_auc_gap_vs_source": metrics.get("roc_auc_gap_vs_source"),
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    pd.DataFrame(metrics_rows).sort_values(["roc_auc", "f1"], ascending=[False, False]).to_csv(
        metrics_path,
        index=False,
    )

    log.info("Summary -> %s", summary_path)
    log.info("Metrics CSV -> %s", metrics_path)
    log.info("Best transfer model: %s (ROC-AUC %.4f, F1 %.4f)",
             best_model_name, best_model_info["roc_auc"], best_model_info["f1"])


if __name__ == "__main__":
    main()
