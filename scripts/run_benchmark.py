"""
Cheap baseline benchmarks for Phase 1 datasets.
Models: majority, logistic regression, random forest, isolation forest (if no labels).
Features: mean/std/delta/rolling stats on pressure, flow_rate, temperature.

Outputs:
  reports/dataset_checks/<name>_benchmark.json
  reports/dataset_checks/<name>_benchmark.md
  models/trained/<name>_<model>.pkl

Run:
  python scripts/run_benchmark.py
  python scripts/run_benchmark.py --datasets scada_pipeline --window 5
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_adapters import normalize, save_processed
from src.data.dataset_registry import get_dataset, is_downloaded

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "dataset_checks"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "trained"
PHASE1_DATASETS = ["scada_pipeline", "water_leak"]
SENSOR_COLS = ["pressure", "flow_rate", "temperature", "vibration", "rpm", "power"]


# ── Feature engineering ──────────────────────────────────────────────────────

def make_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Build cheap time-series features on available canonical sensor columns.
    Adds per-column: rolling_mean, rolling_std, rolling_min, rolling_max, delta.
    Adds cross-column: pressure_flow_ratio (if both available).
    Does not sort by time — caller should pre-sort if needed.
    """
    feats = pd.DataFrame(index=df.index)

    present = [c for c in SENSOR_COLS if c in df.columns]

    for col in present:
        s = df[col].ffill().fillna(0)
        feats[f"{col}_roll_mean"] = s.rolling(window, min_periods=1).mean()
        feats[f"{col}_roll_std"] = s.rolling(window, min_periods=1).std().fillna(0)
        feats[f"{col}_roll_min"] = s.rolling(window, min_periods=1).min()
        feats[f"{col}_roll_max"] = s.rolling(window, min_periods=1).max()
        feats[f"{col}_delta"] = s.diff().fillna(0)
        feats[col] = s  # raw value

    if "pressure" in df.columns and "flow_rate" in df.columns:
        p = df["pressure"].fillna(0)
        f = df["flow_rate"].fillna(0)
        feats["pressure_flow_ratio"] = p / (f.replace(0, np.nan)).fillna(p.mean())

    return feats


def prepare_xy(df: pd.DataFrame, target_col: str, window: int = 5):
    """Build feature matrix and label vector. Returns X, y."""
    # Sort by timestamp if available
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.sort_values("timestamp").reset_index(drop=True)

    X = make_features(df, window=window)
    y = df[target_col].copy()

    # Binarize: if not already 0/1, map to 0/1
    unique_vals = y.dropna().unique()
    if set(unique_vals) != {0, 1}:
        # Try to infer positive class (leak=1, fault=1, etc.)
        y_lower = y.astype(str).str.lower()
        if y_lower.isin(["true", "1", "yes", "leak", "fault", "abnormal"]).any():
            y = y_lower.isin(["true", "1", "yes", "leak", "fault", "abnormal"]).astype(int)
        else:
            # Encode non-zero as 1
            try:
                y_num = pd.to_numeric(y, errors="coerce").fillna(0)
                y = (y_num != 0).astype(int)
            except Exception:
                y = y.factorize()[0]

    mask = y.notna()
    return X[mask], y[mask].astype(int)


# ── Model runners ─────────────────────────────────────────────────────────────

def run_majority(X_train, y_train, X_test, y_test) -> dict:
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return _score(y_test, y_pred, None, "majority", clf)


def run_logistic(X_train, y_train, X_test, y_test) -> dict:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else None
    return _score(y_test, y_pred, y_prob, "logistic_regression", (scaler, clf))


def run_random_forest(X_train, y_train, X_test, y_test) -> dict:
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                                  n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return _score(y_test, y_pred, y_prob, "random_forest", clf)


def run_isolation_forest(X_train, X_test, y_test) -> dict:
    clf = IsolationForest(n_estimators=100, contamination="auto", random_state=42, n_jobs=-1)
    clf.fit(X_train)
    raw = clf.predict(X_test)  # -1 anomaly, 1 normal
    y_pred = (raw == -1).astype(int)
    y_prob = -clf.score_samples(X_test)  # higher = more anomalous
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
    return _score(y_test, y_pred, y_prob, "isolation_forest", clf)


def _score(y_true, y_pred, y_prob, name: str, model) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "model": name,
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "report": report,
        "_model_obj": model,
    }


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(name: str, window: int = 5):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not is_downloaded(name):
        logger.warning("Skipping '%s' — not downloaded. See DOWNLOAD_DATA.md.", name)
        return None

    meta = get_dataset(name)
    target_col = meta.get("target")

    logger.info("=== Benchmark: %s ===", name)
    df = normalize(name)
    if df.empty:
        logger.warning("Empty DataFrame for %s", name)
        return None

    # Find target
    if target_col is None or target_col not in df.columns:
        # Try canonical leak_label
        if "leak_label" in df.columns:
            target_col = "leak_label"
            logger.info("Using canonical 'leak_label' as target for %s", name)
        else:
            logger.warning("No target column found for %s — running isolation forest only", name)
            X = make_features(df, window=window)
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            # No labels — use dummy y_test of zeros for scoring
            y_dummy = pd.Series(np.zeros(len(X_test), dtype=int))
            result = {"dataset": name, "target": None, "models": {}}
            iforest = run_isolation_forest(X_train, X_test, y_dummy)
            result["models"]["isolation_forest"] = {k: v for k, v in iforest.items() if k != "_model_obj"}
            _save_results(name, result)
            return result

    X, y = prepare_xy(df, target_col, window=window)
    if len(X) < 100:
        logger.warning("%s: too few rows (%d) after feature prep — skipping", name, len(X))
        return None

    majority_class_pct = y.value_counts(normalize=True).iloc[0]
    n_positive = int(y.sum())
    logger.info("%s: %d rows, target=%s, positives=%d (%.1f%%)",
                name, len(X), target_col, n_positive, n_positive / len(y) * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    result = {
        "dataset": name,
        "target": target_col,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test": round(float(y_test.mean()), 4),
        "models": {},
    }

    models_to_run = [
        ("majority", run_majority),
        ("logistic_regression", run_logistic),
        ("random_forest", run_random_forest),
    ]
    # Add isolation forest if labels look weak
    if majority_class_pct > 0.95:
        logger.info("Majority class > 95%% — also running isolation forest")
        models_to_run.append(("isolation_forest_unlabeled", None))

    model_objects = {}
    for model_name, runner in models_to_run:
        if model_name == "isolation_forest_unlabeled":
            r = run_isolation_forest(X_train, X_test, y_test)
        else:
            r = runner(X_train, y_train, X_test, y_test)
        model_objects[model_name] = r.pop("_model_obj", None)
        result["models"][model_name] = r
        logger.info("  %s: acc=%.4f f1=%.4f auc=%s",
                    model_name, r["accuracy"], r["f1"],
                    f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A")

    # Save model artifacts
    for model_name, obj in model_objects.items():
        if obj is not None:
            pkl_path = MODELS_DIR / f"{name}_{model_name}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(obj, f)

    # Save features
    feat_df = X.copy()
    feat_df["target"] = y.values
    save_processed(feat_df, name, suffix="features")

    _save_results(name, result)
    return result


def _save_results(name: str, result: dict):
    json_path = REPORTS_DIR / f"{name}_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Benchmark JSON → %s", json_path)

    md_path = REPORTS_DIR / f"{name}_benchmark.md"
    _write_benchmark_md(result, md_path)


def _write_benchmark_md(result: dict, out_path: Path):
    name = result["dataset"]
    n_train = result.get('n_train')
    n_test = result.get('n_test')
    pos_rate = result.get('positive_rate_test')
    lines = [
        f"# Benchmark: {name}",
        "",
        f"**Target:** `{result.get('target', 'N/A')}`  |  "
        f"**Train:** {n_train:,}  |  "
        f"**Test:** {n_test:,}" if (n_train and n_test) else f"**Target:** `{result.get('target', 'N/A')}`",
        f"**Positive rate (test):** {pos_rate:.1%}" if pos_rate is not None else "",
        "",
        "## Model Results",
        "",
        "| Model | Accuracy | F1 | ROC-AUC |",
        "|-------|----------|----|---------|",
    ]
    for mname, metrics in result.get("models", {}).items():
        auc = f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") else "N/A"
        lines.append(f"| {mname} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} | {auc} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Benchmark MD → %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Run cheap baselines on Phase 1 datasets")
    parser.add_argument("--datasets", nargs="+", default=PHASE1_DATASETS)
    parser.add_argument("--window", type=int, default=5, help="Rolling window size")
    args = parser.parse_args()

    results = {}
    for name in args.datasets:
        r = run_benchmark(name, window=args.window)
        if r:
            results[name] = r

    if not results:
        print("\nNo datasets available. See DOWNLOAD_DATA.md.")
        return

    print("\n=== BENCHMARK SUMMARY ===")
    for name, r in results.items():
        print(f"\n{name}:")
        for mname, m in r.get("models", {}).items():
            auc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A"
            print(f"  {mname:30s} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={auc}")


if __name__ == "__main__":
    main()
