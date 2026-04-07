import json

import pandas as pd

from scripts.evaluate_physics_transfer import (
    aggregate_fold_metrics,
    load_baseline_roc_aucs,
    sample_scenarios,
)


def test_aggregate_fold_metrics_uses_cv_prefixes():
    summary = aggregate_fold_metrics(
        {
            "random_forest": [
                {"roc_auc": 0.91, "accuracy": 0.81, "precision": 0.82, "recall": 0.8, "f1": 0.81},
                {"roc_auc": 0.89, "accuracy": 0.79, "precision": 0.8, "recall": 0.78, "f1": 0.79},
            ]
        }
    )

    assert summary["random_forest"]["roc_auc_cv_mean"] == 0.9
    assert summary["random_forest"]["roc_auc_cv_std"] == 0.01
    assert summary["random_forest"]["f1_cv_mean"] == 0.8


def test_load_baseline_roc_aucs_reads_models(tmp_path):
    summary_path = tmp_path / "physics_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "models": {
                    "random_forest": {"roc_auc": 0.98},
                    "xgboost": {"roc_auc": 0.97},
                }
            }
        ),
        encoding="utf-8",
    )

    baselines = load_baseline_roc_aucs(summary_path)

    assert baselines == {"random_forest": 0.98, "xgboost": 0.97}


def test_sample_scenarios_limits_whole_scenarios():
    df = pd.DataFrame(
        {
            "scenario_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "target": [0, 0, 1, 1, 0, 0, 1, 1],
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="min"),
        }
    )

    sampled = sample_scenarios(df, max_scenarios=2, seed=42)

    assert sampled["scenario_id"].nunique() == 2
    assert set(sampled["scenario_id"]).issubset({"a", "b", "c", "d"})
    assert sampled.groupby("scenario_id").size().tolist() == [2, 2]
