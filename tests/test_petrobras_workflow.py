import json
from pathlib import Path

from scripts.compare_petrobras_runs import build_comparison_frame, collect_summary_paths, summary_to_rows
from scripts.download_petrobras_3w import resolve_dataset_dir, resolve_processing_summary_path
from scripts.train_petrobras_models import aggregate_cv_metrics, resolve_output_paths


def test_resolve_dataset_dir_supports_raw_root(tmp_path):
    dataset_dir = tmp_path / "petrobras_3w" / "3W" / "dataset"
    dataset_dir.mkdir(parents=True)

    assert resolve_dataset_dir(tmp_path / "petrobras_3w") == dataset_dir


def test_resolve_processing_summary_path_defaults_next_to_output_csv(tmp_path):
    output_csv = tmp_path / "runs" / "petrobras_3w_scada.csv"

    summary_path = resolve_processing_summary_path(output_csv, None)

    assert summary_path == output_csv.with_name("petrobras_3w_processing_summary.json")


def test_resolve_output_paths_default_under_output_dir(tmp_path):
    output_dir = tmp_path / "artifacts" / "run_01" / "models"

    summary_path, metrics_path = resolve_output_paths(output_dir, None, None)

    assert summary_path == output_dir / "petrobras_training_summary.json"
    assert metrics_path == output_dir / "petrobras_model_metrics.csv"


def test_summary_to_rows_marks_best_model():
    summary = {
        "run_label": "run_01",
        "dataset": "petrobras.csv",
        "n_total": 100,
        "n_train": 80,
        "n_test": 20,
        "label_distribution": {"0": 40, "1": 60},
        "best_model": {"name": "lightgbm", "roc_auc": 0.95, "f1": 0.9},
        "models": {
            "lightgbm": {"roc_auc": 0.95, "accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9},
            "random_forest": {"roc_auc": 0.93, "accuracy": 0.88, "precision": 0.87, "recall": 0.89, "f1": 0.88},
        },
    }

    rows = summary_to_rows(summary, Path("summary.json"))

    assert len(rows) == 2
    assert any(row["model"] == "lightgbm" and row["is_best_model"] for row in rows)
    assert any(row["model"] == "random_forest" and not row["is_best_model"] for row in rows)


def test_collect_summary_paths_deduplicates_glob_and_explicit(tmp_path, monkeypatch):
    summary_path = tmp_path / "run_01" / "models" / "petrobras_training_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    paths = collect_summary_paths("run_01/models/*.json", [str(summary_path)])

    assert paths == [Path(str(summary_path))]


def test_build_comparison_frame_loads_rows(tmp_path):
    summary_path = tmp_path / "run_01" / "models" / "petrobras_training_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "run_label": "run_01",
                "dataset": "petrobras.csv",
                "n_total": 100,
                "n_train": 80,
                "n_test": 20,
                "label_distribution": {"0": 40, "1": 60},
                "models": {
                    "lightgbm": {
                        "roc_auc": 0.95,
                        "accuracy": 0.9,
                        "precision": 0.9,
                        "recall": 0.9,
                        "f1": 0.9,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    frame = build_comparison_frame([summary_path])

    assert list(frame["model"]) == ["lightgbm"]
    assert frame.iloc[0]["run_label"] == "run_01"


def test_aggregate_cv_metrics_computes_mean_and_std():
    summary = aggregate_cv_metrics(
        {
            "random_forest": [
                {"roc_auc": 0.9, "accuracy": 0.8, "precision": 0.81, "recall": 0.79, "f1": 0.8},
                {"roc_auc": 0.8, "accuracy": 0.7, "precision": 0.71, "recall": 0.69, "f1": 0.7},
            ]
        }
    )

    assert summary["random_forest"]["roc_auc"] == 0.85
    assert summary["random_forest"]["accuracy"] == 0.75
    assert summary["random_forest"]["roc_auc_std"] == 0.05
