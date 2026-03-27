from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib


@dataclass(frozen=True)
class ModelArtifact:
    label: str
    path: str
    dataset_type: str
    source: str
    priority: int


def build_model_artifact_path(
    model_dir: str | Path,
    model_name: str,
    dataset_name: str | None = None,
    extension: str = ".joblib",
) -> Path:
    """
    Build a standardized artifact path for a trained model.
    """
    root = Path(model_dir)
    suffix = extension if extension.startswith(".") else f".{extension}"
    stem = f"{dataset_name}_{model_name}" if dataset_name else model_name
    return root / f"{stem}{suffix}"


def save_model_artifact(
    model,
    model_dir: str | Path,
    model_name: str,
    dataset_name: str | None = None,
    extension: str = ".joblib",
) -> str:
    """
    Save a model artifact using the repo-standard serialization format.
    """
    output_path = build_model_artifact_path(
        model_dir=model_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        extension=extension,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return str(output_path)


def _normalize_label(stem: str) -> str:
    if stem.startswith("robust_realtime_realtime_"):
        stem = f"robust_{stem[len('robust_realtime_realtime_'):]}"
    for prefix in ("scada_pipeline_", "water_leak_", "robust_realtime_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem.replace("_", " ").title()


def _infer_dataset_type(path: Path) -> str | None:
    stem = path.stem
    if stem.startswith("water_leak_"):
        return "water_leak"
    if stem.startswith("scada_pipeline_"):
        return "scada"
    if stem.startswith("robust_realtime_"):
        return "scada"
    if stem.startswith("physics_sim_"):
        return "scada"
    if path.parent.name == "realtime":
        return "scada"
    if path.parent.name == "robust":
        return "scada"
    if path.parent.name == "physics_sim":
        return "scada"
    if path.parent.name == "advanced":
        return "scada"
    if stem in {"logistic_regression", "random_forest"}:
        return "scada"
    return None


def _artifact_priority(path: Path) -> int:
    stem = path.stem
    if path.parent.name == "robust":
        return 0
    if path.parent.name == "realtime":
        return 1
    if path.parent.name == "physics_sim":
        return 2
    if path.parent.name == "advanced":
        return 3
    if stem.startswith(("scada_pipeline_", "water_leak_")) and path.suffix == ".joblib":
        return 4
    if stem.startswith(("scada_pipeline_", "water_leak_")):
        return 5
    if path.parent.name == "trained":
        return 6
    return 7


def discover_model_artifacts(model_dir: str | Path, dataset_type: str) -> dict[str, ModelArtifact]:
    """
    Discover dashboard-loadable artifacts and choose one best artifact per model label.

    Priority order:
    1. models/robust/*.joblib
    2. models/realtime/*.joblib
    3. models/physics_sim/*.joblib
    4. models/advanced/*.joblib
    5. dataset-specific models/trained/<dataset>_*.joblib
    6. dataset-specific models/trained/<dataset>_*.pkl
    7. models/trained/*.joblib
    8. models/*.joblib
    """
    root = Path(model_dir)
    candidates = sorted(root.glob("*.joblib"))
    candidates += sorted((root / "trained").glob("*.joblib"))
    candidates += sorted((root / "trained").glob("*.pkl"))
    candidates += sorted((root / "advanced").glob("*.joblib"))
    candidates += sorted((root / "physics_sim").glob("*.joblib"))
    candidates += sorted((root / "realtime").glob("*.joblib"))
    candidates += sorted((root / "robust").glob("*.joblib"))

    selected: dict[str, ModelArtifact] = {}
    for path in candidates:
        if path.stem.endswith("_scaler"):
            continue
        inferred_dataset = _infer_dataset_type(path)
        if inferred_dataset != dataset_type:
            continue

        artifact = ModelArtifact(
            label=_normalize_label(path.stem),
            path=str(path),
            dataset_type=inferred_dataset,
            source=str(path.parent.relative_to(root.parent) if path.parent != root else root.name),
            priority=_artifact_priority(path),
        )
        current = selected.get(artifact.label)
        if current is None or artifact.priority < current.priority:
            selected[artifact.label] = artifact

    return dict(sorted(selected.items()))
