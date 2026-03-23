from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    label: str
    path: str
    dataset_type: str
    source: str
    priority: int


def _normalize_label(stem: str) -> str:
    for prefix in ("scada_pipeline_", "water_leak_"):
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
    if path.parent.name == "advanced":
        return "scada"
    if stem in {"logistic_regression", "random_forest"}:
        return "scada"
    return None


def _artifact_priority(path: Path) -> int:
    stem = path.stem
    if path.parent.name == "advanced":
        return 0
    if stem.startswith(("scada_pipeline_", "water_leak_")):
        return 1
    if path.parent.name == "trained":
        return 2
    return 3


def discover_model_artifacts(model_dir: str | Path, dataset_type: str) -> dict[str, ModelArtifact]:
    """
    Discover dashboard-loadable artifacts and choose one best artifact per model label.

    Priority order:
    1. models/advanced/*.joblib
    2. dataset-specific models/trained/<dataset>_*.pkl
    3. models/trained/*.joblib
    4. models/*.joblib
    """
    root = Path(model_dir)
    candidates = sorted(root.glob("*.joblib"))
    candidates += sorted((root / "trained").glob("*.joblib"))
    candidates += sorted((root / "trained").glob("*.pkl"))
    candidates += sorted((root / "advanced").glob("*.joblib"))

    selected: dict[str, ModelArtifact] = {}
    for path in candidates:
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
