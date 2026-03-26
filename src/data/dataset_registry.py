"""
Dataset registry — loads config/datasets.yaml and exposes dataset metadata.
Does not touch existing loader.py or its schema validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "config" / "datasets.yaml"


def load_registry(config_path: Optional[str] = None) -> Dict:
    """Return full datasets dict from YAML config."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)["datasets"]


def get_dataset(name: str, config_path: Optional[str] = None) -> Dict:
    """Return metadata for a single dataset by key."""
    registry = load_registry(config_path)
    if name not in registry:
        raise KeyError(f"Dataset '{name}' not found. Available: {list(registry)}")
    meta = registry[name].copy()
    meta["key"] = name
    # Resolve paths relative to project root
    meta["local_raw"] = str(_ROOT / meta["local_raw"])
    meta["local_processed"] = str(_ROOT / meta["local_processed"])
    return meta


def _candidate_raw_paths(meta: Dict) -> list[Path]:
    raw_dir = Path(meta["local_raw"])
    if not raw_dir.exists():
        return []

    recursive = bool(meta.get("recursive", False))
    file_globs = meta.get("file_globs") or ["*.csv"]

    candidates: list[Path] = []
    for pattern in file_globs:
        matcher = raw_dir.rglob if recursive else raw_dir.glob
        candidates.extend(sorted(path for path in matcher(pattern) if path.is_file()))
    return candidates


def get_phase_datasets(phase: int, config_path: Optional[str] = None) -> List[Dict]:
    """Return metadata for all datasets at a given phase."""
    registry = load_registry(config_path)
    result = []
    for key, meta in registry.items():
        if meta.get("phase") == phase:
            m = meta.copy()
            m["key"] = key
            m["local_raw"] = str(_ROOT / meta["local_raw"])
            m["local_processed"] = str(_ROOT / meta["local_processed"])
            result.append(m)
    return result


def raw_csv_path(name: str, config_path: Optional[str] = None) -> Optional[str]:
    """Return path to the first raw dataset file if it exists, else None."""
    meta = get_dataset(name, config_path)
    candidates = _candidate_raw_paths(meta)
    return str(candidates[0]) if candidates else None


def raw_data_paths(name: str, config_path: Optional[str] = None) -> List[str]:
    """Return all matching raw dataset files for a dataset."""
    meta = get_dataset(name, config_path)
    return [str(path) for path in _candidate_raw_paths(meta)]


def is_downloaded(name: str, config_path: Optional[str] = None) -> bool:
    return raw_csv_path(name, config_path) is not None
