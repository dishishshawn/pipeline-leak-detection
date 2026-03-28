#!/usr/bin/env python
"""
Task runner for Pipeline Leak Detection project.

Usage:  python tasks.py <task>
        python tasks.py --list
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# ── Resolve venv Python ──────────────────────────────────────────────────────

if sys.platform == "win32":
    PYTHON = str(ROOT / "venv" / "Scripts" / "python.exe")
    STREAMLIT = str(ROOT / "venv" / "Scripts" / "streamlit.exe")
else:
    PYTHON = str(ROOT / "venv" / "bin" / "python")
    STREAMLIT = str(ROOT / "venv" / "bin" / "streamlit")


def run(*args: str, check: bool = True) -> int:
    """Run a command, print it, and return the exit code."""
    cmd = " ".join(args)
    print(f"\n>>> {cmd}")
    result = subprocess.run(args, cwd=str(ROOT))
    if check and result.returncode != 0:
        print(f"\nFAILED (exit code {result.returncode})")
        sys.exit(result.returncode)
    return result.returncode


# ── Task Registry ────────────────────────────────────────────────────────────

TASKS: dict[str, tuple[callable, str]] = {}


def task(name: str, description: str):
    """Decorator to register a task."""
    def decorator(fn):
        TASKS[name] = (fn, description)
        return fn
    return decorator


# ── Setup ────────────────────────────────────────────────────────────────────

@task("install", "Install all dependencies from requirements.txt")
def install():
    run(PYTHON, "-m", "pip", "install", "-r", "requirements.txt")


# ── Data Pipeline ────────────────────────────────────────────────────────────

@task("data-download", "Download Phase 1 datasets via Kaggle API")
def data_download():
    run(PYTHON, "scripts/download_data.py", "--datasets", "scada_pipeline", "water_leak")


@task("data-eda", "Run exploratory data analysis on Phase 1 datasets")
def data_eda():
    run(PYTHON, "scripts/run_eda.py", "--datasets", "scada_pipeline", "water_leak")


@task("data-benchmark", "Run baseline benchmarks on Phase 1 datasets")
def data_benchmark():
    run(PYTHON, "scripts/run_benchmark.py", "--datasets", "scada_pipeline", "water_leak", "--window", "5")


@task("petrobras-download", "Download & process Petrobras 3W dataset (~1.8 GB)")
def petrobras_download():
    run(PYTHON, "scripts/download_petrobras_3w.py")


@task("petrobras-quick", "Download & process Petrobras 3W (100 files only, for quick test)")
def petrobras_quick():
    run(PYTHON, "scripts/download_petrobras_3w.py", "--max-files", "100")


@task("petrobras-train", "Train models on Petrobras 3W real oil well data")
def petrobras_train():
    run(PYTHON, "scripts/train_petrobras_models.py")


@task("petrobras-all", "Download 3W data + train Petrobras models")
def petrobras_all():
    petrobras_download()
    petrobras_train()


# ── Physics Simulator ────────────────────────────────────────────────────────

@task("physics-generate", "Generate physics-based synthetic SCADA dataset (200 scenarios)")
def physics_generate():
    run(PYTHON, "scripts/generate_physics_dataset.py", "--n-normal", "100", "--n-leak", "100", "--duration", "120")


@task("physics-train", "Train 5 model types on physics simulator data")
def physics_train():
    run(PYTHON, "scripts/train_physics_models.py")


@task("physics-validate", "Run physics sanity checks on simulator output")
def physics_validate():
    run(PYTHON, "-c",
        "from src.physics_sim.validate import run_all_checks; "
        "from src.physics_sim import SimConfig, PipelineSimulator; "
        "s=PipelineSimulator(SimConfig()); sol=s.run(); "
        "run_all_checks(sol, s.cfg); print('All physics checks passed')")


@task("physics-all", "Generate data + train physics models")
def physics_all():
    physics_generate()
    physics_train()


# ── Realtime Models ──────────────────────────────────────────────────────────

@task("realtime-generate", "Generate realtime training data from simulator")
def realtime_generate():
    run(PYTHON, "scripts/generate_realtime_training_data.py")


@task("realtime-train", "Train realtime models (RF, XGB, LGB, IF, Hybrid)")
def realtime_train():
    run(PYTHON, "scripts/train_realtime_models.py", "--config", "config/realtime_training.yaml")


@task("realtime-all", "Generate data + train realtime models")
def realtime_all():
    realtime_generate()
    realtime_train()


# ── Robust Models ────────────────────────────────────────────────────────────

@task("robust-corpus", "Build robust training corpus (simulator + external data)")
def robust_corpus():
    run(PYTHON, "scripts/build_robust_training_data.py")


@task("robust-train", "Train robust live-safe fusion models")
def robust_train():
    run(PYTHON, "scripts/train_robust_models.py", "--config", "config/robust_training.yaml")


@task("robust-all", "Build corpus + train robust models")
def robust_all():
    robust_corpus()
    robust_train()


# ── Evaluation ───────────────────────────────────────────────────────────────

@task("eval", "Evaluate all models with scenario metrics and calibrate thresholds")
def evaluate():
    run(PYTHON, "scripts/evaluate_models.py")


# ── Dashboard ────────────────────────────────────────────────────────────────

@task("dashboard", "Launch Streamlit dashboard on port 8510")
def dashboard():
    run(STREAMLIT, "run", "dashboard/app.py", "--server.port", "8510")


# ── Testing ──────────────────────────────────────────────────────────────────

@task("test", "Run all tests")
def test():
    run(PYTHON, "-m", "pytest", "tests/", "-v")


@task("test-fast", "Run tests, stop on first failure")
def test_fast():
    run(PYTHON, "-m", "pytest", "tests/", "-x", "-v")


# ── Cleanup ──────────────────────────────────────────────────────────────────

@task("clean", "Remove Python cache files")
def clean():
    import shutil
    for cache_dir in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    pytest_cache = ROOT / ".pytest_cache"
    if pytest_cache.exists():
        shutil.rmtree(pytest_cache, ignore_errors=True)
    print("Cleaned cache files.")


@task("clean-models", "Remove all trained model artifacts")
def clean_models():
    for folder in ["models/realtime", "models/physics_sim", "models/robust"]:
        p = ROOT / folder
        if p.exists():
            for f in p.glob("*.joblib"):
                f.unlink()
                print(f"  Removed {f.relative_to(ROOT)}")
    print("Cleaned model artifacts.")


@task("clean-data", "Remove generated datasets (keeps raw downloads)")
def clean_data():
    import shutil
    for folder in ["data/physics_sim", "data/realtime"]:
        p = ROOT / folder
        if p.exists():
            shutil.rmtree(p)
            print(f"  Removed {folder}/")
    print("Cleaned generated data.")


@task("clean-all", "Remove everything generated (cache + models + data)")
def clean_all():
    clean()
    clean_models()
    clean_data()


# ── Full Pipelines ───────────────────────────────────────────────────────────

@task("train-all", "Train all model families (physics + realtime + robust)")
def train_all():
    physics_all()
    realtime_all()
    robust_all()
    print("\nAll model families trained.")


@task("pipeline-full", "Train all models + evaluate (full pipeline)")
def pipeline_full():
    train_all()
    evaluate()
    print("\nFull pipeline complete.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def print_help():
    print("\nPipeline Leak Detection — Task Runner")
    print("=" * 55)
    print(f"\nUsage:  python tasks.py <task>\n")

    # Group tasks by category
    categories = {
        "Setup":      ["install"],
        "Data":       ["data-download", "data-eda", "data-benchmark"],
        "Petrobras":  ["petrobras-download", "petrobras-quick", "petrobras-train", "petrobras-all"],
        "Physics":    ["physics-generate", "physics-train", "physics-validate", "physics-all"],
        "Realtime":   ["realtime-generate", "realtime-train", "realtime-all"],
        "Robust":     ["robust-corpus", "robust-train", "robust-all"],
        "Evaluation": ["eval"],
        "Dashboard":  ["dashboard"],
        "Testing":    ["test", "test-fast"],
        "Cleanup":    ["clean", "clean-models", "clean-data", "clean-all"],
        "Pipelines":  ["train-all", "pipeline-full"],
    }

    for category, task_names in categories.items():
        print(f"  {category}:")
        for name in task_names:
            if name in TASKS:
                _, desc = TASKS[name]
                print(f"    {name:<22s} {desc}")
        print()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "help", "--list"):
        print_help()
        return

    task_name = sys.argv[1]
    if task_name not in TASKS:
        print(f"Unknown task: {task_name}")
        print(f"Run 'python tasks.py --list' to see available tasks.")
        sys.exit(1)

    fn, desc = TASKS[task_name]
    print(f"\n{'=' * 55}")
    print(f"  {desc}")
    print(f"{'=' * 55}")
    fn()


if __name__ == "__main__":
    main()
