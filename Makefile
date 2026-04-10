# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Leak Detection — Task Runner
# ──────────────────────────────────────────────────────────────────────────────
# Usage:  make <target>
#
# On Windows without GNU Make, use:  python tasks.py <target>
# ──────────────────────────────────────────────────────────────────────────────

ifeq ($(OS),Windows_NT)
	PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard venv/Scripts/python.exe),venv/Scripts/python.exe,python))
	STREAMLIT := $(if $(wildcard .venv/Scripts/streamlit.exe),.venv/Scripts/streamlit.exe,$(if $(wildcard venv/Scripts/streamlit.exe),venv/Scripts/streamlit.exe,$(PYTHON) -m streamlit))
	CLEAN_CACHE_CMD := for /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	CLEAN_PYTEST_CACHE_CMD := if exist .pytest_cache rd /s /q .pytest_cache
	CLEAN_PHYSICS_DATA_CMD := if exist data\physics_sim rd /s /q data\physics_sim
	CLEAN_REALTIME_DATA_CMD := if exist data\realtime rd /s /q data\realtime
else
	PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,$(if $(wildcard venv/bin/python),venv/bin/python,python3))
	STREAMLIT := $(if $(wildcard .venv/bin/streamlit),.venv/bin/streamlit,$(if $(wildcard venv/bin/streamlit),venv/bin/streamlit,$(PYTHON) -m streamlit))
	CLEAN_CACHE_CMD := find . -type d -name __pycache__ -prune -exec rm -rf {} +
	CLEAN_PYTEST_CACHE_CMD := rm -rf .pytest_cache
	CLEAN_PHYSICS_DATA_CMD := rm -rf data/physics_sim
	CLEAN_REALTIME_DATA_CMD := rm -rf data/realtime
endif

.DEFAULT_GOAL := help

# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: help install create-venv

help: ## Show this help
	@echo
	@echo "  Pipeline Leak Detection — Available Targets"
	@echo "  ============================================"
ifeq ($(OS),Windows_NT)
	@findstr /R "^[a-zA-Z_-]*:.*##" Makefile
else
	@grep -E "^[a-zA-Z_-]+:.*##" Makefile
endif

create-venv: ## Create Python virtual environment
	python -m venv venv
	$(PYTHON) -m pip install --upgrade pip

install: ## Install all dependencies from requirements.txt
	$(PYTHON) -m pip install -r requirements.txt

# ── Data Pipeline ─────────────────────────────────────────────────────────────

.PHONY: data-download data-eda data-benchmark petrobras-download petrobras-quick petrobras-train petrobras-all

data-download: ## Download Phase 1 datasets via Kaggle API
	$(PYTHON) scripts/download_data.py --datasets scada_pipeline water_leak

data-eda: ## Run exploratory data analysis on Phase 1 datasets
	$(PYTHON) scripts/run_eda.py --datasets scada_pipeline water_leak

data-benchmark: ## Run baseline benchmarks on Phase 1 datasets
	$(PYTHON) scripts/run_benchmark.py --datasets scada_pipeline water_leak --window 5

petrobras-download: ## Download & process Petrobras 3W dataset (~1.8 GB)
	$(PYTHON) scripts/download_petrobras_3w.py

petrobras-quick: ## Download & process Petrobras 3W (100 files only)
	$(PYTHON) scripts/download_petrobras_3w.py --max-files 100

petrobras-train: ## Train models on Petrobras 3W real oil well data
	$(PYTHON) scripts/train_petrobras_models.py

petrobras-all: petrobras-download petrobras-train ## Download 3W + train Petrobras models

# ── Physics Simulator ─────────────────────────────────────────────────────────

.PHONY: physics-generate physics-train physics-validate physics-all

physics-generate: ## Generate physics-based synthetic SCADA dataset (200 scenarios)
	$(PYTHON) scripts/generate_physics_dataset.py --n-normal 100 --n-leak 100 --duration 120

physics-train: ## Train 5 model types on physics simulator data
	$(PYTHON) scripts/train_physics_models.py

physics-validate: ## Run physics sanity checks on simulator output
	$(PYTHON) -c "from src.physics_sim.validate import run_all_checks; from src.physics_sim import SimConfig, PipelineSimulator; s=PipelineSimulator(SimConfig()); sol=s.run(); run_all_checks(sol, s.cfg); print('All physics checks passed')"

physics-all: physics-generate physics-train ## Generate data + train physics models

# ── Realtime Models ───────────────────────────────────────────────────────────

.PHONY: realtime-generate realtime-train realtime-all

realtime-generate: ## Generate realtime training data from simulator
	$(PYTHON) scripts/generate_realtime_training_data.py

realtime-train: ## Train realtime models (RF, XGB, LGB, IF, Hybrid)
	$(PYTHON) scripts/train_realtime_models.py --config config/realtime_training.yaml

realtime-all: realtime-generate realtime-train ## Generate data + train realtime models

# ── Robust Models ─────────────────────────────────────────────────────────────

.PHONY: robust-corpus robust-train robust-all

robust-corpus: ## Build robust training corpus (simulator + external data)
	$(PYTHON) scripts/build_robust_training_data.py

robust-train: ## Train robust live-safe fusion models
	$(PYTHON) scripts/train_robust_models.py --config config/robust_training.yaml

robust-all: robust-corpus robust-train ## Build corpus + train robust models

# ── Evaluation ────────────────────────────────────────────────────────────────

.PHONY: eval evaluate physics-transfer

eval: ## Evaluate all models with scenario metrics and calibrate thresholds
	$(PYTHON) scripts/evaluate_models.py

evaluate: eval ## Alias for eval

physics-transfer: ## Evaluate physics-trained models on Petrobras scenarios
	$(PYTHON) scripts/evaluate_physics_transfer.py

# ── Dashboard ─────────────────────────────────────────────────────────────────

.PHONY: dashboard

dashboard: ## Launch Streamlit dashboard
	$(STREAMLIT) run dashboard/app.py --server.port 8510

# ── Testing ───────────────────────────────────────────────────────────────────

.PHONY: test test-v test-fast

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-v: ## Run tests with verbose output and stdout
	$(PYTHON) -m pytest tests/ -v -s

test-fast: ## Run tests, stop on first failure
	$(PYTHON) -m pytest tests/ -x -v

# ── Cleanup ───────────────────────────────────────────────────────────────────

.PHONY: clean clean-models clean-data clean-all

clean: ## Remove Python cache files
	@$(CLEAN_CACHE_CMD)
	@$(CLEAN_PYTEST_CACHE_CMD)

clean-models: ## Remove all trained model artifacts
	@if exist models\realtime\*.joblib del /q models\realtime\*.joblib
	@if exist models\physics_sim\*.joblib del /q models\physics_sim\*.joblib
	@if exist models\robust\*.joblib del /q models\robust\*.joblib

clean-data: ## Remove generated datasets (keeps raw downloads)
	@$(CLEAN_PHYSICS_DATA_CMD)
	@$(CLEAN_REALTIME_DATA_CMD)

clean-all: clean clean-models clean-data ## Remove everything generated

# ── Full Pipelines ────────────────────────────────────────────────────────────

.PHONY: train-all pipeline-full retrain-all

train-all: physics-all physics-transfer realtime-all robust-all ## Train all model families
	@echo All model families trained.

retrain-all: physics-all physics-transfer realtime-all robust-all petrobras-train eval ## Regenerate data, retrain models, run transfer eval, and evaluate
	@echo Full retrain complete.

pipeline-full: train-all eval ## Train all models + evaluate
	@echo Full pipeline complete.
