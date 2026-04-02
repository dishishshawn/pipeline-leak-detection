# Pipeline Leak Detection System

Pipeline leak detection and monitoring project built around SCADA-style sensor data, supervised ML models, anomaly detection, and a Streamlit dashboard with a live simulator.

For a non-technical overview your team can study before investor conversations, see `TEAM_BRIEFING.md`.

The repository now supports three complementary workflows:
- offline benchmarking and evaluation on stored datasets
- advanced model training with XGBoost/LightGBM and experiment tracking hooks
- realtime-oriented training plus a live simulator for stress-testing models against synthetic incidents

## Project Goals

- analyze pipeline telemetry with time-series feature engineering
- detect leaks with multiple model families, not just one classifier
- compare supervised and anomaly-based approaches side by side
- simulate live operational incidents for dashboard-based testing
- keep training paths configurable so newer data can replace current datasets later

## Technology Stack

| Component | Technology |
| --- | --- |
| Language | Python 3.8+ |
| Data Processing | pandas, numpy |
| ML | scikit-learn, XGBoost, LightGBM |
| Tracking / Tuning | MLflow, Optuna |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Configuration | YAML |

## Repository Structure

```text
pipeline-leak-detection/
├── config/
│   ├── datasets.yaml
│   ├── realtime_training.yaml
│   └── training_config.yaml
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   │   ├── scada_pipeline/
│   │   │   └── scada_pipeline.csv
│   │   └── water_leak/
│   │       └── water_leak_detection_1000_rows.csv
│   ├── processed/
│   └── sample/
│       └── scada_sample.csv
├── notebooks/
│   ├── 01_setup_cloud.ipynb
│   ├── 02_prepare_petrobras.ipynb
│   ├── 03_train_petrobras.ipynb
│   └── 04_compare_petrobras_runs.ipynb
├── models/
│   ├── advanced/
│   ├── realtime/
│   ├── trained/
│   └── ...
├── reports/
│   └── dataset_checks/
├── scripts/
│   ├── download_data.py
│   ├── run_benchmark.py
│   ├── run_eda.py
│   ├── train_advanced.py
│   └── train_realtime_models.py
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── artifacts.py
│   │   ├── predict.py
│   │   ├── realtime.py
│   │   ├── realtime_train.py
│   │   └── train.py
│   └── simulation/
│       ├── core.py
│       └── scenarios.py
└── tests/
```

## Setup

### 1. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Data

Included data sources:

- `data/sample/scada_sample.csv`: small SCADA sample for quick local checks
- `data/sample/realtime_training_data.csv`: simulator-generated live-training corpus
- `data/raw/scada_pipeline/scada_pipeline.csv`: alternate raw SCADA dataset retained for dataset exploration
- `data/raw/water_leak/water_leak_detection_1000_rows.csv`: alternate water leak dataset used by the benchmark/dashboard path

The dataset registry in `config/datasets.yaml` now also includes integration entries for:
- `mendeley_water_testbed`
- `petrobras_3w`
- `phmsa_pipeline_incidents`
- `usdot_pipeline_accidents`

Those external sources are integrated with explicit intended roles:
- telemetry-compatible leak datasets can feed the robust live-training corpus when their raw files are present locally
- PHMSA / USDOT incident data is integrated for future asset-risk work, not live leak scoring

For non-Kaggle sources, `python scripts/download_data.py --datasets <name>` now creates the expected raw-data directory and prints the upstream source URL so you can drop the files into the correct place.

### Petrobras 3W Dataset (Real Oil Well Telemetry)

The [Petrobras 3W dataset](https://github.com/petrobras/3W) contains ~1,984 parquet files (~1.8 GB) of real oil well telemetry from Petrobras. It includes 10 event types (normal operation + 9 fault categories including hydrates, slugging, BSW increase, etc.) recorded at 1 Hz from downhole and Xmas-tree sensors.

**The raw data is NOT checked into git** (`data/raw/` is gitignored). To download and process it locally:

```bash
python tasks.py petrobras-download      # Full dataset (~1.8 GB, takes a few minutes)
python tasks.py petrobras-quick          # Quick test with 100 files only
python scripts/download_petrobras_3w.py --skip-download  # Rebuild CSV from existing files
```

This will:
1. Clone the 3W repo to `data/raw/petrobras_3w/3W/` (git depth=1)
2. Map 3W sensor columns to our SCADA schema (P-PDG → P_inlet, P-TPT → P_mid, T-PDG → T_inlet, QGL → Q_inlet)
3. Map events 1-9 to target=1 (anomaly) and event 0 to target=0 (normal)
4. Downsample from 1 Hz to 0.1 Hz for manageable size
5. Export to `data/processed/petrobras_3w/petrobras_3w_scada.csv`

For larger cloud runs, use the notebook workflow:

1. `notebooks/01_setup_cloud.ipynb`
2. `notebooks/02_prepare_petrobras.ipynb`
3. `notebooks/03_train_petrobras.ipynb`
4. `notebooks/04_compare_petrobras_runs.ipynb`

Those notebooks call the repo scripts directly. The cloud-friendly CLI entry points are:

```bash
python scripts/download_petrobras_3w.py --raw-dir <path> --output-csv <path> --summary-json <path>
python scripts/train_petrobras_models.py --input <csv> --output <dir> --summary-json <path> --metrics-csv <path> --run-label <name>
python scripts/compare_petrobras_runs.py --summary-glob "<glob>" --output-csv <path>
```

Notebook and cloud outputs should live under `artifacts/`, which is gitignored.

**You do NOT need this dataset to run the dashboard or train models.** The simulator and existing sample data are sufficient for development. The 3W data is for training higher-quality models on real-world sensor distributions.

The training code is config-driven, so replacing `data.path` in the YAML configs is the intended way to retrain on newer, higher-quality datasets later.

## Core Pipeline

### 1. Data Loading

[`src/data/loader.py`](src/data/loader.py) handles the SCADA-schema path:
- file loading
- required-column validation
- duplicate removal
- timestamp parsing
- dropping rows with missing critical values

[`src/data/dataset_registry.py`](src/data/dataset_registry.py) and [`src/data/dataset_adapters.py`](src/data/dataset_adapters.py) provide a more general dataset-mapping layer for non-SCADA sources.

### 2. Feature Engineering

[`src/features/engineer.py`](src/features/engineer.py) builds the current feature set:
- pressure delta
- flow-rate delta
- pressure and flow percent deltas
- rolling pressure mean/std
- rolling flow mean/std
- rolling pressure/flow z-scores
- pressure-to-flow ratio
- encoded event type

### 3. Prediction Compatibility

[`src/models/predict.py`](src/models/predict.py) provides a compatibility layer so the dashboard can score:
- standard sklearn classifiers
- tuple artifacts like `(scaler, estimator)`
- anomaly-style models using `score_samples`
- ensemble wrappers that expose `predict_proba`

## Training Workflows

### Basic Training

Baseline logistic regression and random forest:

```bash
python -m src.models.train
```

Artifacts are saved under `models/trained/`.

### Advanced Training

Configurable advanced training with optional MLflow, XGBoost, and LightGBM:

```bash
python scripts/train_advanced.py
```

Configuration lives in `config/training_config.yaml`.

Artifacts are saved under `models/advanced/`.

### Realtime Training

Realtime-oriented SCADA models intended for live simulator scoring:

```powershell
venv\Scripts\python.exe scripts\train_realtime_models.py
```

or with an explicit config:

```powershell
venv\Scripts\python.exe scripts\train_realtime_models.py --config config\realtime_training.yaml
```

Configuration lives in `config/realtime_training.yaml`.

By default this trains on `data/sample/realtime_training_data.csv` and saves artifacts under `models/realtime/`.

Current realtime model set:
- `realtime_random_forest`
- `realtime_xgboost`
- `realtime_lightgbm`
- `realtime_isolation_forest`
- `realtime_hybrid_ensemble`

The hybrid ensemble combines multiple model families into one scoring artifact so you can compare a blended signal against single-model behavior in the dashboard.

### Robust Training

There is now a separate robust training path for live-safe fusion models:

```powershell
venv\Scripts\python.exe -m scripts.build_robust_training_data
venv\Scripts\python.exe -m scripts.train_robust_models
```

This path:
- regenerates simulator-aligned training data, including micro-leak scenarios
- fuses it with any locally available telemetry-compatible external datasets
- trains models with scale-invariant pressure/flow features under `models/robust/`

Configuration lives in `config/robust_training.yaml`.

Current defaults use:
- `data/sample/realtime_training_data.csv`
- `water_leak` when present
- `mendeley_water_testbed` when present

Robust artifacts are intended for the live simulator and are discovered alongside realtime artifacts.

### Cloud Notebook Workflow

For large Petrobras runs, prefer the notebook workflow so preprocessing and training can happen on cloud compute while the repo scripts remain the source of truth.

Recommended run layout:

```text
artifacts/petrobras/<run-label>/
  data/
    petrobras_3w_scada.csv
    petrobras_3w_processing_summary.json
  models/
    petrobras_training_summary.json
    petrobras_model_metrics.csv
    petrobras_*.joblib
```

That structure is what the notebook templates expect, and the compare notebook reads the generated `petrobras_training_summary.json` files to rank runs.

## Dashboard

Launch the app with:

```bash
streamlit run dashboard/app.py
```

The dashboard has two primary surfaces:

### Historical Analysis

- filter stored datasets by segment and date
- inspect pressure / flow time series
- run prediction views on stored data
- compare loaded models with confusion matrices and ROC curves

### Live Simulator

- start, pause, and restart a realtime telemetry stream
- choose scenario preset, segment count, tick speed, history size, and scoring model
- simulate incidents against SCADA-shaped live data
- overlay ideal detection markers showing where a strong model should start reacting

The simulator is implemented in:
- `src/simulation/core.py`
- `src/simulation/scenarios.py`

Current scenario presets:
- `Steady State`
- `Demand Shock`
- `Slow Seep`
- `Compound Incident`

## Model Artifact Loading

[`src/models/artifacts.py`](src/models/artifacts.py) now prioritizes models in this order:

1. `models/realtime/`
2. `models/advanced/`
3. dataset-specific files in `models/trained/`
4. legacy fallback artifacts in `models/`

That means newly trained realtime models appear automatically in the dashboard and live simulator without extra wiring.

## Reports and Tests

Benchmark outputs and dataset checks are written under `reports/dataset_checks/`.

Relevant tests include:
- `tests/test_loader.py`
- `tests/test_engineer.py`
- `tests/test_predict.py`
- `tests/test_simulator.py`
- `tests/test_realtime_models.py`

## Notes

- The current SCADA dataset is small and appears easy for the supervised boosted-tree models, so near-perfect metrics should be treated cautiously.
- The realtime training path is intentionally config-driven so you can point it at better future datasets without rewriting code.
- The anomaly detector is included to give you a second signal family for novel or weakly labeled conditions, but it should not be treated as a drop-in replacement for supervised leak labels.
