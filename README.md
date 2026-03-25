# Pipeline Leak Detection System

Pipeline leak detection and monitoring project built around SCADA-style sensor data, supervised ML models, anomaly detection, and a Streamlit dashboard with a live simulator.

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
- `data/raw/scada_pipeline/scada_pipeline.csv`: alternate raw SCADA dataset retained for dataset exploration
- `data/raw/water_leak/water_leak_detection_1000_rows.csv`: alternate water leak dataset used by the benchmark/dashboard path

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
- rolling pressure mean/std
- rolling flow mean/std
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

By default this trains on `data/sample/scada_sample.csv` and saves artifacts under `models/realtime/`.

Current realtime model set:
- `realtime_random_forest`
- `realtime_xgboost`
- `realtime_lightgbm`
- `realtime_isolation_forest`
- `realtime_hybrid_ensemble`

The hybrid ensemble combines multiple model families into one scoring artifact so you can compare a blended signal against single-model behavior in the dashboard.

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
