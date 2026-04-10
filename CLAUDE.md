# CLAUDE.md — Agent Context for Pipeline Leak Detection

> This file gives a Claude Code agent everything it needs to work on this project without re-exploring the codebase. **Update this file after every significant change.**

## Project Overview

Pipeline leak detection system for oil & gas. Startup demo targeting YC application. Three ML training pipelines (physics-simulated, realtime simulator, robust fusion), a live Streamlit dashboard with real-time scoring, and a physics-grounded transient flow simulator.

**Owner:** dishishshawn
**Repo:** https://github.com/dishishshawn/pipeline-leak-detection
**Platform:** Windows (venv at `venv/Scripts/python`)
**Dashboard port:** 8510

## Quick Commands

```bash
python tasks.py test          # Run 49 tests
python tasks.py dashboard     # Launch Streamlit on :8510
python tasks.py physics-all   # Generate physics data + train models
python tasks.py realtime-all  # Generate realtime data + train models
python tasks.py eval          # Evaluate models + calibrate thresholds
python tasks.py pipeline-full # Train everything + evaluate
python tasks.py petrobras-download  # Download Petrobras 3W real data (~1.8 GB)
python scripts/download_petrobras_3w.py --skip-download --max-files 500  # Rebuild a medium 3W training slice
python scripts/compare_petrobras_runs.py --summary-glob "artifacts/petrobras/*/models/petrobras_training_summary.json"
python tasks.py --list        # Show all 26 tasks
```

## Directory Structure

```
config/                    YAML training configs
  training_config.yaml     Advanced training (MLflow, Optuna)
  realtime_training.yaml   Realtime model hyperparams
  robust_training.yaml     Robust fusion model hyperparams
  datasets.yaml            External dataset registry

dashboard/
  app.py                   Streamlit app (historical + live simulator tabs)

data/
  sample/                  Quick test data (scada_sample.csv, realtime_training_data.csv)
  raw/                     Downloaded datasets (scada_pipeline, water_leak)
  processed/               Feature-engineered datasets
  physics_sim/             Physics simulator output (scada_timeseries.csv)

notebooks/
  01_setup_cloud.ipynb     Clone/install + storage setup for Colab/Kaggle
  02_prepare_petrobras.ipynb  Build processed 3W CSV + JSON processing summary
  03_train_petrobras.ipynb    Train Petrobras models into a run-specific artifact dir
  04_compare_petrobras_runs.ipynb  Flatten summaries and compare runs visually

models/
  realtime/                Live-safe models (RF, XGB, LGB, IF, HybridEnsemble)
  physics_sim/             Physics-trained models (5 types)
  robust/                  Fusion models (simulator + external data)
  advanced/                MLflow/Optuna trained models
  trained/                 Dataset-specific models

reports/
  evaluation_results.json  Calibrated thresholds + metrics per model
  evaluation_details.csv   Per-row scores for all models/scenarios

scripts/
  generate_physics_dataset.py    Physics data generation (with per-scenario validation)
  train_physics_models.py        Train on physics data
  generate_realtime_training_data.py  Simulator training data
  train_realtime_models.py       Train realtime models
  build_robust_training_data.py  Build robust corpus
  train_robust_models.py         Train robust models
  evaluate_models.py             Scenario-level evaluation + threshold calibration
  train_advanced.py              Advanced training (MLflow)
  run_eda.py                     Exploratory data analysis
  run_benchmark.py               Baseline benchmarks
  download_data.py               Kaggle dataset download
  download_petrobras_3w.py       Download & process Petrobras 3W (real oil well data)
  compare_petrobras_runs.py      Flatten run summaries for notebook comparison

src/
  data/                    Loaders, adapters, dataset registry, robust corpus
  features/engineer.py     Feature engineering (deltas, rolling stats, z-scores)
  models/
    predict.py             Unified prediction layer (routes PhysicsModelWrapper)
    artifacts.py           Model discovery with priority ordering
    atlas.py               ATLAS model serving layer (caching, fallback, latency monitoring)
    physics_wrapper.py     On-the-fly feature engineering for physics models
    train.py               Basic training (LR, RF)
    realtime_train.py      Realtime model training pipeline (micro-leak sample weight: 5x)
    realtime.py            Custom model classes (IsolationForestLeakDetector, etc.)
    evaluate.py            Classification metrics (confusion matrix, ROC, reports)
  evaluation/
    harness.py             Scenario-level evaluation + threshold calibration
    alert_policy.py        Stateful AlertPolicy (persistence + cooldown)
  physics_sim/
    config.py              SimConfig, PipeConfig, FluidConfig, LeakConfig, etc.
    solver.py              PipelineSimulator (1D transient flow, Radau ODE solver)
    hydraulics.py          Reynolds, friction factor, orifice flow
    scenarios.py           Batch scenario generation (randomized params)
    validate.py            8 physics sanity checks
    export.py              CSV/Parquet export
    plots.py               Pressure profiles, timeseries comparison
  simulation/
    core.py                PipelineTelemetrySimulator (dashboard live view)
    scenarios.py           Scenario presets + manual leak triggers

tests/                     49 tests covering loader, engineer, predict, simulator, Petrobras workflow, models
```

## Architecture Patterns

### Model Loading & Prediction

`src/models/predict.py` is the unified prediction interface. Key routing:

```python
# Physics models get wrapped automatically
def load_model(path):
    if "physics_sim" in str(path):
        return PhysicsModelWrapper(estimator, scaler=scaler)

# PhysicsModelWrapper short-circuits _prepare_features
def predict(model, df):
    if isinstance(model, PhysicsModelWrapper):
        return model.predict(df)  # Does its own 27-feature engineering
    estimator, X = _prepare_features(model, df)
    return estimator.predict(X)
```

### ATLAS Model Serving Layer

`src/models/atlas.py` — Deployment infrastructure for the winning model:
- **Primary model:** Realtime XGBoost (`ATLAS_MODEL = "Realtime Xgboost"` in `dashboard/app.py`)
- Automatic best-model selection with configurable override
- LRU prediction cache, batch prediction, latency monitoring
- Fallback cascade: top-2 candidate models tried in order if primary fails
- Default candidate order in `atlas.py`: XGBoost > RF > LightGBM > Hybrid Ensemble > Petrobras models
- Hot-swap via `server.swap_primary()` or `server.set_primary_by_label()`

### Model Artifact Discovery

`src/models/artifacts.py` scans directories with priority:
1. `models/robust/` (priority 0)
2. `models/realtime/` (priority 1 — preferred for live)
3. `models/petrobras/` (priority 2 — real oil well data)
4. `models/physics_sim/` (priority 3)
5. `models/advanced/` (priority 4)
6. `models/trained/` (priority 5+)

### Alert System

`src/evaluation/alert_policy.py` — Stateful per-segment filtering:
- **Persistence mode:** 3 consecutive above-threshold ticks to fire, then 10-tick cooldown
- Integrated into dashboard live view via session state
- `model_alert` column in scored output, red triangle markers on chart

### Threshold Calibration

`src/evaluation/harness.py` — `calibrate_threshold()`:
1. Compute steady-state noise ceiling: 95th percentile of steady_state scores (primary constraint, 70% weight)
2. Compute demand-shock ceiling: 90th percentile of demand_shock scores (secondary, 30% weight)
3. Threshold = blended ceiling * 1.10 + 0.005 margin
4. Floor: 0.02, cap: 0.85

Current calibrated thresholds (from `reports/evaluation_results.json`):
- Realtime XGBoost: 0.02 (ATLAS primary)
- Realtime RF: 0.07
- Realtime LightGBM: 0.02
- Realtime Hybrid Ensemble: 0.22
- Robust models: 0.44-0.53 range
- Petrobras RF: 0.40, Petrobras XGBoost: 0.04

Micro-leak sensitivity (after expanding-window + EMA feature additions):
- Realtime XGBoost: 95% (up from 73% after CUSUM, 38% before micro-leak features)
- Realtime RF: varies by run
- Realtime Hybrid Ensemble: varies by run

### Physics Simulator

`src/physics_sim/` — 1D transient pipe flow:
- Staggered grid: N pressure cells, N+1 flow faces
- Darcy-Weisbach friction (Swamee-Jain approximation)
- Orifice leak model: Q_leak = Cd * A * sqrt(2 * dP / rho)
- Effective bulk modulus: fluid + pipe wall compressibility
- Scipy Radau implicit solver for stiff ODEs
- Per-scenario physics validation during generation (critical failures drop the scenario)

### Dashboard

`dashboard/app.py` — Two main tabs:
1. **Historical Analysis** — Time-series, Predictions, Model comparison sub-tabs
2. **Live Simulator** — Real-time telemetry with segment health cards, leak severity chart, model leak score chart with threshold + confirmed alert markers

ATLAS model serving: `ATLAS_MODEL = "Realtime Xgboost"` (line 121). The dashboard uses the ATLAS serving layer (`src/models/atlas.py`) for live predictions with caching and fallback.

All predict calls are wrapped in try/except to prevent crashes. Uses `width="stretch"` (not deprecated `use_container_width`).

### Feature Engineering

`src/features/engineer.py` — `build_features(df)`:
- Pressure/flow deltas and percent deltas
- Rolling mean/std over windows [5, 10, 15]
- Z-scores (rolling normalization)
- Pressure-to-flow ratio
- Physics-informed: pressure_delta_deviation, flow_neg_streak, segment deviations
- Micro-leak features (30-step long window):
  - `pressure_roll30_std`, `flow_roll30_std` — long-window rolling standard deviation
  - `pressure_cusum_neg30` — CUSUM cumulative negative pressure drop (30-step window)
  - `pressure_flow_divergence` — normalized pressure-flow trend divergence
  - `pressure_neg_streak15` — sustained negative pressure streak count (15-step window)
- Advanced micro-leak features (key to 95% sensitivity):
  - `pressure_ema_dev`, `flow_ema_dev` — EMA drift detector (span=20)
  - `pressure_accel` — pressure acceleration (second derivative of pressure)
  - `flow_cusum_neg30` — cumulative negative flow deficit (30-step window)
  - `pressure_baseline_pct` — pressure deviation as fraction of 20-step baseline
  - `pressure_baseline60_dev`, `flow_baseline60_dev` — 90-step ultra-long baseline deviation
  - `pressure_expanding_dev`, `flow_expanding_dev` — expanding-window deviation (never adapts)

`src/models/physics_wrapper.py` — `PhysicsModelWrapper._engineer_features(df)`:
- 27 features from raw SCADA columns (P_inlet, P_mid, P_outlet, Q_inlet, Q_outlet, T_*)
- Pressure deltas, flow imbalance ratio, temperature gradient
- Rolling mean/std (window=5)

### Petrobras 3W Dataset

Real oil well telemetry from Petrobras (~2,228 parquet files locally after download, ~1.8 GB). NOT in git — team members run `python tasks.py petrobras-download` to get it locally.

- **Download script:** `scripts/download_petrobras_3w.py`
- **Raw data:** `data/raw/petrobras_3w/3W/` (gitignored)
- **Processed output:** `data/processed/petrobras_3w/petrobras_3w_scada.csv` (gitignored)
- **Cloud/notebook output pattern:** `artifacts/petrobras/<run-label>/data/` and `artifacts/petrobras/<run-label>/models/` (gitignored)
- **Column mapping:** P-PDG → P_inlet, P-TPT → P_mid, P-MON-CKP → P_outlet, T-PDG → T_inlet, T-TPT → T_mid, QGL → Q_inlet
- **Label mapping:** Event 0 = normal (target=0), Events 1-9 = anomaly/fault (target=1), and 100-series variants like 101-109 are normalized back to 1-9 during processing
- **Sensor cleanup:** impossible sentinels like `-1e42` pressure or `-1e38` temperature are scrubbed to NaN before fill/interpolation
- **Training script:** `scripts/train_petrobras_models.py` (27 physics-style features, run labels, summary JSON, metrics CSV)
- **Notebook workflow:** use the four notebooks under `notebooks/` for cloud setup, preprocessing, training, and run comparison
- **Trained models:** `models/petrobras/petrobras_*.joblib` (auto-wrapped in PhysicsModelWrapper)
- **Current medium run:** `--max-files 500` sampled 484 files -> 2,560,071 rows, 218 wells, 484 scenarios
- **Best current result:** LightGBM ROC-AUC 0.9425, F1 0.9297 (500-file rebuild on March 29, 2026)
- **Previous small-sample result:** XGBoost ROC-AUC 0.9955, F1 0.9786 (50-file subset; likely optimistic versus the broader corpus)
- **Config entry:** `config/datasets.yaml` lines 170-208

## Known Issues / Gotchas

1. **xgboost/lightgbm may not be installed** — The venv sometimes lacks these. Models that need them show warnings but don't crash the dashboard.
2. **PhysicsModelWrapper must bypass _prepare_features** — predict.py has isinstance checks; don't remove them or you'll get "17 vs 27 features" errors.
3. **Robust models perform poorly** (~10% FPR) — They were trained on mixed data that doesn't match the simulator distribution well. Known limitation.
4. **Isolation Forest ROC-AUC is ~0.26** — Expected for unsupervised anomaly detection; sigmoid transform of score_samples doesn't align with binary labels.
5. **Multiple Streamlit instances** — Old instances may linger on ports 8501-8503. Kill them before starting on 8510.
6. **Windows encoding** — Use ASCII in print statements, not Unicode symbols (epsilon, checkmark). Windows cp1252 will throw UnicodeEncodeError.
7. **Petrobras 3W has dirty raw values** — Some files contain 100-series class labels and extreme sensor sentinels. Rebuild `data/processed/petrobras_3w/petrobras_3w_scada.csv` with the current script before trusting older Petrobras model metrics.

## Tech Stack

pandas 2.3, numpy 2.4, scikit-learn 1.8, streamlit 1.55, plotly 6.6, scipy 1.17, xgboost 2.0, lightgbm 4.3, mlflow 2.12, optuna 3.5, pytest 8.3

## Trello Board Columns

1. **Create a better model** — Find real datasets, improve physics sim, fix robust models
2. **Dashboard Updates** — Deprecation fixes, error handling, model browsing
3. **Find companies to pitch to** — Contact list
4. **Research** — Compare water leak vs oil/gas SCADA data
5. **Funding** — YC application ($500k)
6. **Real-Time data -> prediction** — Pipeline simulator integration, model selection UI
7. **Training reproducibility** — Standardize model saving, unify train scripts, log dataset versions
