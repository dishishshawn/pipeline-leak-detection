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
python tasks.py test          # Run 40 tests
python tasks.py dashboard     # Launch Streamlit on :8510
python tasks.py physics-all   # Generate physics data + train models
python tasks.py realtime-all  # Generate realtime data + train models
python tasks.py eval          # Evaluate models + calibrate thresholds
python tasks.py pipeline-full # Train everything + evaluate
python tasks.py petrobras-download  # Download Petrobras 3W real data (~1.8 GB)
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

src/
  data/                    Loaders, adapters, dataset registry, robust corpus
  features/engineer.py     Feature engineering (deltas, rolling stats, z-scores)
  models/
    predict.py             Unified prediction layer (routes PhysicsModelWrapper)
    artifacts.py           Model discovery with priority ordering
    physics_wrapper.py     On-the-fly feature engineering for physics models
    train.py               Basic training (LR, RF)
    realtime_train.py      Realtime model training pipeline
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

tests/                     40 tests covering loader, engineer, predict, simulator, models
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

### Model Artifact Discovery

`src/models/artifacts.py` scans directories with priority:
1. `models/realtime/` (priority 0 — preferred for live)
2. `models/physics_sim/` (priority 2)
3. `models/advanced/` (priority 3)
4. `models/robust/` (priority 1)
5. `models/trained/` (priority 4)

### Alert System

`src/evaluation/alert_policy.py` — Stateful per-segment filtering:
- **Persistence mode:** 3 consecutive above-threshold ticks to fire, then 10-tick cooldown
- Integrated into dashboard live view via session state
- `model_alert` column in scored output, red triangle markers on chart

### Threshold Calibration

`src/evaluation/harness.py` — `calibrate_threshold()`:
1. Compute noise ceiling: 95th percentile of non-leak scores (steady_state + demand_shock)
2. Compute leak onset signal: median score at first leak rows in slow_seep
3. Threshold = midpoint between noise ceiling and leak onset
4. Floor: 0.15, cap: 0.85

Current calibrated thresholds (from `reports/evaluation_results.json`):
- Realtime RF: 0.1949
- Realtime XGBoost: 0.15
- Realtime LightGBM: 0.15
- Realtime Hybrid Ensemble: 0.2517
- Robust models: 0.30-0.39 range

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

All predict calls are wrapped in try/except to prevent crashes. Uses `width="stretch"` (not deprecated `use_container_width`).

### Feature Engineering

`src/features/engineer.py` — `build_features(df)`:
- Pressure/flow deltas and percent deltas
- Rolling mean/std over windows [5, 10, 15]
- Z-scores (rolling normalization)
- Pressure-to-flow ratio
- Physics-informed: pressure_delta_deviation, flow_neg_streak, segment deviations

`src/models/physics_wrapper.py` — `PhysicsModelWrapper._engineer_features(df)`:
- 27 features from raw SCADA columns (P_inlet, P_mid, P_outlet, Q_inlet, Q_outlet, T_*)
- Pressure deltas, flow imbalance ratio, temperature gradient
- Rolling mean/std (window=5)

### Petrobras 3W Dataset

Real oil well telemetry from Petrobras (~1,984 parquet files, ~1.8 GB). NOT in git — team members run `python tasks.py petrobras-download` to get it locally.

- **Download script:** `scripts/download_petrobras_3w.py`
- **Raw data:** `data/raw/petrobras_3w/3W/` (gitignored)
- **Processed output:** `data/processed/petrobras_3w/petrobras_3w_scada.csv` (gitignored)
- **Column mapping:** P-PDG → P_inlet, P-TPT → P_mid, P-MON-CKP → P_outlet, T-PDG → T_inlet, T-TPT → T_mid, QGL → Q_inlet
- **Label mapping:** Event 0 = normal (target=0), Events 1-9 = anomaly/fault (target=1)
- **Config entry:** `config/datasets.yaml` lines 170-208

## Known Issues / Gotchas

1. **xgboost/lightgbm may not be installed** — The venv sometimes lacks these. Models that need them show warnings but don't crash the dashboard.
2. **PhysicsModelWrapper must bypass _prepare_features** — predict.py has isinstance checks; don't remove them or you'll get "17 vs 27 features" errors.
3. **Robust models perform poorly** (~10% FPR) — They were trained on mixed data that doesn't match the simulator distribution well. Known limitation.
4. **Isolation Forest ROC-AUC is ~0.26** — Expected for unsupervised anomaly detection; sigmoid transform of score_samples doesn't align with binary labels.
5. **Multiple Streamlit instances** — Old instances may linger on ports 8501-8503. Kill them before starting on 8510.
6. **Windows encoding** — Use ASCII in print statements, not Unicode symbols (epsilon, checkmark). Windows cp1252 will throw UnicodeEncodeError.

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
