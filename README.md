# Pipeline Leak Detection System

AI-based pipeline monitoring system that detects leaks and abnormal pipeline behavior using SCADA sensor data and machine learning. Features a scalable training framework designed to handle datasets from small prototypes to large-scale industrial deployments.

This project combines traditional ML approaches with advanced algorithms, experiment tracking, and a monitoring dashboard for comprehensive pipeline leak detection.

---

# Project Goals

- Analyze pipeline SCADA data with time-series feature engineering
- Detect leaks using multiple ML algorithms (traditional + advanced)
- Compare model performance with comprehensive evaluation metrics
- Provide scalable training pipeline for datasets of any size
- Enable experiment tracking and hyperparameter optimization
- Deliver real-time monitoring dashboard for visualization

This system is designed for **scalability and production-readiness**, supporting everything from prototype demonstrations to large-scale industrial deployments.

---

# Technology Stack

| Component | Technology |
|--------|--------|
| Language | Python 3.8+ |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Configuration | YAML |
| Version Control | Git + GitHub |

---

# Repository Structure

```
pipeline-leak-detection/
├── config/
│   ├── datasets.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/
│   │   ├── scada_pipeline/
│   │   │   └── scada_pipeline.csv
│   │   └── water_leak/
│   │       └── water_leak_detection_1000_rows.csv
│   ├── processed/
│   │   ├── scada_pipeline/
│   │   └── water_leak/
│   └── sample/
│       └── scada_sample.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── advanced/
│   │   ├── logistic_regression.joblib
│   │   └── random_forest.joblib
│   └── trained/
│       ├── logistic_regression.joblib
│       └── random_forest.joblib
├── reports/
│   └── dataset_checks/
├── scripts/
│   ├── download_data.py
│   ├── run_benchmark.py
│   ├── run_eda.py
│   └── train_advanced.py
├── src/
│   ├── data/
│   │   ├── dataset_adapters.py
│   │   ├── dataset_registry.py
│   │   └── loader.py
│   ├── features/
│   │   └── engineer.py
│   └── models/
│       ├── artifacts.py
│       ├── evaluate.py
│       ├── predict.py
│       └── train.py
├── dashboard/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Setup Instructions

### 1. Clone the repository

```

git clone https://github.com/dishishshawn/pipeline-leak-detection
cd pipeline-leak-detection

```

---

### 2. Create a virtual environment

Windows:

```

python -m venv .venv
.venv\Scripts\activate

```

Mac / Linux:

```

python3 -m venv .venv
source .venv/bin/activate

```

---

### 3. Install dependencies

```

pip install -r requirements.txt

```

---

# Dataset

The project includes sample SCADA pipeline data for immediate testing and demonstration.

**Sample Dataset** (included):
- Location: `data/sample/scada_sample.csv`
- Size: ~500 rows for quick testing
- Features: pressure, flow rate, temperature, valve/pump states, event types
- Ready to use for dashboard demonstrations

**Full Dataset** (for large-scale training):
- Source: Kaggle SCADA pipeline operations dataset
- Expected size: 300GB+ for comprehensive model training
- Download using Kaggle CLI:

```

pip install kaggle
kaggle datasets download -d zara2099/scada-pipeline-operations-dataset -p data/raw --unzip

```

The training framework is designed to handle datasets of any size through chunked processing and out-of-core learning.

---

# Machine Learning Pipeline

The system features both basic and advanced ML workflows with comprehensive experiment tracking.

### 1. Data Loading & Preprocessing
- Load SCADA CSV files with schema validation
- Handle missing values and invalid readings
- Normalize timestamps and categorical features
- Support for chunked processing of large datasets

### 2. Feature Engineering
Advanced time-series feature engineering including:

- Pressure/flow rate deltas and rolling statistics
- Rolling means and standard deviations
- Event type encoding
- Compressor, pump, and valve state features

### 3. Model Training

**Basic Models:**
- Logistic Regression
- Random Forest

**Advanced Models (scalable for large datasets):**
- XGBoost - Gradient boosting for superior performance
- LightGBM - Microsoft's high-performance gradient boosting
- Hyperparameter optimization with Optuna
- Experiment tracking with MLflow

**Training Modes:**
- **Basic Training:** `python -m src.models.train`
- **Advanced Training:** `python scripts/train_advanced.py`

### 4. Model Evaluation & Comparison

Comprehensive evaluation metrics:
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves and AUC
- Cross-validation results

Models are compared side-by-side in the dashboard with interactive visualizations.

### 5. Experiment Tracking

- MLflow integration for logging hyperparameters, metrics, and artifacts
- Optuna for automated hyperparameter optimization
- YAML-based configuration for reproducible experiments
- Support for distributed training and large-scale evaluation

---

# Dashboard

Interactive Streamlit dashboard for real-time pipeline monitoring and ML model evaluation.

**Features:**
- **Time-series Visualization:** Pressure, flow rate, and temperature plots with segment filtering
- **Leak Detection:** Real-time anomaly scoring and prediction visualization
- **Model Comparison:** Side-by-side evaluation of all trained models with metrics and ROC curves
- **Interactive Filters:** Date range and segment selection for focused analysis

**Launch Command:**
```bash
streamlit run dashboard/app.py
```

The dashboard automatically loads the most advanced trained models and provides comprehensive monitoring capabilities.

---

# Advanced Training Configuration

The system includes a scalable training framework configured via `config/training_config.yaml`:

```yaml
# Example configuration
data:
  path: "data/sample/scada_sample.csv"
  chunk_size: 100000  # For large datasets

models:
  xgboost:
    enabled: true
    hyperparameters:
      n_estimators: [100, 200, 500]
      max_depth: [3, 6, 9]

training:
  cv_folds: 5
  random_state: 42

tuning:
  enabled: true
  method: "optuna"
  n_trials: 50
```

**Key Features:**
- **Scalable Processing:** Chunked data loading for datasets larger than RAM
- **Model Selection:** Enable/disable different algorithms based on dataset size
- **Hyperparameter Optimization:** Automated tuning with Optuna
- **Experiment Tracking:** MLflow integration for reproducible research
- **Configuration-Driven:** Easy to modify training parameters without code changes

---

# Usage Examples

### Quick Start (Sample Data)
```bash
# Install dependencies
pip install -r requirements.txt

# Run basic training
python -m src.models.train

# Launch dashboard
streamlit run dashboard/app.py
```

### Advanced Training (Large Dataset)
```bash
# Configure training in config/training_config.yaml
# Enable XGBoost, set chunk_size, configure MLflow

# Run advanced training
python scripts/train_advanced.py

# View experiments in MLflow UI
mlflow ui
```

This system is designed for **scalability and production-readiness**, supporting everything from prototype demonstrations to large-scale industrial deployments.