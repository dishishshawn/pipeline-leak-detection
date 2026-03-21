Priority 1 — Make It Runnable (MVP Demo)
1. Fix requirements.txt

File: requirements.txt
Rewrite with clean encoding; current file has spaces between every character (encoding corruption)
Pin versions for: pandas, numpy, scikit-learn, streamlit, plotly, joblib

2. Implement the Streamlit Dashboard

File: dashboard/app.py (currently empty)
Load saved models from models/ using joblib
Load and preprocess data via src/data/loader.py + src/features/engineer.py
Pages/sections:

Overview: KPIs (leak rate, active alarms, segment count)
Time-series view: Pressure & flow rate line charts per segment (Plotly)
Predictions: Table of model outputs with confidence scores
Model comparison: Logistic Regression vs. Random Forest side-by-side metrics


Use Streamlit sidebar for segment/time-range filters

3. Add Sample/Demo Dataset

Dir: data/sample/
Include a small synthetic or anonymized CSV (~500 rows) so the app runs without downloading from Kaggle
Update loader.py to accept a configurable path (env var or CLI arg) instead of hardcoded path


Priority 2 — Complete the Module Structure
4. Implement src/models/predict.py

Referenced in README but missing
Functions: load_model(path), predict(model, df), predict_proba(model, df)
Used by the dashboard and any batch inference scripts

5. Implement src/models/evaluate.py

Referenced in README but missing
Functions: classification_report_df(), confusion_matrix_plot(), roc_curve_plot()
Reuse in dashboard model-comparison panel

6. Add __init__.py Files

src/__init__.py, src/data/__init__.py, src/features/__init__.py, src/models/__init__.py
Enables proper package imports from the project root


Priority 3 — Testing & Quality
7. Unit Tests

Dir: tests/
Use pytest; create tests/test_loader.py, tests/test_engineer.py, tests/test_predict.py
Use data/sample/ CSV as test fixture

8. Logging

Add Python logging to loader, engineer, and train scripts (replace bare prints)


Priority 4 — Stretch / Advanced
9. Exploratory Analysis Notebook

File: notebooks/exploratory_analysis.ipynb
Visualize class imbalance, feature distributions, correlation heatmap

10. Advanced Training Framework (NEW - HIGH PRIORITY)

Implement scalable training pipeline for 300GB dataset:
- Out-of-core processing with pandas chunking
- XGBoost/LightGBM models for better performance on large data
- Hyperparameter tuning with Optuna or scikit-learn
- MLflow experiment tracking for hyperparameters, metrics, artifacts
- Configurable training pipeline (YAML config files)
- Cross-validation with memory-efficient chunking

11. Additional ML Models (DEPRECATED - covered by #10)

Isolation Forest for unsupervised anomaly detection (mentioned in README, not implemented)
Optional: XGBoost for comparison (now part of #10)

12. Hyperparameter Tuning (DEPRECATED - covered by #10)

Add GridSearchCV or RandomizedSearchCV for Random Forest in train.py (now part of #10)