# Notebook Workflow

These notebooks are thin orchestration layers for cloud training. They call the
repo scripts instead of reimplementing preprocessing or model training logic in
cells.

Recommended order:

1. `01_setup_cloud.ipynb`
2. `02_prepare_petrobras.ipynb`
3. `03_train_petrobras.ipynb`
4. `04_compare_petrobras_runs.ipynb`

Suggested storage layout on Colab or Kaggle:

```text
<storage-root>/
  raw/
    petrobras_3w/
      3W/
        dataset/
  artifacts/
    petrobras/
      <run-label>/
        data/
          petrobras_3w_scada.csv
          petrobras_3w_processing_summary.json
        models/
          petrobras_training_summary.json
          petrobras_model_metrics.csv
          petrobras_*.joblib
```

Key CLI options the notebooks use:

- `scripts/download_petrobras_3w.py --raw-dir --output-csv --summary-json`
- `scripts/train_petrobras_models.py --output --summary-json --metrics-csv --run-label`
- `scripts/compare_petrobras_runs.py --summary-glob --output-csv`

The `artifacts/` directory is gitignored so notebook runs do not pollute the repo.
