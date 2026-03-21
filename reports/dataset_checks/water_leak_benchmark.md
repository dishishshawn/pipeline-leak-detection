# Benchmark: water_leak

**Target:** `Leak Status`  |  **Train:** 800  |  **Test:** 200
**Positive rate (test):** 2.0%

## Model Results

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| majority | 0.9800 | 0.0000 | N/A |
| logistic_regression | 0.9100 | 0.2500 | 0.9031 |
| random_forest | 0.9850 | 0.4000 | 0.8444 |
| isolation_forest_unlabeled | 0.9150 | 0.1905 | 0.7564 |