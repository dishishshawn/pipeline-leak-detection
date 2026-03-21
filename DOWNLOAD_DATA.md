# Download Data

## Prerequisites

```bash
pip install kaggle
```

Then add your Kaggle API key to `~/.kaggle/kaggle.json`:
```json
{"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}
```
Get the key at: https://www.kaggle.com/settings → API → Create New Token

On Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

---

## Phase 1 — Required for demo

### 1. SCADA Pipeline Operations Dataset
Direct-fit: SCADA-style pressure/flow/temp with fault labels.

```bash
kaggle datasets download -d zara2099/scada-pipeline-operations-dataset \
  -p data/raw/scada_pipeline --unzip
```

### 2. Water Leak Dataset
Direct-fit: water pipeline leak/no-leak time-series.

```bash
kaggle datasets download -d ziya07/water-leak-dataset \
  -p data/raw/water_leak --unzip
```

---

## Phase 2 — Optional (run after Phase 1 baseline is done)

### 3. Smart Water Leak Detection Dataset
```bash
kaggle datasets download -d talha97s/smart-water-leak-detection-dataset \
  -p data/raw/smart_water_leak --unzip
```

### 4. UCI Condition Monitoring of Hydraulic Systems
```bash
# Automated via download_data.py:
python scripts/download_data.py --datasets uci_hydraulic
```

---

## Phase 3 — Low priority

### 5. High-Pressure Gas Pipeline Acoustic Dataset
```bash
kaggle datasets download -d ziya07/high-pressure-gas-pipeline-acoustic-dataset \
  -p data/raw/gas_pipeline_acoustic --unzip
```

---

## Automated download (after kaggle auth is configured)

```bash
# Phase 1 only
python scripts/download_data.py --phase 1

# Specific dataset
python scripts/download_data.py --datasets scada_pipeline water_leak
```

---

## After download

```bash
# Run EDA on Phase 1 datasets
python scripts/run_eda.py

# Run cheap baselines
python scripts/run_benchmark.py
```

Results saved to `reports/dataset_checks/`.
