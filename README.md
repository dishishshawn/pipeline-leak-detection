# Pipeline Leak Detection Prototype

AI-based pipeline monitoring system that detects leaks and abnormal pipeline behavior using SCADA sensor data and machine learning.

This project is a prototype designed for demonstrations and pilot programs with oil & gas companies.  
The system analyzes time-series pipeline data such as pressure, flow rate, temperature, and valve states to identify potential leaks.

---

# Project Goals

- Analyze pipeline SCADA data
- Detect leaks using machine learning
- Compare traditional threshold detection with ML-based detection
- Simulate real-time sensor streams
- Provide a monitoring dashboard for visualization

This prototype focuses on **simplicity and clarity**, not production deployment.

---

# Technology Stack

| Component | Technology |
|--------|--------|
| Language | Python |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Version Control | Git + GitHub |

---

# Repository Structure

```

pipeline-leak-detection/
│
├── data/
│   ├── raw/            # downloaded datasets (not tracked by git)
│   ├── processed/      # cleaned datasets
│   └── sample/         # small demo datasets
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   │
│   ├── features/
│   │   └── engineer.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   └── utils/
│
├── dashboard/
│   └── app.py          # Streamlit monitoring dashboard
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
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

The project currently uses a SCADA pipeline dataset from Kaggle.

Dataset includes:

- Pressure
- Flow rate
- Temperature
- Valve states
- Pump states
- Event labels (normal / fault)

Download using Kaggle CLI:

```

pip install kaggle

kaggle datasets download -d zara2099/scada-pipeline-operations-dataset -p data/raw --unzip

```

---

# Machine Learning Pipeline

The ML workflow consists of several steps.

### 1. Data Loading
Load SCADA CSV files and validate schema.

### 2. Data Preprocessing
- handle missing values
- remove invalid readings
- normalize timestamps

### 3. Feature Engineering
Derived features may include:

- pressure change
- flow imbalance
- rolling averages
- rolling standard deviation

### 4. Model Training

Initial models include:

- Logistic Regression
- Random Forest
- Isolation Forest

The models attempt to detect leak events or abnormal pipeline behavior.

### 5. Model Evaluation

Models are evaluated using:

- precision
- recall
- F1 score
- confusion matrix

---

# Dashboard

A Streamlit dashboard visualizes pipeline data and ML predictions.

Visualizations include:

- pressure time-series
- flow imbalance
- anomaly scores
- leak detection indicators

Launch the dashboard:

```

streamlit run dashboard/app.py

```

This project is a **prototype for demonstration purposes only**.