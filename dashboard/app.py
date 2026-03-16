"""
Pipeline Leak Detection — Streamlit Dashboard
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_and_prepare
from src.features.engineer import build_features
from src.models.evaluate import classification_report_df, confusion_matrix_df, roc_auc
from src.models.predict import load_model, predict, predict_leak_score
from src.models.train import prepare_training_data

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pipeline Leak Detection",
    page_icon="🔧",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAMPLE_DATA_PATH = "data/sample/scada_sample.csv"
MODEL_DIR = Path("models")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = load_and_prepare(path)
    df = build_features(df)
    return df


@st.cache_resource
def load_models():
    models = {}
    for model_file in MODEL_DIR.glob("*.joblib"):
        label = model_file.stem.replace("_", " ").title()
        models[label] = load_model(str(model_file))
    return models


# ---------------------------------------------------------------------------
# Sidebar — data source & filters
# ---------------------------------------------------------------------------
st.sidebar.title("Pipeline Leak Detection")
st.sidebar.markdown("---")

data_path = st.sidebar.text_input("Data path", value=SAMPLE_DATA_PATH)

try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

segments = sorted(df["segment_id"].unique())
selected_segments = st.sidebar.multiselect(
    "Segments", segments, default=segments
)

min_ts = df["timestamp"].min()
max_ts = df["timestamp"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_ts.date(), max_ts.date()),
    min_value=min_ts.date(),
    max_value=max_ts.date(),
)

# Apply filters
filtered = df[df["segment_id"].isin(selected_segments)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered["timestamp"].dt.date >= date_range[0])
        & (filtered["timestamp"].dt.date <= date_range[1])
    ]

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
try:
    models = load_models()
except Exception as e:
    st.warning(f"Could not load models: {e}")
    models = {}

selected_model_name = st.sidebar.selectbox(
    "Active model", list(models.keys()) if models else ["No models found"]
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.title("Pipeline Leak Detection Dashboard")

total = len(filtered)
leak_count = int(filtered["target"].sum())
leak_rate = leak_count / total * 100 if total > 0 else 0
alarm_count = int(filtered["alarm_triggered"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total readings", f"{total:,}")
col2.metric("Leak events", f"{leak_count:,}", delta=f"{leak_rate:.1f}%")
col3.metric("Alarms triggered", f"{alarm_count:,}")
col4.metric("Segments", len(selected_segments))

st.markdown("---")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_ts, tab_pred, tab_eval = st.tabs(
    ["Time-series", "Predictions", "Model comparison"]
)

# ---- Time-series tab ----
with tab_ts:
    st.subheader("Pressure over time")
    fig_pressure = px.line(
        filtered.sort_values("timestamp"),
        x="timestamp",
        y="pressure",
        color="segment_id",
        labels={"pressure": "Pressure (bar)", "timestamp": "Time", "segment_id": "Segment"},
    )
    st.plotly_chart(fig_pressure, use_container_width=True)

    st.subheader("Flow rate over time")
    fig_flow = px.line(
        filtered.sort_values("timestamp"),
        x="timestamp",
        y="flow_rate",
        color="segment_id",
        labels={"flow_rate": "Flow rate (m\u00b3/s)", "timestamp": "Time", "segment_id": "Segment"},
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    st.subheader("Leak events")
    leak_df = filtered[filtered["target"] == 1]
    if leak_df.empty:
        st.info("No leak events in the selected range.")
    else:
        fig_leaks = px.scatter(
            leak_df,
            x="timestamp",
            y="pressure",
            color="segment_id",
            symbol_sequence=["x"],
            labels={"pressure": "Pressure (bar)", "timestamp": "Time"},
            title="Pressure at leak events",
        )
        st.plotly_chart(fig_leaks, use_container_width=True)

# ---- Predictions tab ----
with tab_pred:
    if not models or selected_model_name not in models:
        st.warning("No trained model available. Run `python src/models/train.py` first.")
    else:
        model = models[selected_model_name]
        X, y = prepare_training_data(filtered)

        if X.empty:
            st.warning("No feature data available after filtering.")
        else:
            scores = predict_leak_score(model, X)
            preds = predict(model, X)

            result_df = filtered[["timestamp", "segment_id", "pressure", "flow_rate", "target"]].copy()
            result_df = result_df.iloc[: len(scores)].copy()
            result_df["leak_score"] = scores.round(3)
            result_df["predicted"] = preds

            st.subheader("Leak probability scores")
            fig_score = px.line(
                result_df.sort_values("timestamp"),
                x="timestamp",
                y="leak_score",
                color="segment_id",
                labels={"leak_score": "Leak probability", "timestamp": "Time"},
            )
            fig_score.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="threshold")
            st.plotly_chart(fig_score, use_container_width=True)

            st.subheader("Prediction detail")
            st.dataframe(
                result_df.sort_values("leak_score", ascending=False).head(50),
                use_container_width=True,
            )

# ---- Model comparison tab ----
with tab_eval:
    if not models:
        st.warning("No trained models found in the `models/` directory.")
    else:
        X_all, y_all = prepare_training_data(filtered)

        if X_all.empty or y_all.nunique() < 2:
            st.warning("Not enough data or only one class present — cannot evaluate.")
        else:
            for name, mdl in models.items():
                st.subheader(name)
                col_left, col_right = st.columns(2)

                preds = predict(mdl, X_all)
                report = classification_report_df(y_all, preds)

                with col_left:
                    st.markdown("**Classification report**")
                    st.dataframe(report.style.format("{:.3f}", na_rep="-"), use_container_width=True)

                cm = confusion_matrix_df(y_all, preds)
                with col_right:
                    st.markdown("**Confusion matrix**")
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels={"color": "Count"},
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                # ROC curve
                if hasattr(mdl, "predict_proba"):
                    scores = predict_leak_score(mdl, X_all)
                    fpr, tpr, auc_score = roc_auc(y_all, scores)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {auc_score:.3f}"))
                    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
                    fig_roc.update_layout(
                        title="ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=350,
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                st.markdown("---")
