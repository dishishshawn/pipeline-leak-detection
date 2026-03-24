"""
Pipeline Leak Detection - Streamlit Dashboard
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.run_benchmark import make_features
from src.data.dataset_adapters import normalize
from src.data.loader import load_and_prepare
from src.features.engineer import build_features
from src.models.artifacts import discover_model_artifacts
from src.models.evaluate import classification_report_df, confusion_matrix_df, roc_auc
from src.models.predict import (
    load_model,
    predict,
    predict_leak_score,
    supports_leak_score,
    supports_probability_scores,
)
from src.models.train import prepare_training_data
from src.simulation import (
    PipelineTelemetrySimulator,
    SimulationConfig,
    build_scenarios,
    get_scenario_presets,
    make_default_profiles,
)

st.set_page_config(
    page_title="Pipeline Leak Detection",
    page_icon="W",
    layout="wide",
)

SAMPLE_DATA_PATH = "data/sample/scada_sample.csv"
MODEL_DIR = Path("models")
LIVE_MODEL_DATASET = "scada"


@st.cache_data
def load_data(path: str, dataset_type: str = "scada") -> pd.DataFrame:
    if dataset_type == "scada":
        df = load_and_prepare(path)
        df = build_features(df)
        return df

    df = normalize("water_leak")
    df = make_features(df, window=5)
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="1min")
    if "segment_id" not in df.columns:
        df["segment_id"] = 1
    if "alarm_triggered" not in df.columns:
        df["alarm_triggered"] = 0
    if "target" not in df.columns and "leak_label" in df.columns:
        df["target"] = df["leak_label"]
    return df


@st.cache_resource
def load_models(dataset_type: str = "scada"):
    models = {}
    for label, artifact in discover_model_artifacts(MODEL_DIR, dataset_type).items():
        models[label] = load_model(artifact.path)
    return models


def ensure_live_state() -> None:
    st.session_state.setdefault("live_simulator", None)
    st.session_state.setdefault("live_history", pd.DataFrame())
    st.session_state.setdefault("live_running", False)
    st.session_state.setdefault("live_signature", None)


def simulator_signature(
    preset_key: str,
    segment_count: int,
    tick_seconds: float,
    step_minutes: int,
    history_limit: int,
    seed: int,
    steps_per_refresh: int,
) -> tuple:
    return (
        preset_key,
        int(segment_count),
        float(tick_seconds),
        int(step_minutes),
        int(history_limit),
        int(seed),
        int(steps_per_refresh),
    )


def build_live_simulator(
    preset_key: str,
    segment_count: int,
    tick_seconds: float,
    step_minutes: int,
    history_limit: int,
    seed: int,
) -> PipelineTelemetrySimulator:
    profiles = make_default_profiles(segment_count)
    config = SimulationConfig(
        segment_profiles=profiles,
        start_time=datetime(2026, 1, 1, 6, 0, 0),
        tick_seconds=tick_seconds,
        step_minutes=step_minutes,
        history_limit=history_limit,
        seed=seed,
    )
    scenarios = build_scenarios(preset_key, [profile.segment_id for profile in profiles])
    return PipelineTelemetrySimulator(config=config, scenarios=scenarios)


def sync_live_history(new_rows: pd.DataFrame) -> None:
    if new_rows.empty:
        return

    history = st.session_state["live_history"]
    history = pd.concat([history, new_rows], ignore_index=True)

    simulator = st.session_state.get("live_simulator")
    if simulator is not None:
        max_rows = simulator.config.history_limit * len(simulator.config.segment_profiles)
        history = history.tail(max_rows).reset_index(drop=True)

    st.session_state["live_history"] = history


def score_live_history(history: pd.DataFrame, model):
    if history.empty:
        return pd.DataFrame(), pd.DataFrame()

    featured = build_features(history.copy())
    X_live, _ = prepare_training_data(featured)
    if X_live.empty:
        return featured, pd.DataFrame()

    result_columns = [
        "timestamp",
        "segment_id",
        "pressure",
        "flow_rate",
        "temperature",
        "event_type",
        "alarm_triggered",
        "target",
        "scenario_context",
    ]
    available_columns = [column for column in result_columns if column in featured.columns]
    result_df = featured[available_columns].copy()
    result_df["predicted"] = predict(model, X_live)

    if supports_leak_score(model):
        result_df["leak_score"] = predict_leak_score(model, X_live).round(3)

    return featured, result_df


def ideal_detection_markers(history: pd.DataFrame) -> pd.DataFrame:
    marker_columns = [
        "segment_id",
        "timestamp",
        "pressure",
        "flow_rate",
        "event_type",
        "target",
        "marker_label",
    ]
    if history.empty or "scenario_context" not in history.columns:
        return pd.DataFrame(columns=marker_columns)

    leak_context = history["scenario_context"].fillna("").str.contains("leak_progression")
    actionable_state = history["target"].eq(1) | history["event_type"].isin(["warning", "fault"])
    candidate_rows = history[leak_context & actionable_state].copy()
    if candidate_rows.empty:
        return pd.DataFrame(columns=marker_columns)

    markers = (
        candidate_rows.sort_values(["segment_id", "timestamp"])
        .groupby("segment_id", as_index=False)
        .first()[["segment_id", "timestamp", "pressure", "flow_rate", "event_type", "target"]]
    )
    markers["marker_label"] = "Ideal detection"
    return markers[marker_columns]


def add_ideal_detection_overlay(
    figure: go.Figure,
    markers: pd.DataFrame,
    *,
    y_column: str,
    chart_name: str,
) -> None:
    if markers.empty or y_column not in markers.columns:
        return

    for marker in markers.itertuples(index=False):
        figure.add_vline(
            x=marker.timestamp,
            line_dash="dot",
            line_color="#d62728",
            opacity=0.65,
        )

    figure.add_trace(
        go.Scatter(
            x=markers["timestamp"],
            y=markers[y_column],
            mode="markers+text",
            marker=dict(size=11, color="#d62728", symbol="diamond"),
            text=[f"Ideal detection S{segment_id}" for segment_id in markers["segment_id"]],
            textposition="top center",
            name=f"{chart_name} ideal detection",
            hovertemplate="Segment %{text}<br>Time=%{x}<br>Value=%{y:.3f}<extra></extra>",
        )
    )


def render_historical_tabs(
    filtered: pd.DataFrame,
    models: dict,
    selected_model_name: str,
) -> None:
    tab_ts, tab_pred, tab_eval = st.tabs(
        ["Time-series", "Predictions", "Model comparison"]
    )

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
            labels={"flow_rate": "Flow rate", "timestamp": "Time", "segment_id": "Segment"},
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

    with tab_pred:
        if not models or selected_model_name not in models:
            st.warning("No trained model available. Run `python -m src.models.train` first.")
        else:
            model = models[selected_model_name]
            X_live, _ = prepare_training_data(filtered)

            if X_live.empty:
                st.warning("No feature data available after filtering.")
            else:
                preds = predict(model, X_live)
                result_df = filtered[
                    ["timestamp", "segment_id", "pressure", "flow_rate", "target"]
                ].copy()
                result_df = result_df.iloc[: len(preds)].copy()
                result_df["predicted"] = preds

                if supports_leak_score(model):
                    scores = predict_leak_score(model, X_live)
                    result_df["leak_score"] = scores.round(3)

                    st.subheader("Leak probability scores")
                    fig_score = px.line(
                        result_df.sort_values("timestamp"),
                        x="timestamp",
                        y="leak_score",
                        color="segment_id",
                        labels={"leak_score": "Leak probability", "timestamp": "Time"},
                    )
                    fig_score.add_hline(
                        y=0.5,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="threshold",
                    )
                    st.plotly_chart(fig_score, use_container_width=True)
                else:
                    st.info("This model only produces class predictions, so leak scores are unavailable.")

                st.subheader("Prediction detail")
                st.dataframe(
                    result_df.sort_values(
                        "leak_score" if "leak_score" in result_df.columns else "predicted",
                        ascending=False,
                    ).head(50),
                    use_container_width=True,
                )

    with tab_eval:
        if not models:
            st.warning("No trained models found in the `models/` directory.")
        else:
            X_all, y_all = prepare_training_data(filtered)

            if X_all.empty or y_all.nunique() < 2:
                st.warning("Not enough data or only one class present - cannot evaluate.")
            else:
                for name, mdl in models.items():
                    st.subheader(name)
                    col_left, col_right = st.columns(2)

                    preds = predict(mdl, X_all)
                    report = classification_report_df(y_all, preds)

                    with col_left:
                        st.markdown("**Classification report**")
                        st.dataframe(
                            report.style.format("{:.3f}", na_rep="-"),
                            use_container_width=True,
                        )

                    cm = confusion_matrix_df(y_all, preds)
                    with col_right:
                        st.markdown("**Confusion matrix**")
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            labels={"color": "Count"},
                        )
                        st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{name}")

                    if supports_probability_scores(mdl):
                        scores = predict_leak_score(mdl, X_all)
                        fpr, tpr, auc_score = roc_auc(y_all, scores)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {auc_score:.3f}"))
                        fig_roc.add_shape(
                            type="line",
                            x0=0,
                            y0=0,
                            x1=1,
                            y1=1,
                            line=dict(dash="dash"),
                        )
                        fig_roc.update_layout(
                            title="ROC Curve",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            height=350,
                        )
                        st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_{name}")

                    st.markdown("---")


def render_live_view(live_models: dict, selected_live_model: str, steps_per_refresh: int) -> None:
    simulator = st.session_state.get("live_simulator")
    running = st.session_state.get("live_running", False)

    if simulator is None:
        st.info("Start the simulator to generate live telemetry and evaluate your model against a streaming incident.")
        return

    if running:
        sync_live_history(simulator.advance_steps(steps_per_refresh))

    history = st.session_state["live_history"]
    if history.empty:
        st.info("Simulator is configured but no telemetry has been emitted yet.")
        return

    model = live_models.get(selected_live_model) if live_models else None
    _, scored = score_live_history(history, model) if model is not None else (history, pd.DataFrame())
    markers = ideal_detection_markers(history)

    latest_timestamp = history["timestamp"].max()
    latest_rows = history[history["timestamp"] == latest_timestamp].sort_values("segment_id")
    latest_scored = (
        scored[scored["timestamp"] == latest_timestamp].sort_values("segment_id")
        if not scored.empty
        else pd.DataFrame()
    )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Simulator status", "Running" if running else "Paused")
    metric2.metric("Simulated time", latest_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    metric3.metric("Rows emitted", f"{len(history):,}")
    metric4.metric("Active leak segments", int(latest_rows["target"].sum()))

    if not latest_scored.empty and "leak_score" in latest_scored.columns:
        high_risk_segments = int((latest_scored["leak_score"] >= 0.5).sum())
        st.caption(f"High-risk segments by model score >= 0.50: {high_risk_segments}")
    if not markers.empty:
        st.caption("Red markers show the idealized earliest point where a strong model should begin flagging the leak.")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_pressure = px.line(
            history.sort_values("timestamp"),
            x="timestamp",
            y="pressure",
            color="segment_id",
            labels={"pressure": "Pressure (bar)", "timestamp": "Time", "segment_id": "Segment"},
            title="Live pressure telemetry",
        )
        add_ideal_detection_overlay(fig_pressure, markers, y_column="pressure", chart_name="Pressure")
        st.plotly_chart(fig_pressure, use_container_width=True)

    with chart_col2:
        fig_flow = px.line(
            history.sort_values("timestamp"),
            x="timestamp",
            y="flow_rate",
            color="segment_id",
            labels={"flow_rate": "Flow rate", "timestamp": "Time", "segment_id": "Segment"},
            title="Live flow telemetry",
        )
        add_ideal_detection_overlay(fig_flow, markers, y_column="flow_rate", chart_name="Flow")
        st.plotly_chart(fig_flow, use_container_width=True)

    if not scored.empty and "leak_score" in scored.columns:
        fig_score = px.line(
            scored.sort_values("timestamp"),
            x="timestamp",
            y="leak_score",
            color="segment_id",
            labels={"leak_score": "Leak score", "timestamp": "Time", "segment_id": "Segment"},
            title=f"Model leak score ({selected_live_model})",
        )
        fig_score.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="alert threshold")
        if not markers.empty:
            score_markers = markers[["segment_id", "timestamp"]].copy()
            score_values = (
                scored.sort_values(["segment_id", "timestamp"])
                .merge(score_markers, on=["segment_id", "timestamp"], how="inner")
            )
            if not score_values.empty:
                add_ideal_detection_overlay(
                    fig_score,
                    score_values.assign(marker_label="Ideal detection"),
                    y_column="leak_score",
                    chart_name="Leak score",
                )
        st.plotly_chart(fig_score, use_container_width=True)

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        st.subheader("Latest telemetry snapshot")
        st.dataframe(
            latest_rows[
                [
                    "segment_id",
                    "pressure",
                    "flow_rate",
                    "temperature",
                    "alarm_triggered",
                    "event_type",
                    "target",
                    "scenario_context",
                ]
            ],
            use_container_width=True,
        )

    with table_col2:
        if latest_scored.empty:
            st.subheader("Model output")
            st.info("Load a SCADA-compatible model to score the live telemetry stream.")
        else:
            visible_columns = [
                column
                for column in [
                    "segment_id",
                    "predicted",
                    "leak_score",
                    "event_type",
                    "target",
                    "scenario_context",
                ]
                if column in latest_scored.columns
            ]
            st.subheader("Latest model output")
            st.dataframe(latest_scored[visible_columns], use_container_width=True)

    st.subheader("Recent telemetry")
    history_columns = [
        "timestamp",
        "segment_id",
        "pressure",
        "flow_rate",
        "temperature",
        "alarm_triggered",
        "event_type",
        "target",
        "scenario_context",
    ]
    recent = history.sort_values("timestamp", ascending=False)[history_columns].head(30)
    st.dataframe(recent, use_container_width=True)


ensure_live_state()

st.sidebar.title("Pipeline Leak Detection")
st.sidebar.markdown("---")

dataset_options = {
    "SCADA Pipeline": "scada",
    "Water Leak": "water_leak",
}
selected_dataset_name = st.sidebar.selectbox(
    "Dataset Type",
    list(dataset_options.keys()),
    index=0,
)
dataset_type = dataset_options[selected_dataset_name]

default_data_path = SAMPLE_DATA_PATH if dataset_type == "scada" else "data/raw/water_leak/water_leak_detection_1000_rows.csv"
data_path = st.sidebar.text_input("Data path", value=default_data_path)

try:
    df = load_data(data_path, dataset_type)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

segments = sorted(df["segment_id"].unique())
selected_segments = st.sidebar.multiselect("Segments", segments, default=segments)

min_ts = df["timestamp"].min()
max_ts = df["timestamp"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_ts.date(), max_ts.date()),
    min_value=min_ts.date(),
    max_value=max_ts.date(),
)

filtered = df[df["segment_id"].isin(selected_segments)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered["timestamp"].dt.date >= date_range[0])
        & (filtered["timestamp"].dt.date <= date_range[1])
    ]

try:
    models = load_models(dataset_type)
    st.sidebar.success(f"Loaded {len(models)} models for {selected_dataset_name}")
except Exception as exc:
    st.sidebar.warning(f"Could not load models: {exc}")
    models = {}

selected_model_name = st.sidebar.selectbox(
    "Active historical model",
    list(models.keys()) if models else ["No models found"],
)

if models:
    with st.sidebar.expander("Available Historical Models"):
        for name in models.keys():
            st.write(f"- {name}")

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

historical_tab, live_tab = st.tabs(["Historical Analysis", "Live Simulator"])

with historical_tab:
    render_historical_tabs(filtered, models, selected_model_name)

with live_tab:
    st.subheader("Real-time Simulator")
    st.caption(
        "This simulator emits SCADA-style telemetry in real time, keeps state per segment, and applies modular incident scenarios."
    )

    live_models = load_models(LIVE_MODEL_DATASET)
    preset_definitions = get_scenario_presets()
    preset_map = {preset.key: preset for preset in preset_definitions}

    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        preset_label = st.selectbox(
            "Scenario preset",
            options=[preset.label for preset in preset_definitions],
            index=1,
            key="live_preset_label",
        )
        segment_count = st.slider("Segments", min_value=2, max_value=6, value=3, key="live_segment_count")
        step_minutes = st.slider("Simulated minutes per tick", min_value=1, max_value=15, value=1, key="live_step_minutes")

    with control_col2:
        tick_seconds = st.slider("Real seconds per tick", min_value=1, max_value=5, value=1, key="live_tick_seconds")
        steps_per_refresh = st.slider("Simulation speed", min_value=1, max_value=20, value=5, help="How many simulator ticks to advance on each dashboard refresh.", key="live_steps_per_refresh")
        history_limit = st.slider("History per segment", min_value=120, max_value=1440, value=360, step=60, key="live_history_limit")
        seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1, key="live_seed")

    with control_col3:
        selected_live_model = st.selectbox(
            "Scoring model",
            options=list(live_models.keys()) if live_models else ["No SCADA models found"],
            key="live_model_name",
        )
        st.markdown("**Preset description**")
        selected_preset_key = next(
            preset.key for preset in preset_definitions if preset.label == preset_label
        )
        st.write(preset_map[selected_preset_key].description)

    signature = simulator_signature(
        selected_preset_key,
        segment_count,
        tick_seconds,
        step_minutes,
        history_limit,
        int(seed),
        steps_per_refresh,
    )

    button_col1, button_col2, button_col3 = st.columns(3)
    if button_col1.button("Start / Resume", use_container_width=True):
        simulator = st.session_state.get("live_simulator")
        if simulator is None or st.session_state.get("live_signature") != signature:
            simulator = build_live_simulator(
                selected_preset_key,
                segment_count,
                tick_seconds,
                step_minutes,
                history_limit,
                int(seed),
            )
            st.session_state["live_simulator"] = simulator
            st.session_state["live_history"] = pd.DataFrame()
            st.session_state["live_signature"] = signature
        simulator.start()
        st.session_state["live_running"] = True

    if button_col2.button("Pause", use_container_width=True):
        simulator = st.session_state.get("live_simulator")
        if simulator is not None:
            simulator.stop()
        st.session_state["live_running"] = False

    if button_col3.button("Restart", use_container_width=True):
        simulator = build_live_simulator(
            selected_preset_key,
            segment_count,
            tick_seconds,
            step_minutes,
            history_limit,
            int(seed),
        )
        st.session_state["live_simulator"] = simulator
        st.session_state["live_history"] = pd.DataFrame()
        st.session_state["live_signature"] = signature
        simulator.start()
        st.session_state["live_running"] = True

    st.markdown("---")

    if st.session_state.get("live_running") and hasattr(st, "fragment"):
        @st.fragment(run_every=f"{int(tick_seconds)}s")
        def _live_fragment() -> None:
            render_live_view(live_models, selected_live_model, steps_per_refresh)

        _live_fragment()
    else:
        if st.session_state.get("live_running") and not hasattr(st, "fragment"):
            st.info("Your Streamlit build does not expose `st.fragment`, so use the page rerun to refresh live telemetry.")
        render_live_view(live_models, selected_live_model, steps_per_refresh)
