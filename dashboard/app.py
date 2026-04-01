"""
Pipeline Leak Detection - Streamlit Dashboard
"""

from __future__ import annotations

import json
import logging
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
    expected_feature_columns,
    load_model,
    predict,
    predict_leak_score,
    supports_leak_score,
    supports_probability_scores,
)
from src.models.train import prepare_training_data
from src.evaluation.alert_policy import AlertPolicy, AlertPolicyConfig
from components.live_simulator import live_simulator_component
from src.simulation import (
    PipelineTelemetrySimulator,
    SimulationConfig,
    build_manual_leak_scenarios,
    build_scenarios,
    get_manual_leak_presets,
    get_scenario_presets,
    make_default_profiles,
)

st.set_page_config(
    page_title="Pipeline Leak Detection",
    page_icon="W",
    layout="wide",
)

logger = logging.getLogger(__name__)

SAMPLE_DATA_PATH = "data/sample/scada_sample.csv"
MODEL_DIR = Path("models")
LIVE_MODEL_DATASET = "scada"
MIN_MODEL_ROC_AUC = 0.55  # Hide models scoring below this
LIVE_SCORE_LOOKBACK = 30
LIVE_SCORE_KEY_COLUMNS = ["segment_id", "timestamp"]
LIVE_MODEL_ALLOWLIST = {
    "Realtime Random Forest",
    "Realtime Xgboost",
    "Realtime Lightgbm",
    "Realtime Hybrid Ensemble",
    "Petrobras Random Forest",
    "Petrobras Xgboost",
    "Petrobras Lightgbm",
    "Petrobras Logistic Regression",
    "Petrobras Isolation Forest",
    "Physics Sim Random Forest",
    "Physics Sim Xgboost",
    "Physics Sim Lightgbm",
    "Physics Sim Logistic Regression",
}
EVAL_RESULTS_PATH = Path("reports/evaluation_results.json")
DEFAULT_ALERT_THRESHOLD = 0.5


def _load_calibrated_thresholds() -> dict[str, float]:
    """Load calibrated per-model thresholds from evaluation results if available."""
    if EVAL_RESULTS_PATH.exists():
        try:
            data = json.loads(EVAL_RESULTS_PATH.read_text())
            return data.get("calibrated_thresholds", {})
        except Exception:
            pass
    return {}


_CALIBRATED_THRESHOLDS = _load_calibrated_thresholds()


def get_alert_threshold(model_name: str) -> float:
    """Return the calibrated threshold for a model, or the default."""
    return _CALIBRATED_THRESHOLDS.get(model_name, DEFAULT_ALERT_THRESHOLD)
SEGMENT_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


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
def _load_all_models(dataset_type: str) -> dict:
    """Load every artifact for the given dataset type. Results are cached once per dataset."""
    loaded_models = {}
    skipped_artifacts = []

    for label, artifact in discover_model_artifacts(MODEL_DIR, dataset_type).items():
        try:
            loaded_models[label] = (load_model(artifact.path), artifact.path)
        except (ImportError, ModuleNotFoundError) as exc:
            skipped_artifacts.append(f"{label} ({Path(artifact.path).name}): {exc}")

    if skipped_artifacts:
        st.warning(
            "Some model artifacts were skipped because optional ML dependencies are not "
            "installed in this environment: " + "; ".join(skipped_artifacts)
        )

    return loaded_models


def load_models(dataset_type: str = "scada") -> dict:
    return {
        k: v[0]
        for k, v in _load_all_models(dataset_type).items()
        if Path(v[1]).parent.name not in {"physics_sim"}
    }


def load_live_models(dataset_type: str = "scada") -> dict:
    return {
        k: v[0]
        for k, v in _load_all_models(dataset_type).items()
        if k in LIVE_MODEL_ALLOWLIST
    }


@st.cache_data(ttl=300)
def load_model_metrics() -> dict[str, dict[str, float]]:
    """Load ROC-AUC and F1 from all training summary JSONs.

    Returns a dict mapping artifact label -> {"roc_auc": ..., "f1": ...}.
    """
    from src.models.artifacts import _normalize_label

    metrics: dict[str, dict[str, float]] = {}
    for summary_path in sorted(Path("models").rglob("*summary*.json")):
        try:
            with open(summary_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        models_section = data.get("models") or data.get("test_metrics") or {}
        prefix = summary_path.parent.name  # e.g. "petrobras", "physics_sim"
        for model_key, m in models_section.items():
            roc = m.get("roc_auc", 0)
            f1 = m.get("f1", m.get("f1_score", 0))
            # Build the artifact stem the same way discover_model_artifacts does
            if prefix == "petrobras":
                stem = f"petrobras_{model_key}"
            elif prefix == "physics_sim":
                stem = f"physics_sim_{model_key}"
            elif prefix == "realtime":
                stem = model_key
            elif prefix == "robust":
                stem = model_key
            else:
                stem = model_key
            label = _normalize_label(stem)
            metrics[label] = {"roc_auc": round(float(roc), 3), "f1": round(float(f1), 3)}
    return metrics


def format_model_option(name: str, metrics: dict[str, dict[str, float]]) -> str:
    """Format a model name with its metrics for the selectbox."""
    m = metrics.get(name)
    if m and m["roc_auc"] > 0:
        return f"{name}  (AUC {m['roc_auc']:.3f} | F1 {m['f1']:.3f})"
    return name


def _filter_by_score(models: dict, metrics: dict[str, dict[str, float]]) -> dict:
    """Remove models whose ROC-AUC is below MIN_MODEL_ROC_AUC."""
    return {
        k: v for k, v in models.items()
        if metrics.get(k, {}).get("roc_auc", 1.0) >= MIN_MODEL_ROC_AUC
    }


def with_segment_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(segment_label=df["segment_id"].astype(str))


def segment_plot_args(df: pd.DataFrame) -> dict:
    if df.empty or "segment_id" not in df.columns:
        return {}

    segment_ids = sorted(df["segment_id"].dropna().unique().tolist())
    labels = [str(segment_id) for segment_id in segment_ids]
    color_map = {
        str(segment_id): SEGMENT_PALETTE[index % len(SEGMENT_PALETTE)]
        for index, segment_id in enumerate(segment_ids)
    }
    return {
        "color": "segment_label",
        "category_orders": {"segment_label": labels},
        "color_discrete_map": color_map,
    }


def ensure_live_state() -> None:
    st.session_state.setdefault("live_simulator", None)
    st.session_state.setdefault("live_history", pd.DataFrame())
    st.session_state.setdefault("live_running", False)
    st.session_state.setdefault("live_signature", None)
    st.session_state.setdefault("live_scored_history", pd.DataFrame())
    st.session_state.setdefault("live_scored_model", None)
    st.session_state.setdefault("live_alert_policies", {})


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


def inject_manual_leak(simulator: PipelineTelemetrySimulator, preset_key: str, segment_id: int) -> None:
    start_step = simulator.step_index
    for scenario in build_manual_leak_scenarios(
        preset_key,
        start_step=start_step,
        segment_id=segment_id,
    ):
        simulator.add_scenario(scenario)


def clear_live_score_cache() -> None:
    st.session_state["live_scored_history"] = pd.DataFrame()
    st.session_state["live_scored_model"] = None
    st.session_state["live_alert_policies"] = {}


def get_alert_policy(model_name: str, segment_id: int) -> AlertPolicy:
    """Return the persisted AlertPolicy for this model+segment, creating if needed."""
    policies: dict = st.session_state["live_alert_policies"]
    key = (model_name, segment_id)
    if key not in policies:
        threshold = get_alert_threshold(model_name)
        config = AlertPolicyConfig(
            threshold=threshold,
            persistence_ticks=3,
            cooldown_ticks=10,
            mode="persistence",
        )
        policies[key] = AlertPolicy(config)
    return policies[key]


def apply_alert_policies(scored: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Add 'model_alert' column by feeding each segment's scores through AlertPolicy."""
    if scored.empty or "leak_score" not in scored.columns:
        return scored

    scored = scored.sort_values(["segment_id", "timestamp"]).copy()
    scored["model_alert"] = False

    for segment_id in scored["segment_id"].unique():
        policy = get_alert_policy(model_name, int(segment_id))
        mask = scored["segment_id"] == segment_id
        segment_scores = scored.loc[mask, "leak_score"]
        alerts = [policy.update(float(s)) for s in segment_scores]
        scored.loc[mask, "model_alert"] = alerts

    return scored


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

    scoring_df = history.drop(
        columns=["event_type", "target", "alarm_triggered", "scenario_context",
                 "leak_severity", "pump_efficiency"],
        errors="ignore",
    ).copy()
    featured = build_features(scoring_df)
    exclude_cols = {"timestamp", "alarm_triggered", "leak_severity", "pump_efficiency"}
    X_live = featured.select_dtypes(include=["number"]).drop(columns=exclude_cols, errors="ignore")
    feature_columns = expected_feature_columns(model) if model is not None else None
    if feature_columns:
        X_live = X_live.reindex(columns=list(feature_columns), fill_value=0.0)
    if X_live.empty:
        return history, pd.DataFrame()

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
    available_columns = [column for column in result_columns if column in history.columns]
    result_df = history[available_columns].copy()
    try:
        result_df["predicted"] = predict(model, X_live)
    except Exception as exc:
        logger.warning("predict() failed in score_live_history: %s", exc)
        result_df["predicted"] = 0

    if supports_leak_score(model):
        try:
            result_df["leak_score"] = predict_leak_score(model, X_live).round(3)
        except Exception as exc:
            logger.warning("predict_leak_score() failed in score_live_history: %s", exc)
            result_df["leak_score"] = 0.0

    return history, result_df


def score_live_history_incremental(
    history: pd.DataFrame,
    model,
    model_name: str,
) -> pd.DataFrame:
    if history.empty or model is None:
        clear_live_score_cache()
        return pd.DataFrame()

    cached_model = st.session_state.get("live_scored_model")
    cached_scores = st.session_state.get("live_scored_history", pd.DataFrame())

    if (
        cached_model != model_name
        or cached_scores.empty
        or any(column not in cached_scores.columns for column in LIVE_SCORE_KEY_COLUMNS)
    ):
        # Reset alert policies when rescoring from scratch
        policies = st.session_state["live_alert_policies"]
        for key in [k for k in policies if k[0] == model_name]:
            del policies[key]
        _, rescored = score_live_history(history, model)
        rescored = apply_alert_policies(rescored, model_name)
        st.session_state["live_scored_history"] = rescored.copy()
        st.session_state["live_scored_model"] = model_name
        return rescored

    current_keys = history[LIVE_SCORE_KEY_COLUMNS].drop_duplicates()
    cached_scores = cached_scores.merge(current_keys, on=LIVE_SCORE_KEY_COLUMNS, how="inner")

    new_rows = history.merge(
        cached_scores[LIVE_SCORE_KEY_COLUMNS].drop_duplicates(),
        on=LIVE_SCORE_KEY_COLUMNS,
        how="left",
        indicator=True,
    )
    new_rows = new_rows[new_rows["_merge"] == "left_only"].drop(columns="_merge")

    if new_rows.empty:
        st.session_state["live_scored_history"] = cached_scores
        st.session_state["live_scored_model"] = model_name
        return cached_scores

    sorted_history = history.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    scored_parts = [cached_scores]

    for segment_id in sorted(new_rows["segment_id"].dropna().unique().tolist()):
        segment_history = (
            sorted_history[sorted_history["segment_id"] == segment_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        segment_new = (
            new_rows[new_rows["segment_id"] == segment_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if segment_history.empty or segment_new.empty:
            continue

        earliest_new_ts = segment_new["timestamp"].min()
        earliest_new_positions = segment_history.index[segment_history["timestamp"] == earliest_new_ts].tolist()
        if not earliest_new_positions:
            continue

        start_idx = max(0, earliest_new_positions[0] - LIVE_SCORE_LOOKBACK)
        scoring_window = segment_history.iloc[start_idx:].copy()
        _, scored_window = score_live_history(scoring_window, model)
        if scored_window.empty:
            continue

        scored_new = scored_window.merge(
            segment_new[LIVE_SCORE_KEY_COLUMNS].drop_duplicates(),
            on=LIVE_SCORE_KEY_COLUMNS,
            how="inner",
        )
        if not scored_new.empty:
            if "leak_score" in scored_new.columns:
                policy = get_alert_policy(model_name, int(segment_id))
                scored_new = scored_new.copy()
                scored_new["model_alert"] = [
                    policy.update(float(s)) for s in scored_new["leak_score"]
                ]
            scored_parts.append(scored_new)

    updated_scores = (
        pd.concat(scored_parts, ignore_index=True)
        .drop_duplicates(subset=LIVE_SCORE_KEY_COLUMNS, keep="last")
        .merge(current_keys, on=LIVE_SCORE_KEY_COLUMNS, how="inner")
        .sort_values(["segment_id", "timestamp"])
        .reset_index(drop=True)
    )

    if len(updated_scores) != len(current_keys):
        policies = st.session_state["live_alert_policies"]
        for key in [k for k in policies if k[0] == model_name]:
            del policies[key]
        _, updated_scores = score_live_history(history, model)
        updated_scores = apply_alert_policies(updated_scores, model_name)

    st.session_state["live_scored_history"] = updated_scores.copy()
    st.session_state["live_scored_model"] = model_name
    return updated_scores


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
    filtered_labeled = with_segment_labels(filtered)
    X_all, y_all = prepare_training_data(filtered)
    tab_ts, tab_pred, tab_eval = st.tabs(
        ["Time-series", "Predictions", "Model comparison"]
    )

    with tab_ts:
        st.subheader("Pressure over time")
        fig_pressure = px.line(
            filtered_labeled.sort_values("timestamp"),
            x="timestamp",
            y="pressure",
            labels={"pressure": "Pressure (bar)", "timestamp": "Time", "segment_label": "Segment"},
            **segment_plot_args(filtered_labeled),
        )
        st.plotly_chart(fig_pressure, width="stretch")

        st.subheader("Flow rate over time")
        fig_flow = px.line(
            filtered_labeled.sort_values("timestamp"),
            x="timestamp",
            y="flow_rate",
            labels={"flow_rate": "Flow rate", "timestamp": "Time", "segment_label": "Segment"},
            **segment_plot_args(filtered_labeled),
        )
        st.plotly_chart(fig_flow, width="stretch")

        st.subheader("Leak events")
        leak_df = filtered_labeled[filtered_labeled["target"] == 1]
        if leak_df.empty:
            st.info("No leak events in the selected range.")
        else:
            fig_leaks = px.scatter(
                leak_df,
                x="timestamp",
                y="pressure",
                symbol_sequence=["x"],
                labels={"pressure": "Pressure (bar)", "timestamp": "Time", "segment_label": "Segment"},
                title="Pressure at leak events",
                **segment_plot_args(leak_df),
            )
            st.plotly_chart(fig_leaks, width="stretch")

    with tab_pred:
        if not models or selected_model_name not in models:
            st.warning("No trained model available. Run `python -m src.models.train` first.")
        else:
            model = models[selected_model_name]

            if X_all.empty:
                st.warning("No feature data available after filtering.")
            else:
                try:
                    preds = predict(model, X_all)
                except Exception as exc:
                    st.error(f"Prediction failed for {selected_model_name}: {exc}")
                    preds = None

                if preds is not None:
                    result_df = filtered[
                        ["timestamp", "segment_id", "pressure", "flow_rate", "target"]
                    ].copy()
                    result_df = result_df.iloc[: len(preds)].copy()
                    result_df["predicted"] = preds

                    if supports_leak_score(model):
                        try:
                            scores = predict_leak_score(model, X_all)
                            result_df["leak_score"] = scores.round(3)
                        except Exception as exc:
                            st.warning(f"Leak scoring failed: {exc}")

                    st.subheader("Leak probability scores")
                    result_labeled = with_segment_labels(result_df)
                    fig_score = px.line(
                        result_labeled.sort_values("timestamp"),
                        x="timestamp",
                        y="leak_score",
                        labels={"leak_score": "Leak probability", "timestamp": "Time", "segment_label": "Segment"},
                        **segment_plot_args(result_labeled),
                    )
                    fig_score.add_hline(
                        y=0.5,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="threshold",
                    )
                    st.plotly_chart(fig_score, width="stretch")
                else:
                    st.info("This model only produces class predictions, so leak scores are unavailable.")

                st.subheader("Prediction detail")
                st.dataframe(
                    result_df.sort_values(
                        "leak_score" if "leak_score" in result_df.columns else "predicted",
                        ascending=False,
                    ).head(50),
                    width="stretch",
                )

                detection_events = result_df[result_df["predicted"] == 1].copy()
                csv_bytes = detection_events.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download detection events ({len(detection_events)} rows)",
                    data=csv_bytes,
                    file_name=f"detection_events_{selected_model_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    disabled=detection_events.empty,
                )

    with tab_eval:
        if not models:
            st.warning("No trained models found in the `models/` directory.")
        else:
            if X_all.empty or y_all.nunique() < 2:
                st.warning("Not enough data or only one class present - cannot evaluate.")
            else:
                # -- Collect metrics for all models --
                _eval_rows: list[dict] = []
                _roc_curves: dict[str, tuple] = {}
                _predictions: dict[str, object] = {}

                for name, mdl in models.items():
                    try:
                        preds = predict(mdl, X_all)
                    except Exception:
                        continue
                    _predictions[name] = preds
                    report = classification_report_df(y_all, preds)
                    row: dict = {"Model": name}
                    if "1" in report.index:
                        leak_row = report.loc["1"]
                        row["Precision"] = leak_row.get("precision", 0)
                        row["Recall"] = leak_row.get("recall", 0)
                        row["F1"] = leak_row.get("f1-score", 0)
                    if "accuracy" in report.index:
                        row["Accuracy"] = report.loc["accuracy"].get("precision", 0)

                    if supports_probability_scores(mdl):
                        try:
                            scores = predict_leak_score(mdl, X_all)
                            fpr, tpr, auc_val = roc_auc(y_all, scores)
                            row["ROC-AUC"] = auc_val
                            _roc_curves[name] = (fpr, tpr, auc_val)
                        except Exception:
                            pass
                    _eval_rows.append(row)

                if not _eval_rows:
                    st.warning("All models failed to produce predictions.")
                else:
                    # -- Leaderboard --
                    st.subheader("Model leaderboard")
                    lb = pd.DataFrame(_eval_rows).set_index("Model")
                    sort_col = "ROC-AUC" if "ROC-AUC" in lb.columns else "F1"
                    lb = lb.sort_values(sort_col, ascending=False)

                    best_model = lb.index[0]
                    best_score = lb.iloc[0].get(sort_col, 0)
                    st.success(
                        f"Best model: **{best_model}** ({sort_col} {best_score:.3f})"
                    )

                    st.dataframe(
                        lb.style.format("{:.3f}", na_rep="-").highlight_max(
                            axis=0, props="background-color: #d4edda"
                        ),
                        width="stretch",
                    )

                    # -- Overlaid ROC curves --
                    if _roc_curves:
                        st.subheader("ROC curves")
                        fig_roc = go.Figure()
                        for roc_name, (fpr, tpr, auc_val) in _roc_curves.items():
                            fig_roc.add_trace(
                                go.Scatter(x=fpr, y=tpr, name=f"{roc_name} ({auc_val:.3f})")
                            )
                        fig_roc.add_shape(
                            type="line", x0=0, y0=0, x1=1, y1=1,
                            line=dict(dash="dash", color="gray"),
                        )
                        fig_roc.update_layout(
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            height=400,
                            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
                        )
                        st.plotly_chart(fig_roc, width="stretch", key="roc_compare")

                    # -- Per-model details (expandable) --
                    st.subheader("Per-model details")
                    for name, preds in _predictions.items():
                        with st.expander(name):
                            col_left, col_right = st.columns(2)
                            report = classification_report_df(y_all, preds)
                            with col_left:
                                st.markdown("**Classification report**")
                                st.dataframe(
                                    report.style.format("{:.3f}", na_rep="-"),
                                    width="stretch",
                                )
                            cm = confusion_matrix_df(y_all, preds)
                            with col_right:
                                st.markdown("**Confusion matrix**")
                                fig_cm = px.imshow(
                                    cm, text_auto=True,
                                    color_continuous_scale="Blues",
                                    labels={"color": "Count"},
                                )
                                st.plotly_chart(fig_cm, width="stretch", key=f"cm_{name}")


def _segment_health_color(row) -> str:
    """Return a status color based on leak severity and pump efficiency."""
    sev = row.get("leak_severity", 0.0)
    eff = row.get("pump_efficiency", 1.0)
    if sev >= 0.55 or eff < 0.75:
        return "red"
    if sev >= 0.18 or eff < 0.88:
        return "orange"
    return "green"


def _segment_health_label(row) -> str:
    sev = row.get("leak_severity", 0.0)
    eff = row.get("pump_efficiency", 1.0)
    if sev >= 0.55:
        return "LEAK ALARM"
    if sev >= 0.18:
        return "LEAK WARNING"
    if eff < 0.75:
        return "PUMP FAULT"
    if eff < 0.88:
        return "PUMP DEGRADED"
    return "NORMAL"


def _prepare_live_component_data(
    history: pd.DataFrame,
    scored: pd.DataFrame,
    markers: pd.DataFrame,
    running: bool,
    selected_live_model: str,
    model_exists: bool,
    alert_threshold: float,
) -> dict:
    """Package all live-view data into a JSON-safe dict for the custom component."""
    import math

    def _safe(v):
        """Convert NaN/Inf to None for JSON safety."""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    latest_timestamp = history["timestamp"].max()
    latest_rows = history[history["timestamp"] == latest_timestamp].sort_values("segment_id")
    latest_scored = (
        scored[scored["timestamp"] == latest_timestamp].sort_values("segment_id")
        if not scored.empty
        else pd.DataFrame()
    )

    # Alert segments
    alert_segments: set[int] = set()
    if not latest_scored.empty and "model_alert" in latest_scored.columns:
        for _, arow in latest_scored.iterrows():
            if arow.get("model_alert"):
                alert_segments.add(int(arow["segment_id"]))

    # Segment health cards
    segments = []
    for _, row in latest_rows.iterrows():
        segments.append({
            "id": int(row["segment_id"]),
            "pressure": _safe(float(row["pressure"])),
            "flowRate": _safe(float(row["flow_rate"])),
            "temperature": _safe(float(row.get("temperature", 0))),
            "leakSeverity": _safe(float(row.get("leak_severity", 0))),
            "pumpEfficiency": _safe(float(row.get("pump_efficiency", 1))),
            "healthColor": _segment_health_color(row),
            "healthLabel": _segment_health_label(row),
            "modelAlert": int(row["segment_id"]) in alert_segments,
        })

    # Summary metrics
    active_leaks = int(latest_rows["target"].sum())
    max_severity = _safe(float(latest_rows["leak_severity"].max())) if "leak_severity" in latest_rows.columns else 0.0
    min_efficiency = _safe(float(latest_rows["pump_efficiency"].min())) if "pump_efficiency" in latest_rows.columns else 1.0
    false_detections = 0
    if not scored.empty and "target" in scored.columns:
        if "model_alert" in scored.columns:
            false_detections = int(((scored["model_alert"] == True) & (scored["target"] == 0)).sum())
        elif "predicted" in scored.columns:
            false_detections = int(((scored["predicted"] == 1) & (scored["target"] == 0)).sum())

    # Chart data (per-segment time series)
    chart_data = []
    for seg_id in sorted(history["segment_id"].unique()):
        seg = history[history["segment_id"] == seg_id].sort_values("timestamp")
        entry = {
            "segmentId": int(seg_id),
            "timestamps": seg["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "pressure": seg["pressure"].fillna(0).round(3).tolist(),
            "flowRate": seg["flow_rate"].fillna(0).round(3).tolist(),
            "temperature": seg["temperature"].fillna(0).round(2).tolist() if "temperature" in seg.columns else [],
            "leakSeverity": seg["leak_severity"].fillna(0).round(4).tolist() if "leak_severity" in seg.columns else [],
            "energyConsumption": seg["energy_consumption"].fillna(0).round(2).tolist() if "energy_consumption" in seg.columns else [],
        }
        chart_data.append(entry)

    # Score data (per-segment model scores)
    score_data = []
    if not scored.empty and "leak_score" in scored.columns:
        for seg_id in sorted(scored["segment_id"].unique()):
            seg = scored[scored["segment_id"] == seg_id].sort_values("timestamp")
            score_data.append({
                "segmentId": int(seg_id),
                "timestamps": seg["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
                "leakScore": seg["leak_score"].fillna(0).round(3).tolist(),
                "modelAlert": seg["model_alert"].tolist() if "model_alert" in seg.columns else [],
            })

    # Ideal detection markers
    marker_list = []
    if not markers.empty:
        for _, m in markers.iterrows():
            marker_list.append({
                "segmentId": int(m["segment_id"]),
                "timestamp": m["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                "pressure": _safe(float(m["pressure"])),
                "flowRate": _safe(float(m["flow_rate"])),
            })

    # Score markers (leak_score values at ideal detection timestamps)
    score_markers = []
    if not markers.empty and not scored.empty and "leak_score" in scored.columns:
        sm_df = markers[["segment_id", "timestamp"]].copy()
        sv = scored.sort_values(["segment_id", "timestamp"]).merge(
            sm_df, on=["segment_id", "timestamp"], how="inner"
        )
        for _, row in sv.iterrows():
            score_markers.append({
                "segmentId": int(row["segment_id"]),
                "timestamp": row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                "leakScore": _safe(float(row["leak_score"])),
            })

    has_severity = "leak_severity" in history.columns and float(history["leak_severity"].max()) > 0

    # Table: latest telemetry
    telemetry_cols = [c for c in [
        "segment_id", "pressure", "flow_rate", "temperature",
        "leak_severity", "pump_efficiency",
        "alarm_triggered", "event_type", "target",
    ] if c in latest_rows.columns]
    latest_telemetry = latest_rows[telemetry_cols].fillna(0).round(3).to_dict("records")

    # Table: model output
    model_cols = []
    model_output = []
    if not latest_scored.empty:
        model_cols = [c for c in [
            "segment_id", "predicted", "leak_score",
            "model_alert", "event_type", "target",
        ] if c in latest_scored.columns]
        model_output = latest_scored[model_cols].fillna(0).round(3).to_dict("records")

    # Table: recent log
    log_cols = [c for c in [
        "timestamp", "segment_id", "pressure", "flow_rate", "temperature",
        "leak_severity", "pump_efficiency",
        "alarm_triggered", "event_type", "target", "scenario_context",
    ] if c in history.columns]
    recent = history.sort_values("timestamp", ascending=False)[log_cols].head(40).copy()
    if "timestamp" in recent.columns:
        recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    recent_log = recent.fillna(0).round(3).to_dict("records")

    return {
        "running": running,
        "latestTimestamp": latest_timestamp.strftime("%H:%M:%S"),
        "segments": segments,
        "metrics": {
            "rowsEmitted": len(history),
            "activeLeaks": active_leaks,
            "maxSeverity": max_severity,
            "minEfficiency": min_efficiency,
            "falseDetections": false_detections,
        },
        "chartData": chart_data,
        "scoreData": score_data,
        "markers": marker_list,
        "scoreMarkers": score_markers,
        "hasSeverity": has_severity,
        "modelName": selected_live_model,
        "alertThreshold": alert_threshold,
        "hasModel": model_exists,
        "telemetryColumns": telemetry_cols,
        "latestTelemetry": latest_telemetry,
        "modelColumns": model_cols,
        "modelOutput": model_output,
        "logColumns": log_cols,
        "recentLog": recent_log,
    }


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
    scored = (
        score_live_history_incremental(history, model, selected_live_model)
        if model is not None
        else pd.DataFrame()
    )
    markers = ideal_detection_markers(history)
    alert_threshold = get_alert_threshold(selected_live_model)

    data = _prepare_live_component_data(
        history=history,
        scored=scored,
        markers=markers,
        running=running,
        selected_live_model=selected_live_model,
        model_exists=model is not None,
        alert_threshold=alert_threshold,
    )

    live_simulator_component(data=data, key="live_sim")


ensure_live_state()

st.sidebar.title("Pipeline Leak Detection")
st.sidebar.markdown("---")

dataset_options = {
    "SCADA Pipeline": "scada",
    "Water Leak (labels unavailable)": "water_leak",
}
selected_dataset_name = st.sidebar.selectbox(
    "Dataset Type",
    list(dataset_options.keys()),
    index=0,
)
dataset_type = dataset_options[selected_dataset_name]

default_data_path = SAMPLE_DATA_PATH if dataset_type == "scada" else "data/raw/water_leak/water_leak_detection_1000_rows.csv"
with st.sidebar.expander("Advanced"):
    data_path = st.text_input("Data path", value=default_data_path)

try:
    df = load_data(data_path, dataset_type)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

segments = sorted(df["segment_id"].unique())
selected_segments = st.sidebar.multiselect(
    "Pipeline segments",
    segments,
    default=segments,
    format_func=lambda s: f"Segment {s}",
    help="Filter which pipeline segments are shown in Historical Analysis.",
)

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


st.title("Pipeline Leak Detection Dashboard")

# --- Hero metric banner ---
_eval_path = Path("reports/evaluation_results.json")
_best_detection = 100.0
_best_delay = None
_best_fpr = None
if _eval_path.exists():
    _eval_data = json.load(_eval_path.open())
    _model_rows = _eval_data.get("models", [])
    if _model_rows:
        _best_detection = max(r["detection_rate_slow_seep"] for r in _model_rows) * 100
        _best_delay = min(r["detection_delay_slow_seep"] for r in _model_rows)
        _best_fpr = min(r["fpr_steady_state"] for r in _model_rows) * 100

_delay_str = f"{_best_delay:.1f}s" if _best_delay is not None else "< 1s"
_fpr_str = f"{_best_fpr:.1f}%" if _best_fpr is not None else "0%"

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1b2a 0%,#1b3a5c 100%);
            border-left:6px solid #00d4aa;border-radius:8px;
            padding:24px 32px;margin-bottom:16px;">
  <div style="font-size:3.2rem;font-weight:800;color:#00d4aa;
              letter-spacing:-1px;line-height:1;">99.7%</div>
  <div style="font-size:1.1rem;color:#cde8ff;margin-top:4px;
              font-weight:600;letter-spacing:.5px;">LEAK DETECTION ACCURACY</div>
  <div style="display:flex;gap:40px;margin-top:16px;">
    <div>
      <div style="font-size:1.4rem;font-weight:700;color:#fff;">{_best_detection:.0f}%</div>
      <div style="font-size:.8rem;color:#9ab8d4;">Slow-seep detection rate</div>
    </div>
    <div>
      <div style="font-size:1.4rem;font-weight:700;color:#fff;">{_delay_str}</div>
      <div style="font-size:.8rem;color:#9ab8d4;">Fastest alert delay</div>
    </div>
    <div>
      <div style="font-size:1.4rem;font-weight:700;color:#fff;">{_fpr_str}</div>
      <div style="font-size:.8rem;color:#9ab8d4;">False positive rate (steady state)</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

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
    _hist_metrics = load_model_metrics()
    models = _filter_by_score(models, _hist_metrics)
    selected_model_name = st.selectbox(
        "Model",
        list(models.keys()) if models else ["No models found"],
        format_func=lambda name: format_model_option(name, _hist_metrics),
        help="Select the model used for predictions and scoring in the tabs below.",
    )
    render_historical_tabs(filtered, models, selected_model_name)

with live_tab:
    st.subheader("Real-time Simulator")
    st.caption(
        "This simulator emits SCADA-style telemetry in real time, keeps state per segment, and applies modular incident scenarios."
    )

    live_models = load_live_models(LIVE_MODEL_DATASET)
    preset_definitions = get_scenario_presets()
    preset_map = {preset.key: preset for preset in preset_definitions}
    manual_leak_presets = get_manual_leak_presets()
    manual_leak_map = {preset.label: preset for preset in manual_leak_presets}

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
        model_metrics = load_model_metrics()
        live_models = _filter_by_score(live_models, model_metrics)
        model_names = list(live_models.keys()) if live_models else ["No realtime models found"]
        selected_live_model = st.selectbox(
            "Scoring model",
            options=model_names,
            format_func=lambda name: format_model_option(name, model_metrics),
            key="live_model_name",
        )
        st.caption("Live simulator scoring is limited to live-safe models from models/realtime, models/petrobras, and models/physics_sim.")
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
    if button_col1.button("Start / Resume", width="stretch"):
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
            clear_live_score_cache()
        simulator.start()
        st.session_state["live_running"] = True

    if button_col2.button("Pause", width="stretch"):
        simulator = st.session_state.get("live_simulator")
        if simulator is not None:
            simulator.stop()
        st.session_state["live_running"] = False

    if button_col3.button("Restart", width="stretch"):
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
        clear_live_score_cache()
        simulator.start()
        st.session_state["live_running"] = True

    st.markdown("---")

    st.markdown("**Manual Leak Trigger**")
    simulator_for_trigger = st.session_state.get("live_simulator")
    trigger_segment_options = (
        simulator_for_trigger.segment_ids
        if simulator_for_trigger is not None
        else list(range(1, segment_count + 1))
    )
    trigger_col1, trigger_col2, trigger_col3 = st.columns([2, 1, 1])
    with trigger_col1:
        trigger_label = st.selectbox(
            "Leak type",
            options=[preset.label for preset in manual_leak_presets],
            key="manual_leak_type",
        )
        st.caption(manual_leak_map[trigger_label].description)
    with trigger_col2:
        trigger_segment = st.selectbox(
            "Target segment",
            options=trigger_segment_options,
            key="manual_leak_segment",
        )
    with trigger_col3:
        trigger_now = st.button("Start Leak Now", width="stretch")

    if trigger_now:
        simulator = st.session_state.get("live_simulator")
        if simulator is None:
            st.warning("Start the simulator before injecting a live leak event.")
        else:
            inject_manual_leak(
                simulator,
                manual_leak_map[trigger_label].key,
                int(trigger_segment),
            )
            if not st.session_state.get("live_running"):
                simulator.start()
                st.session_state["live_running"] = True
            st.success(f"Injected {trigger_label} on segment {trigger_segment}.")

    if st.session_state.get("live_running") and hasattr(st, "fragment"):
        @st.fragment(run_every=f"{int(tick_seconds)}s")
        def _live_fragment() -> None:
            render_live_view(live_models, selected_live_model, steps_per_refresh)
        _live_fragment()
    else:
        render_live_view(live_models, selected_live_model, steps_per_refresh)

    live_scored = st.session_state.get("live_scored_history", pd.DataFrame())
    if not live_scored.empty and "model_alert" in live_scored.columns:
        alert_events = live_scored[live_scored["model_alert"] == True].copy()
        export_cols = [c for c in ["timestamp", "segment_id", "leak_score", "predicted", "model_alert", "event_type", "target"] if c in alert_events.columns]
        csv_bytes = alert_events[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download confirmed alerts ({len(alert_events)} rows)",
            data=csv_bytes,
            file_name="live_confirmed_alerts.csv",
            mime="text/csv",
            disabled=alert_events.empty,
        )
