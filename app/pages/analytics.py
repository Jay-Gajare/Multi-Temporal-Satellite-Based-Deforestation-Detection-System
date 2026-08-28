"""Analytics — interactive model performance analysis with Plotly charts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.charts import (
    prediction_distribution, confusion_matrix_plot, roc_curve_plot,
    pr_curve_plot, latency_histogram, gauge_chart, pie_chart, bar_chart,
    training_curves, PLOTLY_LAYOUT,
)
from app.components.cards import glass_card
from app.components.layout import (
    page_header, section_header, divider, footer, info_banner,
)


# ── Data Loading ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading evaluation data…")
def _load_test_results() -> dict:
    p = Path("models/run_01/test_evaluation/test_results.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600, show_spinner="Loading test predictions…")
def _load_test_predictions() -> pd.DataFrame:
    p = Path("models/run_01/test_evaluation/test_predictions.csv")
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Loading verification…")
def _load_verification() -> dict:
    p = Path("outputs/verification_results.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600, show_spinner="Loading training history…")
def _load_training_history() -> pd.DataFrame:
    p = Path("models/run_01/training_history.csv")
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _compute_roc(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Compute ROC curve without sklearn."""
    desc = np.argsort(-y_score)
    y_true_sorted = y_true[desc]
    P = y_true.sum()
    N = len(y_true) - P
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    tpr = tp / max(P, 1)
    fpr = fp / max(N, 1)
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return fpr.tolist(), tpr.tolist()


def _compute_pr(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Compute Precision-Recall curve without sklearn."""
    desc = np.argsort(-y_score)
    y_true_sorted = y_true[desc]
    tp = np.cumsum(y_true_sorted)
    total_pos = y_true.sum()
    recall = tp / max(total_pos, 1)
    precision = tp / np.maximum(np.arange(1, len(tp) + 1), 1)
    recall = np.concatenate([[0], recall])
    precision = np.concatenate([[1], precision])
    return recall.tolist(), precision.tolist()


def _auc(x: list, y: list) -> float:
    """Compute AUC via trapezoidal rule."""
    x, y = np.array(x), np.array(y)
    return float(np.trapz(y, x))


# ── Plotly Layout Override for Analytics ──────────────────────────
def _chart_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F9FAFB", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        hoverlabel=dict(bgcolor="#1F2937", font_color="#F9FAFB", font_size=13, bordercolor="#374151"),
        height=height,
    )
    if title:
        fig.update_layout(title=dict(
            text=title, font=dict(size=17, color="#F9FAFB", family="Inter, sans-serif"),
            x=0.02, y=0.97,
        ))
    return fig


def render() -> None:
    page_header(
        "Analytics",
        "Model evaluation, ROC analysis, and prediction insights",
        icon="📊",
        icon_gradient="var(--grad-accent)",
    )

    results = _load_test_results()
    preds_df = _load_test_predictions()
    verify = _load_verification()
    history = _load_training_history()

    if results.empty if isinstance(results, pd.DataFrame) else not results:
        info_banner("No evaluation data found. Run model evaluation first.", icon="⚠️")
        return

    m05 = results.get("metrics_at_05", {})
    cm05 = results.get("confusion_matrix_05", [[0, 0], [0, 0]])
    threshold_analysis = results.get("threshold_analysis", [])

    # ── Gauge Charts Row ──────────────────────────────────────────
    section_header("Key Metrics", "Test set @ threshold 0.50", "📈")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.plotly_chart(gauge_chart(m05.get("accuracy", 0), "Accuracy", 1.0, "%"),
                        use_container_width=True, key="g_acc")
    with c2:
        st.plotly_chart(gauge_chart(m05.get("f1", 0), "F1 Score", 1.0, "%"),
                        use_container_width=True, key="g_f1")
    with c3:
        st.plotly_chart(gauge_chart(m05.get("precision", 0), "Precision", 1.0, "%"),
                        use_container_width=True, key="g_prec")
    with c4:
        st.plotly_chart(gauge_chart(m05.get("recall", 0), "Recall", 1.0, "%"),
                        use_container_width=True, key="g_recall")

    divider()

    # ── Row 1: Prediction Distribution + Confidence Histogram ──────
    section_header("Prediction Analysis", "Probability distributions across 502 test patches", "📊")
    c1, c2 = st.columns(2)

    with c1:
        if not preds_df.empty:
            probs = preds_df["probability"].values
            threshold = 0.5

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=probs, nbinsx=40, name="All Patches",
                marker=dict(
                    color=np.where(probs >= threshold, "rgba(239,68,68,0.55)", "rgba(0,200,150,0.55)"),
                    line=dict(width=0),
                ),
            ))
            fig.add_vline(x=threshold, line_dash="dash", line_color="#F59E0B", line_width=2,
                          annotation_text=f"t={threshold}", annotation_font_color="#F59E0B",
                          annotation_font_size=11)
            fig.update_layout(
                xaxis_title="Predicted Probability", yaxis_title="Count",
                bargap=0.05,
            )
            _chart_layout(fig, "Prediction Distribution", 380)
            st.plotly_chart(fig, use_container_width=True, key="pred_dist")

    with c2:
        if not preds_df.empty:
            pos_probs = preds_df[preds_df["true_label"] == 1]["probability"].values
            neg_probs = preds_df[preds_df["true_label"] == 0]["probability"].values

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pos_probs, nbinsx=30, name="Deforestation (y=1)",
                marker=dict(color="rgba(239,68,68,0.5)", line=dict(width=0)),
                opacity=0.75,
            ))
            fig.add_trace(go.Histogram(
                x=neg_probs, nbinsx=30, name="No Deforestation (y=0)",
                marker=dict(color="rgba(0,200,150,0.5)", line=dict(width=0)),
                opacity=0.75,
            ))
            fig.update_layout(
                barmode="overlay", xaxis_title="Predicted Probability", yaxis_title="Count",
                legend=dict(orientation="h", y=1.12, font=dict(size=11)),
                bargap=0.05,
            )
            _chart_layout(fig, "Confidence by True Class", 380)
            st.plotly_chart(fig, use_container_width=True, key="conf_hist")

    divider()

    # ── Row 2: Latency Histogram + Class Distribution ──────────────
    section_header("Latency & Class Balance", "", "⏱️")
    c1, c2 = st.columns(2)

    with c1:
        bench = verify.get("benchmark", {})
        lat = bench.get("latency", {})
        inf_ms = lat.get("time_inference_s", {}).get("mean_ms", 27)
        grad_ms = lat.get("time_gradcam_s", {}).get("mean_ms", 76)
        total_ms = lat.get("time_total_s", {}).get("mean_ms", 2600)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Validate", "File Load", "Preprocess", "Tensor", "Inference", "Grad-CAM", "Visualize", "Save"],
            y=[
                lat.get("time_validate_s", {}).get("mean_ms", 0),
                lat.get("time_fileload_s", {}).get("mean_ms", 0),
                lat.get("time_preprocess_s", {}).get("mean_ms", 0),
                lat.get("time_tensor_s", {}).get("mean_ms", 0),
                lat.get("time_inference_s", {}).get("mean_ms", 0),
                lat.get("time_gradcam_s", {}).get("mean_ms", 0),
                lat.get("time_visualize_s", {}).get("mean_ms", 0),
                lat.get("time_save_s", {}).get("mean_ms", 0),
            ],
            marker=dict(
                color=["#F59E0B", "#3B82F6", "#8B5CF6", "#6B7280", "#00C896", "#EF4444", "#F97316", "#9CA3AF"],
                line=dict(width=0),
            ),
            text=[f"{v:.1f}ms" for v in [
                lat.get("time_validate_s", {}).get("mean_ms", 0),
                lat.get("time_fileload_s", {}).get("mean_ms", 0),
                lat.get("time_preprocess_s", {}).get("mean_ms", 0),
                lat.get("time_tensor_s", {}).get("mean_ms", 0),
                lat.get("time_inference_s", {}).get("mean_ms", 0),
                lat.get("time_gradcam_s", {}).get("mean_ms", 0),
                lat.get("time_visualize_s", {}).get("mean_ms", 0),
                lat.get("time_save_s", {}).get("mean_ms", 0),
            ]],
            textposition="outside", textfont=dict(size=10),
        ))
        fig.update_layout(xaxis_title="Component", yaxis_title="Mean Latency (ms)")
        _chart_layout(fig, "Latency Breakdown", 380)
        st.plotly_chart(fig, use_container_width=True, key="latency_bar")

    with c2:
        if not preds_df.empty:
            label_counts = preds_df["true_label"].value_counts()
            n_defor = int(label_counts.get(1, 0))
            n_safe = int(label_counts.get(0, 0))

            fig = go.Figure(go.Pie(
                labels=["Deforestation", "No Deforestation"],
                values=[n_defor, n_safe],
                marker=dict(colors=["#EF4444", "#00C896"], line=dict(color="#0B1220", width=3)),
                textfont=dict(size=13, color="#F9FAFB"),
                hole=0.5,
                textinfo="label+value+percent",
            ))
            _chart_layout(fig, "Class Distribution", 380)
            st.plotly_chart(fig, use_container_width=True, key="class_pie")

    divider()

    # ── Row 3: Confusion Matrix ────────────────────────────────────
    section_header("Confusion Matrix", "502 test samples @ threshold 0.50", "🔢")
    c1, c2 = st.columns([3, 2])

    with c1:
        tn, fp = cm05[0]
        fn, tp = cm05[1]
        labels = ["No Deforestation", "Deforestation"]
        cm_text = [[f"{v}<br>({v/sum(row)*100:.0f}%)" if sum(row) > 0 else str(v)
                     for v in row] for row in cm05]

        fig = go.Figure(data=go.Heatmap(
            z=cm05, x=labels, y=labels, text=cm_text, texttemplate="%{text}",
            colorscale=[[0, "#0B1527"], [0.3, "#1e3a5f"], [0.6, "#2563EB"], [1, "#3B82F6"]],
            showscale=False,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            textfont=dict(size=16, color="white"),
        ))
        fig.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
        _chart_layout(fig, "", 380)
        st.plotly_chart(fig, use_container_width=True, key="cm_plot")

    with c2:
        st.markdown(f'''
        <div class="glass-card glass-card--no-hover" style="padding:20px 24px;">
            <div style="font-size:13px;font-weight:700;margin-bottom:16px;color:var(--text-primary);">
                Confusion Matrix Breakdown
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div style="background:rgba(0,200,150,0.08);border:1px solid rgba(0,200,150,0.15);border-radius:10px;padding:14px;text-align:center;">
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-muted);font-weight:600;">True Negatives</div>
                    <div style="font-size:28px;font-weight:800;color:#00C896;margin:4px 0;font-family:var(--font-mono);">{tn}</div>
                    <div style="font-size:11px;color:var(--text-muted);">Correctly predicted safe</div>
                </div>
                <div style="background:rgba(0,200,150,0.08);border:1px solid rgba(0,200,150,0.15);border-radius:10px;padding:14px;text-align:center;">
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-muted);font-weight:600;">True Positives</div>
                    <div style="font-size:28px;font-weight:800;color:#00C896;margin:4px 0;font-family:var(--font-mono);">{tp}</div>
                    <div style="font-size:11px;color:var(--text-muted);">Correctly detected deforestation</div>
                </div>
                <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.15);border-radius:10px;padding:14px;text-align:center;">
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-muted);font-weight:600;">False Positives</div>
                    <div style="font-size:28px;font-weight:800;color:#F59E0B;margin:4px 0;font-family:var(--font-mono);">{fp}</div>
                    <div style="font-size:11px;color:var(--text-muted);">False deforestation alerts</div>
                </div>
                <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.15);border-radius:10px;padding:14px;text-align:center;">
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-muted);font-weight:600;">False Negatives</div>
                    <div style="font-size:28px;font-weight:800;color:#EF4444;margin:4px 0;font-family:var(--font-mono);">{fn}</div>
                    <div style="font-size:11px;color:var(--text-muted);">Missed deforestation events</div>
                </div>
            </div>
        </div>''', unsafe_allow_html=True)

    divider()

    # ── Row 4: ROC Curve + Precision-Recall Curve ──────────────────
    section_header("ROC & Precision-Recall Curves", "Threshold-independent model evaluation", "📈")
    c1, c2 = st.columns(2)

    with c1:
        if not preds_df.empty:
            y_true = preds_df["true_label"].values.astype(int)
            y_score = preds_df["probability"].values
            fpr, tpr = _compute_roc(y_true, y_score)
            roc_auc = _auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.4f})",
                line=dict(color="#3B82F6", width=3),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC = 0.5)",
                line=dict(color="#6B7280", width=1, dash="dash"),
            ))
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(orientation="h", y=1.12, font=dict(size=11)),
                xaxis=dict(range=[-0.02, 1.02]),
                yaxis=dict(range=[-0.02, 1.02]),
            )
            _chart_layout(fig, "ROC Curve", 420)
            st.plotly_chart(fig, use_container_width=True, key="roc_curve")

    with c2:
        if not preds_df.empty:
            y_true = preds_df["true_label"].values.astype(int)
            y_score = preds_df["probability"].values
            recall_arr, precision_arr = _compute_pr(y_true, y_score)
            pr_auc = _auc(recall_arr, precision_arr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recall_arr, y=precision_arr, mode="lines",
                name=f"PR (AUC = {pr_auc:.4f})",
                line=dict(color="#00C896", width=3),
                fill="tozeroy", fillcolor="rgba(0,200,150,0.08)",
            ))
            baseline = y_true.sum() / len(y_true)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[baseline, baseline], mode="lines",
                name=f"Baseline ({baseline:.2f})",
                line=dict(color="#6B7280", width=1, dash="dash"),
            ))
            fig.update_layout(
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend=dict(orientation="h", y=1.12, font=dict(size=11)),
                xaxis=dict(range=[-0.02, 1.02]),
                yaxis=dict(range=[0, 1.05]),
            )
            _chart_layout(fig, "Precision-Recall Curve", 420)
            st.plotly_chart(fig, use_container_width=True, key="pr_curve")

    divider()

    # ── Row 5: Threshold Analysis ──────────────────────────────────
    if threshold_analysis:
        section_header("Threshold Analysis", "F1 / Precision / Recall vs Decision Threshold", "📏")
        thresholds = [t["threshold"] for t in threshold_analysis]
        f1s = [t["f1"] for t in threshold_analysis]
        precs = [t["precision"] for t in threshold_analysis]
        recs = [t["recall"] for t in threshold_analysis]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=f1s, name="F1 Score",
                                 line=dict(color="#F59E0B", width=3)))
        fig.add_trace(go.Scatter(x=thresholds, y=precs, name="Precision",
                                 line=dict(color="#3B82F6", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=thresholds, y=recs, name="Recall",
                                 line=dict(color="#00C896", width=2, dash="dot")))

        best_t = results.get("best_threshold", 0.1)
        fig.add_vline(x=best_t, line_dash="dash", line_color="#EF4444", line_width=2,
                      annotation_text=f"Best: t={best_t}",
                      annotation_font_color="#EF4444", annotation_font_size=11)

        fig.update_layout(
            xaxis_title="Threshold", yaxis_title="Score",
            legend=dict(orientation="h", y=1.12, font=dict(size=11)),
            xaxis=dict(range=[0.05, 0.95]),
            yaxis=dict(range=[0.75, 0.95]),
        )
        _chart_layout(fig, "", 400)
        st.plotly_chart(fig, use_container_width=True, key="threshold_sweep")

        divider()

    # ── Row 6: Training Curves ─────────────────────────────────────
    if not history.empty:
        section_header("Training History", f"{len(history)} epochs · Best epoch {results.get('checkpoint_epoch', '?')}", "📉")
        train_loss = history["train_loss"].tolist()
        val_loss = history["val_loss"].tolist()
        train_f1 = history["train_f1"].tolist()
        val_f1 = history["val_f1"].tolist()
        epochs = list(range(1, len(train_loss) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss",
                                 line=dict(color="#3B82F6", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss",
                                 line=dict(color="#EF4444", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=train_f1, name="Train F1",
                                 line=dict(color="#00C896", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=epochs, y=val_f1, name="Val F1",
                                 line=dict(color="#F59E0B", width=2, dash="dash")))

        best_ep = results.get("checkpoint_epoch", 27)
        fig.add_vline(x=best_ep, line_dash="dash", line_color="#8B5CF6", line_width=2,
                      annotation_text=f"Best ({best_ep})",
                      annotation_font_color="#8B5CF6", annotation_font_size=11)

        fig.update_layout(
            xaxis_title="Epoch", yaxis_title="Value",
            legend=dict(orientation="h", y=1.12, font=dict(size=11)),
        )
        _chart_layout(fig, "Training History", 400)
        st.plotly_chart(fig, use_container_width=True, key="train_curves")

    footer()
