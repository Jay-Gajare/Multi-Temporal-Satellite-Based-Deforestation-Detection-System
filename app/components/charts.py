"""Reusable chart components using Plotly."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#F9FAFB", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    hoverlabel=dict(bgcolor="#1F2937", font_color="#F9FAFB", font_size=13, bordercolor="#374151"),
    height=350,
)


def _apply_layout(fig: go.Figure, title: str = "", height: int = 350) -> go.Figure:
    layout = dict(PLOTLY_LAYOUT, height=height)
    if title:
        layout["title"] = dict(text=title, font=dict(size=16, color="#F9FAFB"), x=0.02, y=0.97)
    fig.update_layout(**layout)
    return fig


def prediction_distribution(probabilities: list[float], threshold: float = 0.5) -> go.Figure:
    """Histogram of prediction probabilities."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=probabilities, nbinsx=40,
        marker=dict(
            color=np.where(np.array(probabilities) >= threshold, "rgba(239,68,68,0.6)", "rgba(0,200,150,0.6)"),
            line=dict(width=0),
        ),
        name="Predictions",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#F59E0B", line_width=2,
                  annotation_text=f"Threshold: {threshold:.2f}", annotation_font_color="#F59E0B")
    fig.update_layout(barmode="overlay", xaxis_title="Probability", yaxis_title="Count")
    return _apply_layout(fig, "Prediction Distribution")


def confusion_matrix_plot(cm: list[list[int]]) -> go.Figure:
    """Interactive confusion matrix."""
    labels = ["No Deforestation", "Deforestation"]
    text = [[f"{v}<br>({v/sum(row)*100:.0f}%)" if sum(row) > 0 else str(v) for v in row] for row in cm]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, text=text, texttemplate="%{text}",
        colorscale=[[0, "#111827"], [0.5, "#1e3a5f"], [1, "#3B82F6"]],
        showscale=False, hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="True", height=350)
    return _apply_layout(fig, "Confusion Matrix")


def roc_curve_plot(fpr: list[float], tpr: list[float], auc: float) -> go.Figure:
    """ROC curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.4f})",
                             line=dict(color="#3B82F6", width=3), fill="tozeroy",
                             fillcolor="rgba(59,130,246,0.1)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="#6B7280", width=1, dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _apply_layout(fig, "ROC Curve")


def pr_curve_plot(precision: list[float], recall: list[float], auc: float) -> go.Figure:
    """Precision-Recall curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AUC = {auc:.4f})",
                             line=dict(color="#00C896", width=3), fill="tozeroy",
                             fillcolor="rgba(0,200,150,0.1)"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return _apply_layout(fig, "Precision-Recall Curve")


def latency_histogram(latencies: list[float], bins: int = 30) -> go.Figure:
    """Histogram of inference latencies."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=latencies, nbinsx=bins,
                               marker=dict(color="rgba(99,102,241,0.6)", line=dict(width=0))))
    fig.update_layout(xaxis_title="Latency (ms)", yaxis_title="Count")
    return _apply_layout(fig, "Inference Latency Distribution")


def gauge_chart(value: float, title: str = "", max_val: float = 1.0, suffix: str = "") -> go.Figure:
    """Gauge chart for a metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(suffix=suffix, font=dict(size=28, color="#F9FAFB")),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor="#6B7280"),
            bar=dict(color="#00C896"),
            bgcolor="#1F2937",
            borderwidth=0,
            steps=[dict(range=[0, max_val * 0.6], color="rgba(239,68,68,0.15)"),
                   dict(range=[max_val * 0.6, max_val * 0.8], color="rgba(245,158,11,0.15)"),
                   dict(range=[max_val * 0.8, max_val], color="rgba(0,200,150,0.15)")],
            threshold=dict(line=dict(color="#F9FAFB", width=2), thickness=0.8, value=value),
        ),
    ))
    return _apply_layout(fig, title, height=280)


def pie_chart(labels: list[str], values: list[str], colors: Optional[list[str]] = None) -> go.Figure:
    """Pie chart."""
    default_colors = ["#00C896", "#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors or default_colors[:len(labels)], line=dict(color="#0B1220", width=2)),
        textfont=dict(size=13, color="#F9FAFB"),
        hole=0.45,
    ))
    return _apply_layout(fig, height=350)


def bar_chart(x: list, y: list, title: str = "", color: str = "#3B82F6", horizontal: bool = False) -> go.Figure:
    """Bar chart."""
    if horizontal:
        fig = go.Figure(go.Bar(x=y, y=x, orientation="h", marker=dict(color=color, line=dict(width=0))))
        fig.update_layout(yaxis=dict(autorange="reversed"))
    else:
        fig = go.Figure(go.Bar(x=x, y=y, marker=dict(color=color, line=dict(width=0))))
    return _apply_layout(fig, title)


def training_curves(train_loss: list, val_loss: list, train_f1: list, val_f1: list) -> go.Figure:
    """Training curves overlay."""
    epochs = list(range(1, len(train_loss) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss", line=dict(color="#3B82F6", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss", line=dict(color="#EF4444", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=train_f1, name="Train F1", line=dict(color="#00C896", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=epochs, y=val_f1, name="Val F1", line=dict(color="#F59E0B", width=2, dash="dash")))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Value", legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)))
    return _apply_layout(fig, "Training History")
