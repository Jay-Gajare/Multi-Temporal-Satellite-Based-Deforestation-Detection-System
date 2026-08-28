"""Model Details — architecture, training config, evaluation results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.cards import metric_card
from app.components.cards import glass_card
from app.components.layout import (
    page_header, section_header, divider, footer, pipeline_step, kv_grid,
)
from app.components.charts import training_curves, confusion_matrix_plot


@st.cache_data(ttl=3600, show_spinner="Loading test results…")
def _load_test_results() -> dict:
    p = Path("models/run_01/test_evaluation/test_results.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


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


def render() -> None:
    page_header(
        "Model Details",
        "Architecture, training configuration, and evaluation results",
        icon="🧠",
        icon_gradient="var(--grad-accent)",
    )

    # ── Load Data ─────────────────────────────────────────────────
    results = _load_test_results()
    verify = _load_verification()
    history = _load_training_history()

    # ── Architecture ──────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])

    with c1:
        glass_card(title="Architecture", subtitle="ResNet18 backbone", icon="🏗️", icon_gradient="var(--grad-primary)")
        pipeline_step(1, "Backbone", "ResNet18 (pretrained=False) — 11.51M parameters")
        pipeline_step(2, "Input", "108 channels × 64 × 64 (12 months × 9 features)")
        pipeline_step(3, "Output", "Binary classification (sigmoid activation)")
        pipeline_step(4, "Regularization", "Dropout(0.3) + Weight Decay (1e-4)")
        pipeline_step(5, "Optimizer", "AdamW (lr=1e-4, weight_decay=1e-4)")
        pipeline_step(6, "Scheduler", "CosineAnnealingLR with 3-epoch warmup")

    with c2:
        section_header("Quick Stats", "", "📊")
        metric_card("Parameters", "11.51M", icon="🧠", gradient="var(--grad-accent)")
        metric_card("Input Channels", "108", icon="📥", gradient="var(--grad-cool)")
        metric_card("Spatial Size", "64×64", icon="📐", gradient="var(--grad-primary)")
        best_ep = results.get("checkpoint_epoch", 27)
        total_ep = results.get("total_epochs_run", 37)
        metric_card("Best Epoch", f"{best_ep} / {total_ep}", icon="⭐", gradient="var(--grad-warm)")

    divider()

    # ── Training Config ───────────────────────────────────────────
    glass_card(title="Training Configuration", subtitle="Hyperparameters and settings", icon="⚙️", icon_gradient="var(--grad-cool)")

    kv_grid([
        {"key": "Seed", "value": "42", "icon": "🎲"},
        {"key": "Batch Size", "value": "32", "icon": "📦"},
        {"key": "Learning Rate", "value": "1e-4", "icon": "📈"},
        {"key": "Weight Decay", "value": "1e-4", "icon": "⚖️"},
        {"key": "Max Epochs", "value": "50 (early stop: 10)", "icon": "🔄"},
        {"key": "Warmup", "value": "3 epochs", "icon": "🔥"},
        {"key": "Loss", "value": "BCEWithLogitsLoss", "icon": "📉"},
        {"key": "Optimizer", "value": "AdamW", "icon": "⚙️"},
        {"key": "Scheduler", "value": "CosineAnnealingLR", "icon": "📅"},
        {"key": "Input Shape", "value": "(B, 108, 64, 64)", "icon": "📐"},
        {"key": "Dropout", "value": "0.3", "icon": "🎲"},
        {"key": "Device", "value": "CPU", "icon": "🖥️"},
    ], columns=4)

    divider()

    # ── Training History ──────────────────────────────────────────
    section_header("Training History", "", "📈")
    if not history.empty:
        n_epochs = len(history)
        best_ep = results.get("checkpoint_epoch", 27)
        st.info(f"Training completed for {n_epochs} epochs (early stopped). "
                f"Best model saved at epoch {best_ep}.")

        train_loss = history["train_loss"].tolist()
        val_loss = history["val_loss"].tolist()
        train_f1 = history["train_f1"].tolist()
        val_f1 = history["val_f1"].tolist()

        st.plotly_chart(training_curves(train_loss, val_loss, train_f1, val_f1),
                        width='stretch', key="detail_training_curves")
    else:
        st.warning("Training history not found. Run training first.")

    divider()

    # ── Test Evaluation ───────────────────────────────────────────
    section_header("Test Evaluation", "502 held-out test patches", "📊")

    m05 = results.get("metrics_at_05", {})
    cm05 = results.get("confusion_matrix_05", [[0, 0], [0, 0]])
    best_t = results.get("best_threshold", 0.10)

    # Try to get metrics from verification first, then test_results
    metrics = verify.get("metrics", {})
    if not metrics:
        metrics = {
            "accuracy": m05.get("accuracy", 0),
            "f1": m05.get("f1", 0),
            "precision": m05.get("precision", 0),
            "recall": m05.get("recall", 0),
            "roc_auc": m05.get("roc_auc", 0),
            "confusion_matrix": cm05,
        }

    cm = metrics.get("confusion_matrix", cm05)

    # ── Metric Cards ─────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Accuracy", f"{metrics.get('accuracy', 0):.4f}", icon="🎯", gradient="var(--grad-primary)")
    with c2:
        metric_card("F1", f"{metrics.get('f1', 0):.4f}", icon="📊", gradient="var(--grad-accent)")
    with c3:
        metric_card("Precision", f"{metrics.get('precision', 0):.4f}", icon="✅", gradient="var(--grad-cool)")
    with c4:
        metric_card("Recall", f"{metrics.get('recall', 0):.4f}", icon="🔍", gradient="var(--grad-warm)")
    with c5:
        metric_card("MCC", f"{metrics.get('mcc', 0):.4f}", icon="📐", gradient="var(--grad-safe)")
    with c6:
        metric_card("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}", icon="📈", gradient="var(--grad-warm)")

    st.plotly_chart(confusion_matrix_plot(cm), width='stretch', key="detail_cm")

    # ── Best Threshold Info ───────────────────────────────────────
    info_banner(
        f"Best threshold = {best_t:.2f} (F1 = {results.get('best_f1', 0):.4f})",
        icon="📏",
        sub="Threshold-independent evaluation",
    )

    # ── Dataset Info ──────────────────────────────────────────────
    divider()
    section_header("Dataset", "", "📦")

    kv_grid([
        {"key": "Total Patches", "value": "5,001", "icon": "📦"},
        {"key": "Spatial Resolution", "value": "30m", "icon": "📐"},
        {"key": "Patch Size", "value": "1,920m × 1,920m", "icon": "🗺️"},
        {"key": "ROI", "value": "Rondonia, Brazil", "icon": "🌍"},
        {"key": "Time Range", "value": "2023 (12 monthly)", "icon": "📅"},
        {"key": "Bands", "value": "6 raw + 3 derived = 9 features", "icon": "📡"},
        {"key": "Train / Val / Test", "value": "4,000 / 499 / 502", "icon": "🔀"},
        {"key": "Positive Rate", "value": "77.3% deforestation", "icon": "🔴"},
        {"key": "Labels", "value": "Hansen GFC v2025", "icon": "🏷️"},
    ], columns=3)

    footer()
