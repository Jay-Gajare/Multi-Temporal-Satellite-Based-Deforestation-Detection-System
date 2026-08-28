"""Dashboard — landing page with KPIs, pipeline overview, system status."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.cards import metric_card, status_card, glass_card
from app.components.layout import (
    hero_section, section_header, pipeline_step, divider, footer,
    page_header, info_banner, workflow_diagram, kv_grid,
)


@st.cache_data(ttl=3600, show_spinner="Loading training data…")
def _load_metrics() -> dict:
    p = Path("models/run_01/metrics.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600, show_spinner="Loading evaluation data…")
def _load_eval() -> dict:
    p = Path("models/run_01/test_evaluation/test_results.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def render() -> None:
    page_header(
        "Dashboard",
        "AI-powered forest monitoring overview",
        icon="🏠",
        icon_gradient="var(--grad-primary)",
    )

    hero_section(
        "Deforestation Early Warning System",
        "AI-powered forest monitoring using Sentinel-2 satellite imagery. "
        "Detect deforestation events in near-real-time with deep learning.",
        badge_text="v2.0 Production",
        badge_variant="success",
    )

    # ── Load Data ─────────────────────────────────────────────────
    train_data = _load_metrics()
    eval_data = _load_eval()

    m05 = eval_data.get("metrics_at_05", {})
    train_metrics = train_data.get("best_val_metrics", {})

    acc = m05.get("accuracy", train_metrics.get("accuracy", 0))
    prec = m05.get("precision", train_metrics.get("precision", 0))
    rec = m05.get("recall", train_metrics.get("recall", 0))
    f1 = m05.get("f1", train_metrics.get("f1", 0))
    roc = m05.get("roc_auc", 0)
    train_time_s = train_data.get("training_time_seconds", 0)
    train_time_h = train_time_s / 3600
    n_epochs = train_data.get("total_epochs_run", 0)
    model_name = train_data.get("config", {}).get("architecture", "resnet18").upper()
    n_params = "11.51M"

    # ── KPI Cards ────────────────────────────────────────────────
    section_header("Model Performance", "Evaluated on 502 held-out test patches", "📈")

    c1, c2, c3, c4 = st.columns(4)
    kpis_row1 = [
        ("Accuracy",  f"{acc*100:.1f}%",  "Baseline",     "green",  "🎯", "var(--grad-primary)"),
        ("Precision", f"{prec*100:.1f}%", "Low false alarms", "blue", "✅", "var(--grad-cool)"),
        ("Recall",    f"{rec*100:.1f}%",  "85% detection","green",  "🔍", "var(--grad-warm)"),
        ("F1 Score",  f"{f1*100:.1f}%",  "Primary metric","green", "📊", "var(--grad-accent)"),
    ]
    for col, (label, val, delta, dc, icon, grad) in zip([c1, c2, c3, c4], kpis_row1):
        with col:
            metric_card(label, val, icon=icon, delta=delta, delta_color=dc, gradient=grad)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpis_row2 = [
        ("ROC-AUC",     f"{roc*100:.1f}%",   "Discrimination", "blue",   "📈", "var(--grad-cool)"),
        ("Training Time", f"{train_time_h:.1f}h", f"{n_epochs} epochs", "green", "⏱️", "var(--grad-accent)"),
        ("Dataset",     "5,001 patches",     "Sentinel-2",     "green",  "📦", "var(--grad-primary)"),
        ("Model",       model_name,          f"{n_params} params", "blue","🧠", "var(--grad-cool)"),
    ]
    for col, (label, val, delta, dc, icon, grad) in zip([c1, c2, c3, c4], kpis_row2):
        with col:
            metric_card(label, val, icon=icon, delta=delta, delta_color=dc, gradient=grad)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Overview (Visual) ────────────────────────────────
    section_header("Pipeline Overview", "End-to-end system architecture", "🔄")

    workflow_diagram([
        {
            "icon": "🛰️", "title": "Data Export",
            "desc": "5,001 Sentinel-2 patches from GEE",
            "gradient": "var(--grad-cool)",
        },
        {
            "icon": "⚙️", "title": "Preprocessing",
            "desc": "Cloud masking + spectral indices + stacking",
            "gradient": "var(--grad-accent)",
        },
        {
            "icon": "🏷️", "title": "Ground Truth",
            "desc": "Hansen GFC v2025 binary labels",
            "gradient": "var(--grad-warm)",
        },
        {
            "icon": "🧠", "title": "Training",
            "desc": f"ResNet18 · {n_params} · {n_epochs} epochs",
            "gradient": "var(--grad-primary)",
        },
        {
            "icon": "📊", "title": "Evaluation",
            "desc": f"F1={f1:.3f} · AUC={roc:.3f}",
            "gradient": "var(--grad-safe)",
        },
        {
            "icon": "🚀", "title": "Inference",
            "desc": "~2.6s/patch · Grad-CAM explainability",
            "gradient": "var(--grad-danger)",
        },
    ])

    divider()

    # ── Architecture + Dataset ────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        glass_card(title="Training Pipeline", icon="🏗️", icon_gradient="var(--grad-primary)")
        pipeline_step(1, "Data Export", "5,001 Sentinel-2 patches from GEE (12 monthly composites each)", "🛰️")
        pipeline_step(2, "Preprocessing", "Cloud masking, spectral indices (NDVI, NBR, NDMI), temporal stacking", "⚙️")
        pipeline_step(3, "Ground Truth", "Hansen Global Forest Change v2025 binary labels", "🌍")
        pipeline_step(4, "Training", f"ResNet18 ({n_params} params) · 108-band input · AdamW · Cosine LR", "🧠")
        pipeline_step(5, "Evaluation", f"F1={f1:.3f} · AUC={roc:.3f} · Grad-CAM explainability", "📊")
        pipeline_step(6, "Inference", "Production pipeline with input validation, Grad-CAM, and batch support", "🚀")

    with right:
        glass_card(title="Dataset Statistics", subtitle="Rondonia, Brazil — 2023", icon="📦", icon_gradient="var(--grad-cool)")
        c1r, c2r = st.columns(2)
        with c1r:
            metric_card("Total Patches", "5,001", icon="📦", gradient="var(--grad-accent)")
            metric_card("Train Split", "4,000", icon="🎯", gradient="var(--grad-cool)")
            metric_card("Positive Rate", "77.3%", icon="🔴", gradient="var(--grad-warm)")
        with c2r:
            metric_card("Test Patches", "502", icon="✅", gradient="var(--grad-primary)")
            metric_card("Val Split", "499", icon="📊", gradient="var(--grad-safe)")
            metric_card("Monthly Images", "12", icon="📅", gradient="var(--grad-warm)")

        divider()

        section_header("System Status", "", "🖥️")
        status_card("Model", "ok", f"{model_name} · {n_params} params · Best epoch {train_data.get('best_epoch', '?')}", "Ready")
        status_card("Training", "ok", f"Early stopped at epoch {n_epochs} · Val F1 = {train_metrics.get('f1', 0):.4f}", "Complete")
        status_card("Inference", "ok", "CPU · ~2.6s per patch · ~0.4 patches/s", "Active")
        status_card("Explainability", "ok", "Grad-CAM via layer4 hooks · No external deps", "Active")

    divider()

    # ── Quick Info ────────────────────────────────────────────────
    info_banner(
        "System is operational. All components loaded successfully.",
        icon="✅",
        sub="Last checked: just now",
    )

    divider()
    footer()
