"""About — project overview, workflow diagram, technology stack, roadmap, credits."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.cards import glass_card
from app.components.layout import (
    page_header, section_header, divider, footer,
    workflow_diagram, tech_stack_grid, timeline, kv_grid, info_banner,
)


def render() -> None:
    page_header(
        "About",
        "Deforestation Early Warning System",
        icon="ℹ️",
        icon_gradient="var(--grad-primary)",
    )

    # ── Project Overview ──────────────────────────────────────────
    glass_card(title="Deforestation Early Warning System", icon="", accent=True)
    st.markdown("""
    <div style="font-size:14px;color:var(--text-secondary);line-height:1.8;max-width:720px;padding-top:4px;">
        An AI-powered system for detecting deforestation events in near-real-time using
        Sentinel-2 satellite imagery. The pipeline processes multi-temporal spectral data,
        computes vegetation indices (NDVI, NBR, NDMI), and applies a deep learning model
        (ResNet18) to classify 1.9km&sup2; forest patches as deforested or intact.
        <br><br>
        The system covers <strong style="color:var(--accent);">Rondonia, Brazil</strong> — one of the most affected regions by
        tropical deforestation — and processes 12 monthly Sentinel-2 composites per patch
        for temporal analysis.
    </div>""", unsafe_allow_html=True)

    # ── Key Stats ─────────────────────────────────────────────────
    kv_grid([
        {"key": "Region", "value": "Rondonia, Brazil", "icon": "🌍"},
        {"key": "Resolution", "value": "30m (1.9km\u00b2 patches)", "icon": "📐"},
        {"key": "Dataset", "value": "5,001 labeled patches", "icon": "📦"},
        {"key": "Model", "value": "ResNet18 (11.51M params)", "icon": "🧠"},
        {"key": "F1 Score", "value": "0.847 @ threshold 0.50", "icon": "📊"},
        {"key": "Training", "value": "37 epochs, ~2.4h on CPU", "icon": "⏱️"},
    ], columns=3)

    divider()

    # ── Workflow Diagram ──────────────────────────────────────────
    section_header("System Workflow", "End-to-end deforestation detection pipeline", "🔄")

    workflow_diagram([
        {
            "icon": "🛰️", "title": "Data Acquisition",
            "desc": "Sentinel-2 SR imagery exported from Google Earth Engine with cloud masking",
            "gradient": "var(--grad-cool)",
        },
        {
            "icon": "⚙️", "title": "Preprocessing",
            "desc": "Spectral indices (NDVI, NBR, NDMI) + 12-month temporal stacking → 108 channels",
            "gradient": "var(--grad-accent)",
        },
        {
            "icon": "🏷️", "title": "Labeling",
            "desc": "Hansen Global Forest Change v2025 binary ground truth labels applied",
            "gradient": "var(--grad-warm)",
        },
        {
            "icon": "🧠", "title": "Training",
            "desc": "ResNet18 + AdamW + Cosine LR + Early Stopping on 4,000 training patches",
            "gradient": "var(--grad-primary)",
        },
        {
            "icon": "📊", "title": "Evaluation",
            "desc": "F1=0.847, AUC=0.773, Grad-CAM explainability on 502 held-out test patches",
            "gradient": "var(--grad-safe)",
        },
        {
            "icon": "🚀", "title": "Deployment",
            "desc": "Interactive Streamlit dashboard with map, batch inference, and analytics",
            "gradient": "var(--grad-danger)",
        },
    ])

    divider()

    # ── Technology Stack ──────────────────────────────────────────
    section_header("Technology Stack", "Tools and frameworks powering the system", "🛠️")

    tech_stack_grid([
        {"icon": "🌍", "name": "Google Earth Engine", "desc": "Sentinel-2 data export and cloud masking"},
        {"icon": "📡", "name": "Sentinel-2 SR", "desc": "6-band multispectral satellite imagery"},
        {"icon": "🗺️", "name": "Hansen GFC v2025", "desc": "Global forest change ground truth labels"},
        {"icon": "🐍", "name": "Python 3.11", "desc": "Core language for all pipeline components"},
        {"icon": "🔥", "name": "PyTorch", "desc": "Deep learning framework for ResNet18 model"},
        {"icon": "🧠", "name": "ResNet18", "desc": "CNN backbone with 11.51M parameters"},
        {"icon": "📊", "name": "Plotly", "desc": "Interactive charts and visualization"},
        {"icon": "🗺️", "name": "Folium", "desc": "Interactive GIS map with leaflet.js"},
        {"icon": "🖥️", "name": "Streamlit", "desc": "Web application framework for the dashboard"},
        {"icon": "📐", "name": "NumPy", "desc": "Numerical computing and array operations"},
        {"icon": "🐼", "name": "Pandas", "desc": "Data manipulation and CSV processing"},
        {"icon": "🖼️", "name": "Pillow", "desc": "Image processing for Grad-CAM overlays"},
    ])

    divider()

    # ── Data Pipeline Detail ──────────────────────────────────────
    section_header("Data Pipeline", "From raw satellite data to prediction", "📦")
    glass_card(title="Input → Output", icon="🔄", icon_gradient="var(--grad-accent)")

    kv_grid([
        {"key": "Input Bands", "value": "B2, B3, B4, B8, B11, B12", "icon": "📡"},
        {"key": "Derived Indices", "value": "NDVI, NBR, NDMI", "icon": "📊"},
        {"key": "Temporal Stack", "value": "12 months × 9 features", "icon": "📅"},
        {"key": "Input Shape", "value": "(B, 108, 64, 64)", "icon": "📐"},
        {"key": "Output", "value": "Binary (deforestation / intact)", "icon": "🎯"},
        {"key": "Latency", "value": "~2.6s per patch (CPU)", "icon": "⏱️"},
    ], columns=3)

    divider()

    # ── Model Architecture ────────────────────────────────────────
    section_header("Model Architecture", "ResNet18 adapted for multi-spectral remote sensing", "🧠")
    glass_card(title="Architecture Details", icon="🏗️", icon_gradient="var(--grad-cool)")

    kv_grid([
        {"key": "Backbone", "value": "ResNet18 (pretrained=False)", "icon": "🏗️"},
        {"key": "Parameters", "value": "11.51M", "icon": "🔢"},
        {"key": "Input Channels", "value": "108 (replaces 3 RGB)", "icon": "📥"},
        {"key": "Spatial Size", "value": "64×64 pixels", "icon": "📐"},
        {"key": "Head", "value": "Dropout(0.3) → Linear(1)", "icon": "🎯"},
        {"key": "Activation", "value": "Sigmoid (probability)", "icon": "📊"},
        {"key": "Loss", "value": "BCEWithLogitsLoss", "icon": "📉"},
        {"key": "Optimizer", "value": "AdamW (lr=1e-4, wd=1e-4)", "icon": "⚙️"},
        {"key": "Scheduler", "value": "CosineAnnealing + 3-epoch warmup", "icon": "📅"},
    ], columns=3)

    divider()

    # ── Future Work / Roadmap ─────────────────────────────────────
    section_header("Future Work", "Planned improvements and research directions", "🔮")

    timeline([
        {
            "phase": "Phase 1 — Short Term",
            "title": "Multi-temporal Architecture",
            "desc": "LSTM/Transformer models to capture temporal deforestation patterns across monthly composites instead of treating months independently.",
            "tags": ["PyTorch", "LSTM", "Transformer"],
        },
        {
            "phase": "Phase 2 — Medium Term",
            "title": "GPU Acceleration & API",
            "desc": "CUDA inference for real-time batch processing. FastAPI backend for programmatic access and integration with external monitoring systems.",
            "tags": ["CUDA", "FastAPI", "REST API"],
        },
        {
            "phase": "Phase 3 — Medium Term",
            "title": "Real-time Alert System",
            "desc": "Automated monitoring pipeline with GEE integration. Email and webhook notifications when deforestation is detected in tracked regions.",
            "tags": ["GEE", "Webhooks", "Email Alerts"],
        },
        {
            "phase": "Phase 4 — Long Term",
            "title": "Higher Resolution Analysis",
            "desc": "Sub-30m analysis with Super-Resolution models. Integration with Planet Labs and commercial satellite providers for daily revisits.",
            "tags": ["Super-Resolution", "Planet Labs", "30m+"],
        },
        {
            "phase": "Phase 5 — Long Term",
            "title": "Multi-Region Expansion",
            "desc": "Expand to other deforestation hotspots: Amazon, Congo Basin, Southeast Asia. Transfer learning across regions with domain adaptation.",
            "tags": ["Transfer Learning", "Global", "Domain Adaptation"],
        },
    ])

    divider()

    # ── Credits ───────────────────────────────────────────────────
    section_header("Credits & Acknowledgments", "", "👥")
    glass_card(title="Acknowledgments", icon="🤝", icon_gradient="var(--grad-primary)")
    st.markdown("""
    <div style="font-size:13px;color:var(--text-secondary);line-height:2.2;padding-top:4px;">
        <strong style="color:var(--text-primary);">Data:</strong> Google Earth Engine, Copernicus Sentinel-2, Hansen/UMD Global Forest Change<br>
        <strong style="color:var(--text-primary);">Framework:</strong> PyTorch, Streamlit, Plotly, scikit-learn<br>
        <strong style="color:var(--text-primary);">Architecture:</strong> ResNet (He et al., 2015) with custom multi-spectral modifications<br>
        <strong style="color:var(--text-primary);">Explainability:</strong> Grad-CAM (Selvaraju et al., 2017) implemented from scratch<br>
        <strong style="color:var(--text-primary);">GIS:</strong> Folium + Leaflet.js for interactive map visualization
    </div>""", unsafe_allow_html=True)

    divider()

    # ── Quick Reference ───────────────────────────────────────────
    section_header("Quick Reference", "Key numbers at a glance", "📋")
    kv_grid([
        {"key": "Total Patches", "value": "23,547", "icon": "📦"},
        {"key": "Labeled Patches", "value": "5,001", "icon": "🏷️"},
        {"key": "Train / Val / Test", "value": "4,000 / 499 / 502", "icon": "📊"},
        {"key": "Best Epoch", "value": "27 / 37", "icon": "⭐"},
        {"key": "Training Time", "value": "8,757s (~2.4h)", "icon": "⏱️"},
        {"key": "Best Threshold", "value": "0.10 (F1=0.860)", "icon": "📏"},
        {"key": "F1 @ 0.50", "value": "0.847", "icon": "🎯"},
        {"key": "ROC-AUC", "value": "0.773", "icon": "📈"},
        {"key": "PR-AUC", "value": "0.912", "icon": "📊"},
    ], columns=3)

    divider()
    footer()
