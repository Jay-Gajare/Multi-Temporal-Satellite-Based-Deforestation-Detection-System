"""
Deforestation Early Warning System — Dashboard

Premium Streamlit web application for AI-powered forest monitoring.
Run with: streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.styles import CUSTOM_CSS
from app.utils.session import init_session
from app.components.sidebar import render_sidebar
from app.components.layout import footer

from app.pages.dashboard import render as render_dashboard
from app.pages.map_explorer import render as render_map_explorer
from app.pages.single_prediction import render as render_single
from app.pages.batch_prediction import render as render_batch
from app.pages.analytics import render as render_analytics
from app.pages.model_details import render as render_model
from app.pages.about import render as render_about

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Deforestation Early Warning",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Deforestation Early Warning System — "
            "AI-powered forest monitoring using Sentinel-2 time series."
        ),
        "Report a bug": "https://github.com/anomalyco/opencode/issues",
    },
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ── Session init ─────────────────────────────────────────────────
init_session()

# ── Sidebar & routing ────────────────────────────────────────────
page = render_sidebar()

# ── Page rendering ───────────────────────────────────────────────
PAGE_MAP = {
    "dashboard":         render_dashboard,
    "map_explorer":      render_map_explorer,
    "single_prediction": render_single,
    "batch_prediction":  render_batch,
    "analytics":         render_analytics,
    "model_details":     render_model,
    "about":             render_about,
}

renderer = PAGE_MAP.get(page, render_dashboard)
renderer()
