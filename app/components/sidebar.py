"""Custom sidebar — navigation, system status, threshold."""
from __future__ import annotations

import sys
from pathlib import Path

import psutil
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


NAV_ITEMS = [
    ("Dashboard",         "dashboard",         "🏠"),
    ("Map Explorer",      "map_explorer",      "🗺️"),
    ("Single Prediction", "single_prediction",  "🔍"),
    ("Batch Prediction",  "batch_prediction",   "📂"),
    ("Analytics",         "analytics",          "📊"),
    ("Model Details",     "model_details",      "🧠"),
    ("About",             "about",              "ℹ️"),
]


def render_sidebar() -> str:
    """Render the premium sidebar and return the selected page key."""
    with st.sidebar:
        # ── Brand ──
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand__logo">
                🌿 <span style="background:var(--grad-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">DEW</span>
            </div>
            <div class="sidebar-brand__sub">Deforestation Early Warning</div>
            <div class="sidebar-brand__line"></div>
        </div>""", unsafe_allow_html=True)

        # ── Navigation ──
        labels = [f"{icon}  {name}" for name, _, icon in NAV_ITEMS]
        selected_label = st.radio(
            "Navigation",
            labels,
            label_visibility="collapsed",
            key="nav_radio",
        )

        idx = labels.index(selected_label) if selected_label in labels else 0
        page_key = NAV_ITEMS[idx][1]

        st.markdown('<div class="divider" style="margin:16px 0;"></div>', unsafe_allow_html=True)

        # ── System Status ──
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        device = "GPU" if torch.cuda.is_available() else "CPU"

        cpu_color = "#00C896" if cpu < 60 else "#F59E0B" if cpu < 85 else "#EF4444"
        mem_color = "#00C896" if mem.percent < 60 else "#F59E0B" if mem.percent < 85 else "#EF4444"
        device_color = "#00C896"

        st.markdown(f"""
        <div class="sidebar-status">
            <div class="sidebar-status__title">System Status</div>
            <div class="sidebar-status__row">
                <div class="sidebar-status__dot" style="background:{cpu_color};box-shadow:0 0 6px {cpu_color};"></div>
                <span class="sidebar-status__label">CPU</span>
                <span class="sidebar-status__val">{cpu:.0f}%</span>
            </div>
            <div class="sidebar-status__row">
                <div class="sidebar-status__dot" style="background:{mem_color};box-shadow:0 0 6px {mem_color};"></div>
                <span class="sidebar-status__label">RAM</span>
                <span class="sidebar-status__val">{mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB</span>
            </div>
            <div class="sidebar-status__row">
                <div class="sidebar-status__dot" style="background:{device_color};box-shadow:0 0 6px {device_color};"></div>
                <span class="sidebar-status__label">Device</span>
                <span class="sidebar-status__val" style="color:{device_color};">{device}</span>
            </div>
            <div class="sidebar-status__row">
                <div class="sidebar-status__dot" style="background:#00C896;box-shadow:0 0 6px #00C896;"></div>
                <span class="sidebar-status__label">Model</span>
                <span class="sidebar-status__val" style="color:#00C896;">Ready</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider" style="margin:8px 0 16px 0;"></div>', unsafe_allow_html=True)

        # ── Threshold ──
        threshold = st.slider(
            "Threshold",
            0.0, 1.0, 0.5, 0.05,
            key="sidebar_threshold",
            help="Decision threshold for deforestation prediction",
            label_visibility="visible",
        )

        st.markdown(f"""
        <div class="threshold-display">
            <div class="threshold-display__label">Current Threshold</div>
            <div class="threshold-display__value">{threshold:.2f}</div>
        </div>""", unsafe_allow_html=True)

        # ── Footer ──
        st.markdown("""
        <div class="sidebar-footer">
            <div class="sidebar-footer__text">v2.0 · ResNet18 · Sentinel-2</div>
        </div>""", unsafe_allow_html=True)

    return page_key
