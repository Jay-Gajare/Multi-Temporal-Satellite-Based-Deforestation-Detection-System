"""Reusable card components — GlassCard, MetricCard, StatusBadge, PredictionCard."""
from __future__ import annotations

import streamlit as st
from typing import Optional


# ── Glass Card ───────────────────────────────────────────────────
def glass_card(
    title: str = "",
    subtitle: str = "",
    icon: str = "",
    icon_gradient: str = "var(--grad-primary)",
    accent: bool = False,
    content: str = "",
    key: str = "",
) -> None:
    """Render a glassmorphism card with optional header and content."""
    accent_cls = " glass-card--accent" if accent else ""
    header_html = ""

    if title:
        icon_html = (
            f'<div class="glass-card__header-icon" style="background:{icon_gradient};">{icon}</div>'
            if icon else ""
        )
        sub_html = f'<div class="glass-card__header-sub">{subtitle}</div>' if subtitle else ""
        header_html = f"""
        <div class="glass-card__header">
            {icon_html}
            <div>
                <div class="glass-card__header-title">{title}</div>
                {sub_html}
            </div>
        </div>"""

    body = content if content else ""
    html = f'<div class="glass-card glass-card--no-hover{accent_cls}">{header_html}{body}</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Metric Card ──────────────────────────────────────────────────
def metric_card(
    label: str,
    value: str,
    icon: str = "",
    delta: Optional[str] = None,
    delta_color: str = "green",
    gradient: str = "var(--grad-primary)",
    help_text: str = "",
) -> None:
    """Render a premium KPI metric card."""
    color_map = {
        "green": "#00C896",
        "red": "#EF4444",
        "yellow": "#F59E0B",
        "blue": "#3B82F6",
    }
    delta_bg = color_map.get(delta_color, "#00C896")
    delta_html = ""
    if delta:
        delta_html = (
            f'<div class="mc-delta" style="background:{delta_bg}12;color:{delta_bg};">{delta}</div>'
        )

    help_html = ""
    if help_text:
        help_html = f'<div style="font-size:10px;color:var(--text-muted);margin-top:6px;">{help_text}</div>'

    html = f"""
    <div class="metric-card">
        <div>
            <div class="mc-icon" style="background:{gradient};">{icon}</div>
            <div class="mc-value" style="background:{gradient};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                {value}
            </div>
            <div class="mc-label">{label}</div>
            {delta_html}
        </div>
        {help_html}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ── Status Badge ─────────────────────────────────────────────────
def status_badge(text: str, variant: str = "ok") -> str:
    """Return HTML string for an inline status badge."""
    return (
        f'<span class="status-badge status-badge--{variant}">'
        f'<span class="status-badge__dot"></span>{text}</span>'
    )


def status_card(title: str, status: str, details: str = "", value: str = "") -> None:
    """Render a system status card row."""
    color_map = {"ok": "#00C896", "warning": "#F59E0B", "error": "#EF4444", "info": "#38BDF8"}
    color = color_map.get(status, "#00C896")
    value_html = f'<div class="status-card__value" style="color:{color};">{value}</div>' if value else ""
    html = f"""
    <div class="status-card">
        <div class="status-card__dot" style="background:{color};box-shadow:0 0 8px {color};"></div>
        <div>
            <div class="status-card__text-title">{title}</div>
            <div class="status-card__text-sub">{details}</div>
        </div>
        {value_html}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ── Prediction Card ──────────────────────────────────────────────
def prediction_card(
    prediction: int,
    probability: float,
    confidence: float,
    inference_time: float,
    threshold: float = 0.5,
    model_name: str = "ResNet18",
) -> None:
    """Render a premium prediction result card."""
    is_defor = prediction == 1
    label = "DEFORESTATION DETECTED" if is_defor else "NO DEFORESTATION"
    color = "#EF4444" if is_defor else "#00C896"
    bg_cls = "prediction-card--defor" if is_defor else "prediction-card--safe"
    icon = "🔴" if is_defor else "🟢"

    conf_label = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
    conf_color = (
        "#EF4444" if confidence >= 0.8
        else "#F59E0B" if confidence >= 0.5
        else "#38BDF8"
    )

    html = f"""
    <div class="prediction-card {bg_cls} anim-fade-in">
        <div style="font-size:40px;margin-bottom:14px;">{icon}</div>
        <div class="prediction-card__label">{label}</div>
        <div class="prediction-card__value" style="color:{color};">{probability:.1%}</div>
        <div class="prediction-card__prob">Confidence: {confidence:.1%}</div>
        <div class="prediction-card__meta">
            <div class="prediction-card__meta-item">
                <div class="prediction-card__meta-label">Confidence Level</div>
                <div class="prediction-card__meta-value" style="color:{conf_color};">{conf_label}</div>
            </div>
            <div class="prediction-card__meta-item">
                <div class="prediction-card__meta-label">Latency</div>
                <div class="prediction-card__meta-value" style="color:var(--text-primary);">{inference_time:.0f}ms</div>
            </div>
            <div class="prediction-card__meta-item">
                <div class="prediction-card__meta-label">Model</div>
                <div class="prediction-card__meta-value" style="color:var(--text-primary);">{model_name}</div>
            </div>
                <div class="prediction-card__meta-item">
                <div class="prediction-card__meta-label">Threshold</div>
                <div class="prediction-card__meta-value" style="color:var(--text-primary);">{threshold:.2f}</div>
            </div>
        </div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ── Prediction Result Card (v2 — animated, icon-rich) ────────────
def prediction_result_card(
    prediction: int,
    probability: float,
    confidence: float,
    threshold: float = 0.5,
    model_name: str = "ResNet18",
    latency_ms: float = 0,
) -> None:
    """Render a premium prediction result card with animated confidence bar."""
    is_defor = prediction == 1
    label = "DEFORESTATION DETECTED" if is_defor else "NO DEFORESTATION"
    color = "#EF4444" if is_defor else "#00C896"
    rgb = "239,68,68" if is_defor else "0,200,150"
    icon = "⚠️" if is_defor else "✅"

    conf_label = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
    conf_color = "#EF4444" if confidence >= 0.8 else "#F59E0B" if confidence >= 0.5 else "#38BDF8"
    conf_bg = f"rgba({rgb},0.08)"
    conf_border = f"rgba({rgb},0.18)"

    html = f'''
    <div class="pred-result anim-fade-in" style="background:linear-gradient(135deg,{conf_bg} 0%,rgba({rgb},0.02) 100%);
        border:1px solid {conf_border};border-radius:var(--radius-xl);padding:0;overflow:hidden;">

        <div style="padding:32px 36px 24px;text-align:center;">
            <div style="font-size:13px;text-transform:uppercase;letter-spacing:1.5px;
                color:var(--text-secondary);margin-bottom:8px;font-weight:600;">Prediction Result</div>
            <div style="display:flex;align-items:center;justify-content:center;gap:14px;margin-bottom:6px;">
                <span style="font-size:42px;line-height:1;">{icon}</span>
                <span style="font-size:44px;font-weight:900;letter-spacing:-2px;line-height:1;color:{color};">{label}</span>
            </div>
            <div style="font-size:17px;font-weight:600;color:var(--text-secondary);margin-bottom:24px;">
                Probability {probability:.1%} &nbsp;·&nbsp; Confidence {confidence:.1%}
            </div>

            <div style="max-width:520px;margin:0 auto;">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.6px;">Confidence Level</span>
                    <span style="font-size:11px;font-weight:700;color:{conf_color};font-family:var(--font-mono);">{conf_label}</span>
                </div>
                <div style="width:100%;height:10px;background:rgba(255,255,255,0.04);border-radius:99px;overflow:hidden;border:1px solid rgba(255,255,255,0.04);">
                    <div style="width:{confidence*100:.1f}%;height:100%;border-radius:99px;
                        background:linear-gradient(90deg,{conf_color} 0%,{color} 100%);
                        animation:confBarFill 1.2s cubic-bezier(0.22,1,0.36,1) both;
                        box-shadow:0 0 12px {conf_color}40;"></div>
                </div>
            </div>

            <div style="display:flex;justify-content:center;gap:28px;margin-top:24px;flex-wrap:wrap;">
                <div style="text-align:center;min-width:80px;">
                    <div style="font-size:18px;margin-bottom:4px;">⚡</div>
                    <div style="font-size:16px;font-weight:800;font-family:var(--font-mono);color:var(--text-primary);">{latency_ms:.0f}ms</div>
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-muted);font-weight:600;margin-top:2px;">Latency</div>
                </div>
                <div style="text-align:center;min-width:80px;">
                    <div style="font-size:18px;margin-bottom:4px;">🧠</div>
                    <div style="font-size:16px;font-weight:800;font-family:var(--font-mono);color:var(--text-primary);">{model_name}</div>
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-muted);font-weight:600;margin-top:2px;">Model</div>
                </div>
                <div style="text-align:center;min-width:80px;">
                    <div style="font-size:18px;margin-bottom:4px;">📏</div>
                    <div style="font-size:16px;font-weight:800;font-family:var(--font-mono);color:var(--text-primary);">{threshold:.2f}</div>
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-muted);font-weight:600;margin-top:2px;">Threshold</div>
                </div>
            </div>
        </div>
    </div>'''
    st.markdown(html, unsafe_allow_html=True)


# ── Timing Breakdown ─────────────────────────────────────────────
def timing_breakdown(result: dict) -> None:
    """Render 4 timing metric cards from a prediction result dict."""
    total_ms = result.get("time_total_s", 0) * 1000
    infer_ms = result.get("time_inference_s", 0) * 1000
    pre_ms = result.get("time_preprocess_s", 0) * 1000
    cam_ms = result.get("time_gradcam_s", 0) * 1000

    items = [
        ("🧠", "Inference", f"{infer_ms:.0f}ms", "var(--grad-cool)"),
        ("⚙️", "Preprocess", f"{pre_ms:.0f}ms", "var(--grad-accent)"),
        ("🔥", "Grad-CAM", f"{cam_ms:.0f}ms", "var(--grad-warm)"),
        ("⏱️", "Total", f"{total_ms:.0f}ms", "var(--grad-primary)"),
    ]

    cols = st.columns(4)
    for col, (icon, label, val, grad) in zip(cols, items):
        with col:
            st.markdown(f'''
            <div class="metric-card" style="min-height:110px;padding:18px 16px;text-align:center;">
                <div class="mc-icon" style="background:{grad};width:36px;height:36px;font-size:16px;margin:0 auto 10px;">{icon}</div>
                <div class="mc-value" style="font-size:20px;background:{grad};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:4px;">{val}</div>
                <div class="mc-label" style="font-size:11px;">{label}</div>
            </div>''', unsafe_allow_html=True)


# ── Inference Pipeline Progress ──────────────────────────────────
def inference_pipeline_progress(result: dict) -> None:
    """Render an animated step-by-step pipeline progress indicator."""
    steps = [
        ("📥", "Load", result.get("time_fileload_s", 0)),
        ("🔬", "Preprocess", result.get("time_preprocess_s", 0)),
        ("🧮", "Tensor", result.get("time_tensor_s", 0)),
        ("🧠", "Inference", result.get("time_inference_s", 0)),
        ("🔥", "Grad-CAM", result.get("time_gradcam_s", 0)),
        ("🎨", "Visualize", result.get("time_visualize_s", 0)),
        ("💾", "Save", result.get("time_save_s", 0)),
    ]
    total = sum(s[2] for s in steps) or 1

    items_html = ""
    for i, (icon, label, elapsed) in enumerate(steps):
        is_last = i == len(steps) - 1
        connector = "" if is_last else f'''<div style="flex:1;height:2px;background:var(--border);
            margin:0 4px;position:relative;top:14px;"></div>'''
        delay = i * 0.08
        items_html += f'''
        <div style="display:flex;align-items:center;animation:fadeInUp 0.4s {delay}s both;">
            <div style="text-align:center;min-width:60px;">
                <div style="width:30px;height:30px;border-radius:50%;background:var(--bg-tertiary);
                    border:2px solid var(--border);display:flex;align-items:center;justify-content:center;
                    font-size:13px;margin:0 auto 6px;">{icon}</div>
                <div style="font-size:10px;font-weight:600;color:var(--text-secondary);">{label}</div>
                <div style="font-size:10px;font-weight:700;font-family:var(--font-mono);color:var(--accent);margin-top:2px;">
                    {elapsed*1000:.0f}ms</div>
            </div>
            {connector}
        </div>'''

    st.markdown(f'''
    <div class="glass-card glass-card--no-hover" style="padding:20px 24px;">
        <div style="font-size:12px;font-weight:700;color:var(--text-secondary);margin-bottom:16px;
            text-transform:uppercase;letter-spacing:0.8px;">Pipeline Execution</div>
        <div style="display:flex;align-items:flex-start;justify-content:center;">{items_html}</div>
    </div>''', unsafe_allow_html=True)
