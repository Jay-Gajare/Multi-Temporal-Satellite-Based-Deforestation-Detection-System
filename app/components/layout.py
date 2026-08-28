"""Layout helpers — header, footer, section helpers, pipeline, workflow, tech stack."""
from __future__ import annotations

from typing import Optional

import streamlit as st


def page_header(
    title: str,
    subtitle: str = "",
    icon: str = "",
    icon_gradient: str = "var(--grad-primary)",
    actions_html: str = "",
) -> None:
    """Render a premium page header bar."""
    icon_html = (
        f'<div class="page-header__icon" style="background:{icon_gradient};">{icon}</div>'
        if icon else ""
    )
    sub_html = f'<div class="page-header__subtitle">{subtitle}</div>' if subtitle else ""
    actions = f'<div class="page-header__right">{actions_html}</div>' if actions_html else ""
    html = f"""
    <div class="page-header anim-fade-in">
        <div class="page-header__left">
            {icon_html}
            <div>
                <div class="page-header__title">{title}</div>
                {sub_html}
            </div>
        </div>
        {actions}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def hero_section(title: str, subtitle: str, badge_text: str = "", badge_variant: str = "success") -> None:
    """Render the hero banner section."""
    badge_cls = f"badge-{badge_variant}"
    badge_html = (
        f'<div class="hero-badge {badge_cls}">{badge_text}</div>' if badge_text else ""
    )
    st.markdown(f"""
    <div class="hero-section anim-fade-in">
        {badge_html}
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>""", unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a section header with optional subtitle."""
    icon_html = f'<span style="margin-right:8px;">{icon}</span>' if icon else ''
    subtitle_html = f'<div class="section-subheader">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
    <div class="anim-fade-in">
        <div class="section-header">{icon_html}{title}</div>
        {subtitle_html}
    </div>""", unsafe_allow_html=True)


def divider() -> None:
    """Render a horizontal divider."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def badge(text: str, variant: str = "success") -> str:
    """Return HTML string for a badge."""
    return f'<span class="badge badge-{variant}">{text}</span>'


def pipeline_step(number: int, title: str, description: str, icon: str = "") -> None:
    """Render a numbered pipeline step."""
    icon_html = f'<span style="margin-right:6px;">{icon}</span>' if icon else ''
    st.markdown(f"""
    <div class="pipeline-step">
        <div class="pipeline-step__num">{number}</div>
        <div>
            <div class="pipeline-step__title">{icon_html}{title}</div>
            <div class="pipeline-step__desc">{description}</div>
        </div>
    </div>""", unsafe_allow_html=True)


def footer() -> None:
    """Render the page footer."""
    st.markdown("""
    <div class="app-footer">
        <strong>Deforestation Early Warning System</strong><br>
        <span style="margin-top:4px;display:inline-block;">
            Powered by Sentinel-2 &middot; ResNet18 &middot; PyTorch &middot; Streamlit
        </span>
    </div>""", unsafe_allow_html=True)


def three_col_metrics(m1: dict, m2: dict, m3: dict) -> None:
    """Render three metric cards in a 3-column layout."""
    from app.components.cards import metric_card
    c1, c2, c3 = st.columns(3)
    for col, m in zip([c1, c2, c3], [m1, m2, m3]):
        with col:
            metric_card(**m)


def info_banner(text: str, icon: str = "ℹ️", sub: str = "") -> None:
    """Render an inline info banner."""
    sub_html = f'<span class="info-banner__sub">{sub}</span>' if sub else ""
    st.markdown(f"""
    <div class="info-banner anim-fade">
        <span class="info-banner__icon">{icon}</span>
        <span class="info-banner__text">{text}</span>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def empty_state(icon: str, title: str, description: str) -> None:
    """Render an empty state placeholder."""
    st.markdown(f"""
    <div class="empty-state anim-fade">
        <div class="empty-state__icon">{icon}</div>
        <div class="empty-state__title">{title}</div>
        <div class="empty-state__desc">{description}</div>
    </div>""", unsafe_allow_html=True)


# ── Workflow Diagram ──────────────────────────────────────────────
def workflow_diagram(steps: list[dict]) -> None:
    """Render a horizontal workflow diagram with connected nodes.

    Parameters
    ----------
    steps : List of dicts with keys: icon, title, desc, gradient (optional).
    """
    if not steps:
        return

    nodes_html = ""
    for i, step in enumerate(steps):
        grad = step.get("gradient", "var(--grad-primary)")
        nodes_html += f'''
        <div class="workflow-node">
            <div class="workflow-node__icon" style="background:{grad};">{step["icon"]}</div>
            <div class="workflow-node__step">Step {i + 1}</div>
            <div class="workflow-node__title">{step["title"]}</div>
            <div class="workflow-node__desc">{step["desc"]}</div>
        </div>'''
        if i < len(steps) - 1:
            nodes_html += '<div class="workflow-arrow"><div class="workflow-arrow__line"></div></div>'

    st.markdown(f'''
    <div class="glass-card glass-card--no-hover" style="padding:24px 20px;overflow-x:auto;">
        <div class="workflow-container">{nodes_html}</div>
    </div>''', unsafe_allow_html=True)


# ── Tech Stack Grid ───────────────────────────────────────────────
def tech_stack_grid(items: list[dict]) -> None:
    """Render a responsive grid of technology cards.

    Parameters
    ----------
    items : List of dicts with keys: icon, name, desc.
    """
    if not items:
        return

    cards_html = ""
    for item in items:
        cards_html += f'''
        <div class="tech-card">
            <div class="tech-card__icon">{item["icon"]}</div>
            <div class="tech-card__name">{item["name"]}</div>
            <div class="tech-card__desc">{item["desc"]}</div>
        </div>'''

    st.markdown(f'<div class="tech-grid">{cards_html}</div>', unsafe_allow_html=True)


# ── Timeline / Roadmap ────────────────────────────────────────────
def timeline(items: list[dict]) -> None:
    """Render a vertical timeline / roadmap.

    Parameters
    ----------
    items : List of dicts with keys: phase, title, desc, tags (optional list of str).
    """
    if not items:
        return

    items_html = ""
    for item in items:
        tags = item.get("tags", [])
        tags_html = ""
        if tags:
            tags_html = '<div class="timeline-item__tags">'
            for tag in tags:
                tags_html += f'<span class="badge badge-info">{tag}</span>'
            tags_html += '</div>'

        items_html += f'''
        <div class="timeline-item">
            <div class="timeline-item__phase">{item["phase"]}</div>
            <div class="timeline-item__title">{item["title"]}</div>
            <div class="timeline-item__desc">{item["desc"]}</div>
            {tags_html}
        </div>'''

    st.markdown(f'<div class="timeline">{items_html}</div>', unsafe_allow_html=True)


# ── Key-Value Grid ────────────────────────────────────────────────
def kv_grid(items: list[dict], columns: int = 3) -> None:
    """Render a grid of key-value pairs (e.g., stats, specs).

    Parameters
    ----------
    items : List of dicts with keys: key, value, icon (optional).
    columns : Number of columns.
    """
    if not items:
        return

    cols = st.columns(columns)
    for i, item in enumerate(items):
        col = cols[i % columns]
        with col:
            icon_html = f'<span style="margin-right:6px;">{item["icon"]}</span>' if item.get("icon") else ""
            st.markdown(f'''
            <div style="padding:12px 0;">
                <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;
                    color:var(--text-muted);margin-bottom:4px;">{icon_html}{item["key"]}</div>
                <div style="font-size:15px;font-weight:700;color:var(--text-primary);
                    font-family:var(--font-mono);">{item["value"]}</div>
            </div>''', unsafe_allow_html=True)
