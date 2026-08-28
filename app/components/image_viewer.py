"""Professional image viewer with tabs, zoom, fullscreen, and download."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import streamlit as st
from PIL import Image

_VIEWER_COUNTER = "image_viewer_calls"


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _panel_crop(img: Image.Image, index: int, total: int = 3) -> Image.Image:
    w, h = img.size
    pw = w // total
    x0 = index * pw
    x1 = (index + 1) * pw if index < total - 1 else w
    return img.crop((x0, 0, x1, h))


def render_image_viewer(
    image_path: str | Path,
    patch_id: str = "",
    panel_labels: list[str] | None = None,
    panel_icons: list[str] | None = None,
    key_prefix: str = "viewer",
) -> None:
    """Render a tabbed image viewer with zoom, fullscreen, and download.

    Parameters
    ----------
    image_path : Path to a multi-panel PNG (3 panels: RGB | Grad-CAM | Overlay).
    patch_id : Patch identifier for display.
    panel_labels : Labels for each tab.
    panel_icons : Emoji icons for each tab.
    key_prefix : Unique prefix for Streamlit widget keys.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        st.warning(f"Image not found: {image_path.name}")
        return

    img = Image.open(image_path)
    n_panels = 3
    labels = panel_labels or ["RGB Composite", "Grad-CAM Heatmap", "Overlay"]
    icons = panel_icons or ["🛰️", "🔥", "🎯"]

    panels = [_panel_crop(img, i, n_panels) for i in range(n_panels)]
    zoom_key = f"{key_prefix}_zoom"
    if zoom_key not in st.session_state:
        st.session_state[zoom_key] = 100

    tabs = st.tabs([f"{icons[i]} {labels[i]}" for i in range(n_panels)])

    for i, tab in enumerate(tabs):
        with tab:
            c_viewer, c_controls = st.columns([5, 1])

            with c_controls:
                st.markdown(
                    '<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:1px;color:var(--text-muted);margin-bottom:10px;">Controls</div>',
                    unsafe_allow_html=True,
                )

                zoom_val = st.slider(
                    "Zoom",
                    50, 300, st.session_state[zoom_key],
                    step=10, key=f"{key_prefix}_zoom_{i}",
                    label_visibility="collapsed",
                )
                st.session_state[zoom_key] = zoom_val

                zoom_label = f"{zoom_val}%"
                st.markdown(
                    f'<div style="text-align:center;font-size:12px;font-weight:700;'
                    f'font-family:var(--font-mono);color:var(--accent);margin-bottom:8px;">'
                    f'🔍 {zoom_label}</div>',
                    unsafe_allow_html=True,
                )

                fs_id = f"{key_prefix}_fs_{i}"
                st.markdown(
                    f'<button id="{fs_id}" onclick="(function(){{'
                    f"var c=this.closest('.stTab');if(c){{var img=c.querySelector('img');"
                    f"if(img){{if(img.requestFullscreen)img.requestFullscreen();"
                    f"else if(img.webkitRequestFullscreen)img.webkitRequestFullscreen();}}}}"
                    f'}})()" '
                    f'style="width:100%;padding:8px 0;background:var(--bg-tertiary);'
                    f'border:1px solid var(--border);border-radius:var(--radius-sm);'
                    f'color:var(--text-primary);font-size:12px;font-weight:600;cursor:pointer;'
                    f'font-family:var(--font-sans);transition:all 0.15s;margin-bottom:8px;">'
                    f'⛶ Fullscreen</button>',
                    unsafe_allow_html=True,
                )

                panel_b64 = _b64_png(panels[i])
                fname = f"{patch_id}_{labels[i].lower().replace(' ', '_')}.png"
                st.download_button(
                    label=f"📥 Download",
                    data=base64.b64decode(panel_b64),
                    file_name=fname,
                    mime="image/png",
                    key=f"{key_prefix}_dl_{i}",
                    use_container_width=True,
                )

            with c_viewer:
                scale = zoom_val / 100.0
                panel_b64 = _b64_png(panels[i])
                st.markdown(
                    f'<div class="image-viewer" style="overflow:auto;border-radius:var(--radius-md);'
                    f'border:1px solid var(--border);background:#000;">'
                    f'<img src="data:image/png;base64,{panel_b64}" '
                    f'style="width:{scale*100:.0f}%;height:auto;display:block;'
                    f'transition:width 0.3s ease;" />'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 0;margin-top:4px;">'
        f'<span style="font-size:11px;color:var(--text-muted);">Source: {image_path.name}</span>'
        f'<span style="font-size:11px;color:var(--text-muted);">{img.size[0]}×{img.size[1]}px</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
