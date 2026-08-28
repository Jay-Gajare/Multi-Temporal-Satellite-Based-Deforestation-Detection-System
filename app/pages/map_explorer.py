"""Map Explorer — interactive GIS + on-demand inference for deforestation patches."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import folium
import folium.plugins
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.cards import glass_card
from app.components.layout import (
    page_header, section_header, divider, footer, info_banner,
)
from app.components.cards import (
    prediction_result_card, timing_breakdown, inference_pipeline_progress,
)
from app.components.image_viewer import render_image_viewer
from app.utils.map_data import (
    load_geojson, load_labels, merge_labels,
    get_filtered_features, features_to_centroids,
    patch_exists_on_disk, get_patch_path,
)

# ── Constants ────────────────────────────────────────────────────
ROI_CENTER = [-11.25, -61.5]
ROI_BOUNDS = [[-12.5, -63.5], [-10.0, -59.5]]
DEFAULT_ZOOM = 8

COLOR_DEFOR = "#EF4444"
COLOR_SAFE = "#00C896"
COLOR_UNLABELED = "#6B7280"
COLOR_HIGHLIGHT = "#FBBF24"
COLOR_PENDING = "#F59E0B"
COLOR_PRED_DEFOR = "#DC2626"
COLOR_PRED_SAFE = "#10B981"

STYLE_DEFOR = {"fillColor": COLOR_DEFOR, "color": COLOR_DEFOR, "weight": 1, "fillOpacity": 0.45}
STYLE_SAFE = {"fillColor": COLOR_SAFE, "color": COLOR_SAFE, "weight": 1, "fillOpacity": 0.45}
STYLE_UNLABELED = {"fillColor": COLOR_UNLABELED, "color": COLOR_UNLABELED, "weight": 1, "fillOpacity": 0.20}
STYLE_HIGHLIGHT = {"fillColor": COLOR_HIGHLIGHT, "color": COLOR_HIGHLIGHT, "weight": 3, "fillOpacity": 0.65}
STYLE_PRED_DEFOR = {"fillColor": COLOR_PRED_DEFOR, "color": COLOR_PRED_DEFOR, "weight": 2, "fillOpacity": 0.70}
STYLE_PRED_SAFE = {"fillColor": COLOR_PRED_SAFE, "color": COLOR_PRED_SAFE, "weight": 2, "fillOpacity": 0.70}

JS_COORDS_DISPLAY = """
<script>
(function(){
  var d=document.createElement('div');
  d.id='coord-display';
  d.style.cssText='position:fixed;bottom:32px;right:12px;z-index:9999;background:rgba(11,21,39,0.88);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:8px 14px;font-family:JetBrains Mono,monospace;font-size:12px;color:#E8ECF1;pointer-events:none;min-width:180px;';
  d.innerHTML='<span style="color:#8B97AD;">Move cursor over map</span>';
  document.body.appendChild(d);
  var m=document.querySelector('.folium-map');
  if(m){m.addEventListener('mousemove',function(e){var ll=m._leaflet_map?m._leaflet_map.mouseEventToLatLng(e):null;if(ll){d.innerHTML='<span style="color:#00C896;">Lat</span> '+ll.lat.toFixed(6)+'  <span style="color:#00C896;">Lon</span> '+ll.lng.toFixed(6);}});}
})();
</script>
"""


# ── Helpers ──────────────────────────────────────────────────────
def _label_color(label) -> str:
    if label == 1: return COLOR_DEFOR
    if label == 0: return COLOR_SAFE
    return COLOR_UNLABELED

def _label_text(label) -> str:
    if label == 1: return "Deforestation"
    if label == 0: return "No Deforestation"
    return "Unlabeled"

def _prediction_color(pred: int) -> str:
    return COLOR_PRED_DEFOR if pred == 1 else COLOR_PRED_SAFE

def _prediction_text(pred: int) -> str:
    return "Deforestation" if pred == 1 else "No Deforestation"

def _popup_html(props: dict) -> str:
    pid = props.get("patch_id", "N/A")
    lat = props.get("centroid_lat", 0)
    lon = props.get("centroid_lon", 0)
    fc = props.get("forest_coverage", "N/A")
    label = props.get("label")
    label_txt = _label_text(label)
    label_color = _label_color(label)

    extra = ""
    lp = props.get("loss_percentage")
    tp = props.get("tree_cover_percentage")
    ly = props.get("loss_year")
    if lp is not None:
        extra += f'<tr><td style="padding:4px 8px;color:#8B97AD;font-size:12px;">Loss %</td><td style="padding:4px 8px;font-weight:600;font-size:12px;">{lp:.2f}%</td></tr>'
    if tp is not None:
        extra += f'<tr><td style="padding:4px 8px;color:#8B97AD;font-size:12px;">Tree Cover</td><td style="padding:4px 8px;font-weight:600;font-size:12px;">{tp:.1f}%</td></tr>'
    if ly is not None and ly != 0:
        extra += f'<tr><td style="padding:4px 8px;color:#8B97AD;font-size:12px;">Loss Year</td><td style="padding:4px 8px;font-weight:600;font-size:12px;">{int(ly)}</td></tr>'

    return f'''
    <div style="font-family:Inter,sans-serif;min-width:220px;padding:4px;">
      <div style="font-size:14px;font-weight:800;color:#E8ECF1;margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:8px;">{pid}</div>
      <table style="width:100%;border-collapse:collapse;">
        <tr><td style="padding:4px 8px;color:#8B97AD;font-size:12px;">Latitude</td><td style="padding:4px 8px;font-weight:600;font-family:JetBrains Mono,monospace;font-size:12px;">{lat:.6f}</td></tr>
        <tr><td style="padding:4px 8px;color:#8B97AD;font-size:12px;">Longitude</td><td style="padding:4px 8px;font-weight:600;font-family:JetBrains Mono,monospace;font-size:12px;">{lon:.6f}</td></tr>
        <tr><td style="padding:4px 8px;color:#8B97AD;font-size:12px;">Forest Cover</td><td style="padding:4px 8px;font-weight:600;font-size:12px;">{fc}%</td></tr>
        <tr><td style="padding:4px 8px;color:#8B97AD;font-size:12px;">Ground Truth</td><td style="padding:4px 8px;font-size:12px;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{label_color};box-shadow:0 0 6px {label_color};margin-right:6px;vertical-align:middle;"></span><span style="font-weight:600;color:{label_color};">{label_txt}</span></td></tr>
        {extra}
      </table>
    </div>'''


def _style_function(feature, selected_id: str = "", predictions: dict = None) -> dict:
    props = feature.get("properties", {})
    pid = props.get("patch_id", "")
    if pid == selected_id:
        return STYLE_HIGHLIGHT
    if predictions and pid in predictions:
        p = predictions[pid]
        return STYLE_PRED_DEFOR if p["prediction"] == 1 else STYLE_PRED_SAFE
    label = props.get("label")
    if label == 1: return STYLE_DEFOR
    if label == 0: return STYLE_SAFE
    return STYLE_UNLABELED


def _highlight_function(feature, selected_id: str = "", predictions: dict = None) -> dict:
    props = feature.get("properties", {})
    pid = props.get("patch_id", "")
    if pid == selected_id:
        return {"weight": 4, "fillOpacity": 0.75}
    if predictions and pid in predictions:
        return {"weight": 3, "fillOpacity": 0.80}
    return {"weight": 3, "fillOpacity": 0.65}


# ── Prediction Panel ─────────────────────────────────────────────
def _render_prediction_panel(pid: str, result: dict) -> None:
    """Render the prediction result, timing, pipeline, and image viewer."""
    from pathlib import Path as _P

    prediction_result_card(
        prediction=result["prediction"],
        probability=result["probability"],
        confidence=result["confidence"],
        threshold=result["threshold"],
        model_name="ResNet18",
        latency_ms=result["time_total_s"] * 1000,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    timing_breakdown(result)

    st.markdown("<br>", unsafe_allow_html=True)

    inference_pipeline_progress(result)

    overlay_path = result.get("overlay_path")
    if overlay_path:
        overlay_file = _P(overlay_path)
        if overlay_file.exists():
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Imagery Analysis", f"RGB · Grad-CAM · Overlay — {pid}", "🖼️")
            render_image_viewer(
                image_path=overlay_file,
                patch_id=pid,
                panel_labels=["RGB Composite", "Grad-CAM Heatmap", "Overlay"],
                panel_icons=["🛰️", "🔥", "🎯"],
                key_prefix=f"pred_{pid}",
            )


def _render_pending_card(pid: str) -> None:
    """Render a pending state card."""
    st.markdown(f'''
    <div class="glass-card glass-card--no-hover" style="padding:32px;text-align:center;
        border:1px dashed rgba(245,158,11,0.25);background:rgba(245,158,11,0.04);">
        <div style="font-size:36px;margin-bottom:10px;opacity:0.6;">⏳</div>
        <div style="font-size:13px;text-transform:uppercase;letter-spacing:1.2px;
            color:var(--text-muted);margin-bottom:6px;font-weight:600;">Awaiting Prediction</div>
        <div style="font-size:20px;font-weight:800;color:var(--text-primary);margin-bottom:6px;">No Analysis Yet</div>
        <div style="font-size:12px;color:var(--text-muted);">Click <strong>"▶ Run Prediction"</strong> to analyze {pid}</div>
    </div>''', unsafe_allow_html=True)


# ── Main Render ──────────────────────────────────────────────────
def render() -> None:
    page_header(
        "Map Explorer",
        "Interactive GIS visualization with on-demand deforestation inference",
        icon="🗺️",
        icon_gradient="linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%)",
    )

    # ── Load Data ─────────────────────────────────────────────────
    geojson = load_geojson()
    labels_df = load_labels()
    geojson_merged = merge_labels(geojson, labels_df)

    # Session state for predictions
    if "map_predictions" not in st.session_state:
        st.session_state["map_predictions"] = {}

    # ── Sidebar Filters ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;
                    color:var(--text-muted);margin-bottom:14px;margin-top:8px;">Map Filters</div>
        """, unsafe_allow_html=True)

        display_mode = st.radio("Display Mode", ["Polygons", "Markers"],
                                key="map_display_mode", horizontal=True)
        label_filter = st.selectbox("Ground Truth",
            ["All", "Deforestation", "No Deforestation", "Unlabeled"],
            key="map_label_filter")
        fc_range = st.slider("Forest Coverage %", 0.0, 100.0, (0.0, 100.0),
                             key="map_fc_range")
        selected_only = st.checkbox("Training patches only (5,001)", value=False,
                                    key="map_selected_only")
        max_features = st.slider("Max patches displayed", 500, 25000, 8000, step=500,
                                 key="map_max_features",
                                 help="Reduce for faster rendering")

    # ── Apply Filters ─────────────────────────────────────────────
    filtered = get_filtered_features(geojson_merged, label_filter=label_filter,
                                     fc_min=fc_range[0], fc_max=fc_range[1],
                                     selected_only=selected_only)
    display_features = filtered[:max_features]
    total_filtered = len(filtered)
    total_displayed = len(display_features)
    predictions = st.session_state["map_predictions"]
    n_predicted = sum(1 for pid in predictions if any(f["properties"]["patch_id"] == pid for f in display_features))

    # ── Stats Banner ──────────────────────────────────────────────
    parts = [f"**{total_displayed:,}** patches visible"]
    if total_filtered > total_displayed:
        parts.append(f"of **{total_filtered:,}** matching")
    parts.append(f"(of **{len(geojson_merged['features']):,}** total)")
    pred_info = f" &nbsp;|&nbsp; **{n_predicted}** predicted" if n_predicted > 0 else ""
    st.markdown(f'''
    <div class="info-banner anim-fade">
        <span class="info-banner__icon">🗺️</span>
        <span class="info-banner__text">{" ".join(parts)}{pred_info}</span>
        <span class="info-banner__sub">Display: {display_mode}</span>
    </div>''', unsafe_allow_html=True)

    # ── Search Box ────────────────────────────────────────────────
    sc, fc = st.columns([4, 1])
    with sc:
        search_id = st.text_input("Search by Patch ID", placeholder="e.g. patch_000000",
                                  key="map_search_id", label_visibility="collapsed")
    with fc:
        search_btn = st.button("🔍 Fly to Patch", key="map_fly_btn", use_container_width=True)

    fly_lat, fly_lon = None, None
    if search_id and search_btn:
        for feat in geojson_merged["features"]:
            if feat["properties"].get("patch_id") == search_id:
                fly_lat = feat["properties"]["centroid_lat"]
                fly_lon = feat["properties"]["centroid_lon"]
                break
        if fly_lat is None:
            st.warning(f"Patch '{search_id}' not found in the dataset.")

    # ── Build Map ─────────────────────────────────────────────────
    selected_id = st.session_state.get("map_selected_patch", "")

    m = folium.Map(location=ROI_CENTER, zoom_start=DEFAULT_ZOOM, tiles=None,
                   control_scale=True, prefer_canvas=True)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="🛰️ Esri Satellite", overlay=False).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap", name="🗺️ OpenStreetMap", overlay=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="🌙 Dark Basemap", overlay=False).add_to(m)

    folium.plugins.Fullscreen(position="topleft").add_to(m)
    folium.plugins.MousePosition(position="bottomleft", prefix="Lat/Lon:",
                                 num_digits=6, separator=" , ").add_to(m)
    folium.plugins.Draw(export=True, position="topleft",
        draw_options={"polyline": False, "circle": False, "circlemarker": False, "marker": False}).add_to(m)

    # ── Add Patches ───────────────────────────────────────────────
    if display_mode == "Polygons":
        layer = folium.GeoJson(
            data={"type": "FeatureCollection", "features": display_features},
            name="📊 Patch Polygons",
            style_function=lambda f: _style_function(f, selected_id, predictions),
            highlight_function=lambda f: _highlight_function(f, selected_id, predictions),
            tooltip=folium.GeoJsonTooltip(
                fields=["patch_id", "forest_coverage"],
                aliases=["Patch", "Forest %"],
                style="background-color:#111D33;border:1px solid rgba(255,255,255,0.1);border-radius:6px;color:#E8ECF1;font-family:Inter,sans-serif;font-size:12px;padding:6px 10px;",
                sticky=True),
            show=True)
        layer.add_to(m)
    else:
        mg = folium.FeatureGroup(name="📍 Patch Markers", show=True)
        for c in features_to_centroids(display_features):
            pid = c.get("patch_id", "")
            if pid in predictions:
                color = _prediction_color(predictions[pid]["prediction"])
            else:
                color = _label_color(c.get("label"))
            folium.CircleMarker(
                location=[c["lat"], c["lon"]], radius=5,
                color=color, fill=True, fill_color=color, fill_opacity=0.7, weight=1,
                popup=folium.Popup(_popup_html(c), max_width=300),
                tooltip=f'{pid}').add_to(mg)
        mg.add_to(m)

    folium.Rectangle(
        bounds=[[ROI_BOUNDS[0][0], ROI_BOUNDS[0][1]], [ROI_BOUNDS[1][0], ROI_BOUNDS[1][1]]],
        color="#3B82F6", weight=2, fill=False, dash_array="8 4",
        tooltip="ROI — Rondonia, Brazil").add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    if fly_lat is not None:
        m.location = [fly_lat, fly_lon]
        m.zoom_start = 14

    m.get_root().html.add_child(folium.Element(JS_COORDS_DISPLAY))

    # ── Render Map ────────────────────────────────────────────────
    map_data = st_folium(m, width=None, height=680,
                          returned_objects=["last_clicked", "all_drawn_features", "zoom"],
                          key="folium_map")

    # ── Click Handling ────────────────────────────────────────────
    if map_data and map_data.get("last_clicked"):
        clat = map_data["last_clicked"]["lat"]
        clon = map_data["last_clicked"]["lng"]
        nearest, min_d = None, float("inf")
        for feat in display_features:
            p = feat["properties"]
            d = (p.get("centroid_lat",0)-clat)**2 + (p.get("centroid_lon",0)-clon)**2
            if d < min_d:
                min_d = d
                nearest = p
        if nearest and min_d < 0.01:
            st.session_state["map_selected_patch"] = nearest.get("patch_id", "")
        else:
            st.session_state.pop("map_selected_patch", None)

    # ── Selected Patch Panel ──────────────────────────────────────
    sel_id = st.session_state.get("map_selected_patch", "")
    if sel_id:
        sel_props = None
        for feat in display_features:
            if feat["properties"].get("patch_id") == sel_id:
                sel_props = feat["properties"]
                break
        if sel_props is None:
            for feat in geojson_merged["features"]:
                if feat["properties"].get("patch_id") == sel_id:
                    sel_props = feat["properties"]
                    break

        if sel_props:
            _show_patch_detail(sel_props, predictions)

    # ── Drawn Features ────────────────────────────────────────────
    if map_data and map_data.get("all_drawn_features"):
        drawn = map_data["all_drawn_features"]
        if drawn:
            section_header("Drawn Features", f"{len(drawn)} shape(s)", "✏️")
            for i, feat in enumerate(drawn):
                geom = feat.get("geometry", {})
                gt = geom.get("type", "?")
                st.markdown(f'''
                <div class="status-card" style="margin-bottom:8px;">
                    <div class="status-card__dot" style="background:#FBBF24;box-shadow:0 0 8px #FBBF24;"></div>
                    <div><div class="status-card__text-title">Feature {i+1}</div>
                    <div class="status-card__text-sub">{gt}</div></div>
                </div>''', unsafe_allow_html=True)

    divider()
    _render_legend(total_filtered, total_displayed, len(geojson_merged["features"]), n_predicted)
    footer()


# ── Patch Detail + Prediction ────────────────────────────────────
def _show_patch_detail(props: dict, predictions: dict) -> None:
    pid = props.get("patch_id", "N/A")
    lat = props.get("centroid_lat", 0)
    lon = props.get("centroid_lon", 0)
    fc = props.get("forest_coverage", 0)
    label = props.get("label")
    loss_pct = props.get("loss_percentage")
    tree_pct = props.get("tree_cover_percentage")
    loss_yr = props.get("loss_year")
    water = props.get("water_fraction", 0)
    obs = props.get("valid_obs_pct", 0)
    label_txt = _label_text(label)
    label_color = _label_color(label)
    has_data = patch_exists_on_disk(pid)
    existing_pred = predictions.get(pid)

    section_header("Selected Patch", pid, "📍")

    # ── Info Metrics ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="mc-icon" style="background:var(--grad-cool);">📍</div>
            <div class="mc-value" style="font-size:18px;font-family:var(--font-mono);background:none;-webkit-text-fill-color:var(--text-primary);">{lat:.4f}</div>
            <div class="mc-label">Latitude</div>
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="mc-icon" style="background:var(--grad-cool);">📍</div>
            <div class="mc-value" style="font-size:18px;font-family:var(--font-mono);background:none;-webkit-text-fill-color:var(--text-primary);">{lon:.4f}</div>
            <div class="mc-label">Longitude</div>
        </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="mc-icon" style="background:var(--grad-primary);">🌲</div>
            <div class="mc-value" style="font-size:18px;background:var(--grad-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{fc}%</div>
            <div class="mc-label">Forest Coverage</div>
        </div>''', unsafe_allow_html=True)
    with c4:
        st.markdown(f'''
        <div class="metric-card">
            <div class="mc-icon" style="background:linear-gradient(135deg,{label_color},{label_color});">🏷️</div>
            <div class="mc-value" style="font-size:16px;background:none;-webkit-text-fill-color:{label_color};">{label_txt}</div>
            <div class="mc-label">Ground Truth</div>
        </div>''', unsafe_allow_html=True)

    # ── Extended Properties ───────────────────────────────────────
    ext = f'''<div class="glass-card glass-card--no-hover" style="padding:20px 24px;">
        <div style="display:flex;gap:32px;flex-wrap:wrap;">
            <div><div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);font-weight:600;">Water %</div>
            <div style="font-size:16px;font-weight:700;font-family:var(--font-mono);margin-top:4px;">{water}%</div></div>
            <div><div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);font-weight:600;">Observations</div>
            <div style="font-size:16px;font-weight:700;font-family:var(--font-mono);margin-top:4px;">{obs}%</div></div>'''
    if tree_pct is not None:
        ext += f'<div><div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);font-weight:600;">Tree Cover</div><div style="font-size:16px;font-weight:700;font-family:var(--font-mono);margin-top:4px;">{tree_pct}%</div></div>'
    if loss_pct is not None and loss_pct > 0:
        ext += f'<div><div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);font-weight:600;">Loss %</div><div style="font-size:16px;font-weight:700;font-family:var(--font-mono);margin-top:4px;color:#EF4444;">{loss_pct:.2f}%</div></div>'
    if loss_yr is not None and loss_yr != 0:
        ext += f'<div><div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);font-weight:600;">Loss Year</div><div style="font-size:16px;font-weight:700;font-family:var(--font-mono);margin-top:4px;color:#F59E0B;">{int(loss_yr)}</div></div>'
    ext += '</div></div>'
    st.markdown(ext, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction Section ────────────────────────────────────────
    if existing_pred:
        _render_prediction_panel(pid, existing_pred)
    elif has_data:
        st.markdown('''
        <div class="glass-card glass-card--no-hover" style="padding:16px 20px;display:flex;
            align-items:center;gap:12px;border-left:3px solid var(--accent);margin-bottom:16px;">
            <span style="font-size:20px;">✅</span>
            <div>
                <div style="font-size:14px;font-weight:700;color:var(--text-primary);">Preprocessed Data Ready</div>
                <div style="font-size:12px;color:var(--text-muted);">12 monthly Sentinel-2 GeoTIFFs on disk</div>
            </div>
        </div>''', unsafe_allow_html=True)

        if st.button("▶ Run Prediction", key=f"run_pred_{pid}", type="primary",
                     use_container_width=True):
            _run_prediction(pid)
    else:
        _render_pending_card(pid)


def _run_prediction(pid: str) -> None:
    """Run inference and store result in session state."""
    from app.utils.inference_wrapper import load_predictor, predict_patch

    threshold = st.session_state.get("sidebar_threshold", 0.5)
    patch_dir = get_patch_path(pid)

    with st.spinner(f"Running inference on {pid}..."):
        t0 = time.perf_counter()
        try:
            result = predict_patch(patch_dir, threshold=threshold)
            elapsed = time.perf_counter() - t0
            result["map_latency_total"] = elapsed
            st.session_state["map_predictions"][pid] = result
            st.success(f"Inference complete — {result['prediction_label']} "
                       f"(P={result['probability']:.4f}) in {elapsed:.1f}s")
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    st.rerun()


# ── Legend ────────────────────────────────────────────────────────
def _render_legend(filtered: int, displayed: int, total: int, n_pred: int = 0) -> None:
    pred_html = ""
    if n_pred > 0:
        pred_html = f'''
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="width:14px;height:14px;border-radius:3px;background:{COLOR_PRED_DEFOR};opacity:0.8;display:inline-block;"></span>
                <span style="font-size:12px;color:var(--text-secondary);">Pred. Deforestation</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="width:14px;height:14px;border-radius:3px;background:{COLOR_PRED_SAFE};opacity:0.8;display:inline-block;"></span>
                <span style="font-size:12px;color:var(--text-secondary);">Pred. No Deforestation</span>
            </div>'''

    st.markdown(f'''
    <div class="glass-card glass-card--no-hover" style="padding:20px 24px;">
        <div style="font-size:13px;font-weight:700;margin-bottom:14px;color:var(--text-primary);">Map Legend</div>
        <div style="display:flex;gap:28px;flex-wrap:wrap;align-items:center;">
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="width:14px;height:14px;border-radius:3px;background:{COLOR_DEFOR};opacity:0.7;display:inline-block;"></span>
                <span style="font-size:12px;color:var(--text-secondary);">Deforestation (GT)</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="width:14px;height:14px;border-radius:3px;background:{COLOR_SAFE};opacity:0.7;display:inline-block;"></span>
                <span style="font-size:12px;color:var(--text-secondary);">No Deforestation (GT)</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="width:14px;height:14px;border-radius:3px;background:{COLOR_UNLABELED};opacity:0.4;display:inline-block;"></span>
                <span style="font-size:12px;color:var(--text-secondary);">Unlabeled</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="width:14px;height:14px;border-radius:3px;background:{COLOR_HIGHLIGHT};opacity:0.8;display:inline-block;"></span>
                <span style="font-size:12px;color:var(--text-secondary);">Selected</span>
            </div>
            {pred_html}
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="width:14px;height:2px;border-top:2px dashed #3B82F6;display:inline-block;"></span>
                <span style="font-size:12px;color:var(--text-secondary);">ROI Boundary</span>
            </div>
        </div>
        <div style="margin-top:12px;font-size:11px;color:var(--text-muted);">
            Showing {displayed:,} of {filtered:,} matching patches ({total:,} total)
            {f" &nbsp;|&nbsp; {n_pred} predicted" if n_pred > 0 else ""}
        </div>
    </div>''', unsafe_allow_html=True)
