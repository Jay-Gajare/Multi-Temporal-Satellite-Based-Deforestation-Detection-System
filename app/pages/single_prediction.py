"""Single Prediction — upload GeoTIFFs, run inference, Grad-CAM visualization."""
from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.cards import metric_card, prediction_card
from app.components.layout import (
    page_header, section_header, divider, footer, empty_state, info_banner,
)
from app.utils.inference_wrapper import load_predictor, predict_from_uploaded_files
from app.utils.session import add_to_history


def render() -> None:
    page_header(
        "Single Prediction",
        "Upload a patch directory or individual GeoTIFFs for deforestation analysis",
        icon="🔍",
        icon_gradient="var(--grad-accent)",
    )

    predictor = load_predictor(st.session_state.get("sidebar_threshold", 0.5))

    # ── Upload Zone ───────────────────────────────────────────────
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-zone__icon">📁</div>
        <div class="upload-zone__title">Drop files here or click to browse</div>
        <div class="upload-zone__sub">
            Accepts GeoTIFF files (.tif) or a folder of 12 monthly images<br>
            Supported bands: B2, B3, B4, B8, B11, B12 (and derived indices)
        </div>
    </div>""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="single_upload",
    )

    if uploaded_files:
        # ── File Info Banner ──────────────────────────────────────
        file_names = ", ".join(f.name for f in uploaded_files[:5])
        extra = f" +{len(uploaded_files) - 5} more" if len(uploaded_files) > 5 else ""
        info_banner(
            f"{len(uploaded_files)} file(s) uploaded",
            icon="📎",
            sub=f"{file_names}{extra}",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(" Run Prediction", key="run_single", type="primary", width='stretch'):
            with st.spinner("Preprocessing and running inference..."):
                result = predict_from_uploaded_files(
                    uploaded_files,
                    predictor=predictor,
                    threshold=st.session_state.get("sidebar_threshold", 0.5),
                )

            if result:
                st.session_state.current_prediction = result
                add_to_history(result)

                # ── Prediction Result ─────────────────────────────
                prediction_card(
                    prediction=result["prediction"],
                    probability=result["probability"],
                    confidence=result["confidence"],
                    inference_time=result["time_total_s"] * 1000,
                    threshold=st.session_state.get("sidebar_threshold", 0.5),
                    model_name="ResNet18",
                )

                # ── Timing Metrics ────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    metric_card("Inference Time", f"{result['time_total_s']*1000:.0f}ms",
                                icon="⏱", gradient="var(--grad-accent)")
                with c2:
                    metric_card("Model Load", f"{result.get('time_inference_s', 0)*1000:.0f}ms",
                                icon="🧠", gradient="var(--grad-cool)")
                with c3:
                    metric_card("Grad-CAM", f"{result.get('time_gradcam_s', 0)*1000:.0f}ms",
                                icon="🔥", gradient="var(--grad-warm)")
                with c4:
                    metric_card("Visualization", f"{result.get('time_visualize_s', 0)*1000:.0f}ms",
                                icon="📊", gradient="var(--grad-safe)")

                # ── Visualizations ────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Visualizations", "Model attention and prediction overlay", "📊")

                viz_path = Path(result.get("output_dir", ""))
                gradcam_path = viz_path / "gradcam.png"
                overlay_path = viz_path / "prediction_overlay.png"

                if gradcam_path.exists() or overlay_path.exists():
                    tabs = st.tabs(["Grad-CAM Heatmap", "Prediction Overlay"])

                    with tabs[0]:
                        if gradcam_path.exists():
                            img = Image.open(gradcam_path)
                            st.image(img, caption="Grad-CAM — Areas the model focuses on", width='stretch')
                        else:
                            st.info("Grad-CAM image not available")

                    with tabs[1]:
                        if overlay_path.exists():
                            img = Image.open(overlay_path)
                            st.image(img, caption="Prediction overlay on RGB composite", width='stretch')
                        else:
                            st.info("Overlay image not available")

                    # ── Download Buttons ──────────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if gradcam_path.exists():
                            with open(gradcam_path, "rb") as f:
                                st.download_button("Download Grad-CAM", f.read(),
                                                   file_name="gradcam.png", mime="image/png",
                                                   width='stretch')
                    with c2:
                        if overlay_path.exists():
                            with open(overlay_path, "rb") as f:
                                st.download_button("Download Overlay", f.read(),
                                                   file_name="overlay.png", mime="image/png",
                                                   width='stretch')
                    with c3:
                        export = {k: v for k, v in result.items()
                                  if not isinstance(v, np.ndarray) and k != "temporal_stack"}
                        st.download_button("Download JSON", data=json.dumps(export, indent=2, default=str),
                                           file_name="prediction.json", mime="application/json",
                                           width='stretch')
                else:
                    st.warning("No visualizations generated. Check that the input files are valid GeoTIFFs.")

    else:
        empty_state(
            icon="🛰️",
            title="No files uploaded yet",
            description="Upload Sentinel-2 GeoTIFF files (monthly composites) to run deforestation detection. "
                        "The model expects 12 monthly images with bands B2, B3, B4, B8, B11, B12.",
        )

    footer()
