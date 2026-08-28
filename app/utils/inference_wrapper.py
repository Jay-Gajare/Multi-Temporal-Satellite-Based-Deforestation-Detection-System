"""Inference wrapper for the Streamlit app.

Caches the model via @st.cache_resource so it loads only once per session.
"""
from __future__ import annotations

import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from inference.predict import DeforestationPredictor

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(threshold: float = 0.5) -> DeforestationPredictor:
    """Load and cache the predictor (loads once per session per threshold)."""
    predictor = DeforestationPredictor(threshold=threshold)
    predictor.load_model()
    return predictor


def predict_patch(
    patch_dir: Path,
    predictor: Optional[DeforestationPredictor] = None,
    threshold: Optional[float] = None,
) -> dict:
    """Run inference on a patch directory."""
    if predictor is None:
        t = threshold if threshold is not None else st.session_state.get("sidebar_threshold", 0.5)
        predictor = load_predictor(t)

    if threshold is not None and predictor.threshold != threshold:
        predictor.threshold = threshold

    result = predictor.predict(patch_dir)
    return result


def predict_from_uploaded_files(
    uploaded_files: list,
    predictor: Optional[DeforestationPredictor] = None,
    threshold: Optional[float] = None,
) -> dict:
    """Run inference on uploaded files (creates a temp directory)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="dew_upload_"))
    try:
        for f in uploaded_files:
            dest = tmp_dir / f.name
            with open(dest, "wb") as out:
                out.write(f.getbuffer())

        if predictor is None:
            t = threshold if threshold is not None else st.session_state.get("sidebar_threshold", 0.5)
            predictor = load_predictor(t)

        result = predictor.predict(tmp_dir)
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
