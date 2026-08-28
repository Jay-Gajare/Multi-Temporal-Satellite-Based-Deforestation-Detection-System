"""Session state management."""
from __future__ import annotations

import streamlit as st
from typing import Any


def init_session() -> None:
    """Initialize session state variables with defaults."""
    defaults = {
        "prediction_history": [],
        "batch_results": None,
        "batch_results_v2": None,
        "current_prediction": None,
        "model_loaded": False,
        "map_predictions": {},
        "map_selected_patch": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def add_to_history(result: dict) -> None:
    """Add a prediction result to session history (max 50 entries)."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    st.session_state.prediction_history.append(result)
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]


def get_history() -> list[dict]:
    """Get the current prediction history list."""
    return st.session_state.get("prediction_history", [])


def clear_history() -> None:
    """Clear the prediction history."""
    st.session_state.prediction_history = []
