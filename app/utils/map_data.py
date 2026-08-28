"""GeoJSON data loading and caching for the Map Explorer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


EXPORT_DIR = Path(__file__).resolve().parent.parent.parent / "exports"
GEOJSON_PATH = EXPORT_DIR / "patch_grid_filtered.geojson"
LABELS_PATH = EXPORT_DIR / "patch_labels.csv"
METADATA_PATH = EXPORT_DIR / "final_patch_metadata.csv"
PATCHES_DIR = EXPORT_DIR / "patches"


@st.cache_data(ttl=3600, show_spinner="Loading patch geometries…")
def load_geojson() -> dict:
    """Load the filtered patch grid GeoJSON (23,547 features)."""
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner="Loading ground truth labels…")
def load_labels() -> pd.DataFrame:
    """Load the patch labels CSV (5,001 labeled patches)."""
    df = pd.read_csv(LABELS_PATH)
    return df


@st.cache_data(ttl=3600, show_spinner="Loading metadata…")
def load_metadata() -> pd.DataFrame:
    """Load the final patch metadata CSV (23,547 patches)."""
    return pd.read_csv(METADATA_PATH)


def merge_labels(geojson: dict, labels_df: pd.DataFrame) -> dict:
    """Merge ground truth labels into GeoJSON feature properties.

    Adds: label, loss_percentage, tree_cover_percentage, loss_year, gain_percentage
    to each feature that has a matching patch_id in labels_df.
    """
    label_lookup = labels_df.set_index("patch_id").to_dict(orient="index")
    merged = 0
    for feature in geojson["features"]:
        pid = feature["properties"].get("patch_id", "")
        if pid in label_lookup:
            feature["properties"].update(label_lookup[pid])
            merged += 1
    return geojson


def get_filtered_features(
    geojson: dict,
    label_filter: str = "All",
    fc_min: float = 0.0,
    fc_max: float = 100.0,
    selected_only: bool = False,
) -> list[dict]:
    """Return filtered GeoJSON features based on UI controls.

    Filters:
    - label_filter: "All", "Deforestation", "No Deforestation", "Unlabeled"
    - fc_min/fc_max: forest coverage range
    - selected_only: only show the 5,001 training patches
    """
    features = []
    for feat in geojson["features"]:
        props = feat["properties"]

        # Forest coverage filter
        fc = props.get("forest_coverage", 0)
        if fc < fc_min or fc > fc_max:
            continue

        # Label filter
        label = props.get("label")
        if label_filter == "Deforestation":
            if label != 1:
                continue
        elif label_filter == "No Deforestation":
            if label != 0:
                continue
        elif label_filter == "Unlabeled":
            if label is not None:
                continue

        # Selected only filter
        if selected_only and label is None:
            continue

        features.append(feat)

    return features


def features_to_centroids(features: list[dict]) -> list[dict]:
    """Extract centroid coordinates from features for marker mode."""
    centroids = []
    for feat in features:
        props = feat["properties"]
        centroids.append({
            "patch_id": props.get("patch_id", ""),
            "lat": props.get("centroid_lat", 0),
            "lon": props.get("centroid_lon", 0),
            "label": props.get("label"),
            "forest_coverage": props.get("forest_coverage", 0),
            "loss_percentage": props.get("loss_percentage"),
            "loss_year": props.get("loss_year"),
            "tree_cover_percentage": props.get("tree_cover_percentage"),
        })
    return centroids


@st.cache_data(ttl=3600, show_spinner=False)
def get_available_patch_ids() -> set[str]:
    """Return set of patch_ids that have data on disk (month_01.tif .. month_12.tif)."""
    if not PATCHES_DIR.exists():
        return set()
    return {
        d.name for d in PATCHES_DIR.iterdir()
        if d.is_dir() and (d / "month_01.tif").exists()
    }


def patch_exists_on_disk(patch_id: str) -> bool:
    """Check if a specific patch has preprocessed data on disk."""
    return patch_id in get_available_patch_ids()


def get_patch_path(patch_id: str) -> Path:
    """Return the filesystem path for a patch directory."""
    return PATCHES_DIR / patch_id
