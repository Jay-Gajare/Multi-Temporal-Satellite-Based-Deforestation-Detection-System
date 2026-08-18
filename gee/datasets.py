"""
Reusable helpers for loading GEE datasets and building the ROI geometry.
"""

from __future__ import annotations

import ee

from configs.settings import (
    DATE_END,
    DATE_START,
    HANSEN_BANDS,
    ROI_COORDINATES,
    S2_BANDS,
    SCALE,
)

HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"


def get_roi() -> ee.Geometry.Rectangle:
    """Return the ROI as an ee.Geometry.Rectangle."""
    coords = ROI_COORDINATES
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return ee.Geometry.Rectangle([min(lons), min(lats), max(lons), max(lats)])


def get_viz_region(scale_factor: float = 0.3) -> ee.Geometry.Rectangle:
    """
    Return a small region around ROI center for visualization thumbnails.

    Using a ~1.2x0.75 degree crop centered on the ROI for fast rendering.
    """
    cx, cy = -61.5, -11.25
    hw, hh = 0.6, 0.375
    return ee.Geometry.Rectangle([cx - hw, cy - hh, cx + hw, cy + hh])


def load_hansen() -> ee.Image:
    """Load Hansen Global Forest Change v1.13 (2025 release)."""
    return ee.Image(HANSEN_ASSET).select(HANSEN_BANDS)


def load_sentinel2_raw(
    start: str = DATE_START, end: str = DATE_END
) -> ee.ImageCollection:
    """
    Load raw Sentinel-2 SR Harmonized — no cloud masking applied.

    Filtering:
        - Spatial: ROI bounds
        - Temporal: date range
        - Cloud metadata: < 30% cloudy pixel percentage (scene-level filter)

    Band selection and scaling are deferred to the cloud masking step.
    """
    roi = get_roi()
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )
