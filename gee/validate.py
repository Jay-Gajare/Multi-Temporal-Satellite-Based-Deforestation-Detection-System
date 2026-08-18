"""
Feature validation for spectral indices.

For each monthly composite, computes per-band statistics (min, max, mean, std)
and verifies that derived indices fall within the expected mathematical range.

Uses a representative 2x2 degree sample area at 300m scale for fast computation.
"""

from __future__ import annotations

import ee

from configs.settings import (
    DRY_SEASON_MONTHS,
    FEATURE_BANDS,
    INDEX_VALID_MAX,
    INDEX_VALID_MIN,
    WET_SEASON_MONTHS,
)

# Representative sample area centered on ROI for fast statistics
_SAMPLE_AREA = ee.Geometry.Rectangle([-62.5, -12.0, -60.5, -10.0])
_VALIDATION_SCALE = 300  # metres — fast but representative


def validate_composite(composite: ee.Image, month: int) -> dict:
    """
    Compute statistics for a single monthly composite and validate index ranges.

    Uses a representative sample area to keep computation tractable.
    Returns a dict with:
        - month: int
        - stats: {band: {min, max, mean, std}, ...}
        - validation: {band: {in_range: bool, min_actual, max_actual}, ...}
        - passed: bool (True if all index bands are within valid range)
    """
    reducer = (
        ee.Reducer.min()
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
    )

    # Compute all bands in a single reduceRegion call
    result = composite.select(FEATURE_BANDS).reduceRegion(
        reducer=reducer,
        geometry=_SAMPLE_AREA,
        scale=_VALIDATION_SCALE,
        maxPixels=1e9,
    ).getInfo()

    stats_dict = {}
    for band in FEATURE_BANDS:
        stats_dict[band] = {
            "min": result.get(f"{band}_min"),
            "max": result.get(f"{band}_max"),
            "mean": result.get(f"{band}_mean"),
            "std": result.get(f"{band}_stdDev"),
        }

    # Validate index bands only
    index_bands = ["NDVI", "NBR", "NDMI"]
    validation = {}
    all_passed = True

    for band in index_bands:
        s = stats_dict[band]
        min_val = s["min"]
        max_val = s["max"]
        in_range = True
        if min_val is not None and min_val < INDEX_VALID_MIN:
            in_range = False
            all_passed = False
        if max_val is not None and max_val > INDEX_VALID_MAX:
            in_range = False
            all_passed = False
        validation[band] = {
            "in_range": in_range,
            "min_actual": min_val,
            "max_actual": max_val,
        }

    return {
        "month": month,
        "stats": stats_dict,
        "validation": validation,
        "passed": all_passed,
    }


def seasonal_comparison(composites: list[dict]) -> dict:
    """
    Compare dry-season vs wet-season index statistics.

    Uses the same representative sample area.
    """
    index_bands = ["NDVI", "NBR", "NDMI"]

    dry_imgs = []
    wet_imgs = []
    for entry in composites:
        if entry["composite"] is None:
            continue
        if entry["month"] in DRY_SEASON_MONTHS:
            dry_imgs.append(entry["composite"])
        elif entry["month"] in WET_SEASON_MONTHS:
            wet_imgs.append(entry["composite"])

    def _season_stats(imgs: list[ee.Image], label: str) -> dict:
        if not imgs:
            return {"label": label, "count": 0, "means": {}}
        col = ee.ImageCollection(imgs)
        means = {}
        for band in index_bands:
            avg = col.select(band).mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=_SAMPLE_AREA,
                scale=_VALIDATION_SCALE,
                maxPixels=1e9,
            ).getInfo()
            means[band] = avg.get(band)
        return {"label": label, "count": len(imgs), "means": means}

    dry_stats = _season_stats(dry_imgs, "dry_season")
    wet_stats = _season_stats(wet_imgs, "wet_season")

    return {
        "dry_season": dry_stats,
        "wet_season": wet_stats,
    }
