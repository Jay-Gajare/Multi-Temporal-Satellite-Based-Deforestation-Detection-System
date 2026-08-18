"""
GEE extraction and GeoTIFF writing utilities for patch export.

Uses ee.data.computePixels for direct pixel extraction without Drive.
Supports both per-patch and tile-based extraction (512x512 tiles
covering 8x8 patches each, cropped locally).
"""

from __future__ import annotations

import math
import ee
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from configs.settings import FEATURE_BANDS, PATCH_SIZE, SCALE

TILE_PATCHES = 8
TILE_SIZE = PATCH_SIZE * TILE_PATCHES  # 512 pixels


def build_patch_grid_spec(
    west: float, south: float, east: float, north: float,
) -> dict:
    """Build a computePixels grid spec for a single patch (64x64)."""
    deg_x = (east - west) / PATCH_SIZE
    deg_y = (south - north) / PATCH_SIZE
    return {
        "dimensions": {"width": PATCH_SIZE, "height": PATCH_SIZE},
        "affineTransform": {
            "scaleX": deg_x,
            "shearX": 0,
            "shearY": 0,
            "scaleY": deg_y,
            "translateX": west,
            "translateY": north,
        },
        "crsCode": "EPSG:4326",
    }


def build_tile_grid_spec(
    west: float, south: float, east: float, north: float,
) -> dict:
    """Build a computePixels grid spec for a tile (512x512)."""
    deg_x = (east - west) / TILE_SIZE
    deg_y = (south - north) / TILE_SIZE
    return {
        "dimensions": {"width": TILE_SIZE, "height": TILE_SIZE},
        "affineTransform": {
            "scaleX": deg_x,
            "shearX": 0,
            "shearY": 0,
            "scaleY": deg_y,
            "translateX": west,
            "translateY": north,
        },
        "crsCode": "EPSG:4326",
    }


def extract_patch(
    composite: ee.Image,
    west: float, south: float, east: float, north: float,
) -> np.ndarray:
    """Extract a single 64x64 patch from a composite."""
    grid = build_patch_grid_spec(west, south, east, north)
    return ee.data.computePixels({
        "expression": composite.select(FEATURE_BANDS),
        "grid": grid,
        "fileFormat": "NUMPY_NDARRAY",
    })


def extract_tile(
    composite: ee.Image,
    west: float, south: float, east: float, north: float,
) -> np.ndarray:
    """Extract a 512x512 tile from a composite."""
    grid = build_tile_grid_spec(west, south, east, north)
    return ee.data.computePixels({
        "expression": composite.select(FEATURE_BANDS),
        "grid": grid,
        "fileFormat": "NUMPY_NDARRAY",
    })


def crop_tile_to_patches(
    tile_arr: np.ndarray,
    tile_west: float, tile_south: float,
    tile_east: float, tile_north: float,
    patch_indices: list[tuple[int, int, str]],
) -> list[tuple[str, np.ndarray, tuple[float, float, float, float]]]:
    """
    Crop a 512x512 tile into individual 64x64 patches.

    Parameters
    ----------
    tile_arr : structured ndarray (512, 512) with band names
    tile_west, tile_south, tile_east, tile_north : tile geographic bounds
    patch_indices : list of (row_in_tile, col_in_tile, patch_id)

    Returns
    -------
    list of (patch_id, patch_arr, (west, south, east, north))
    """
    n_bands = len(tile_arr.dtype.names)
    tile_height = tile_arr.shape[0]
    tile_width = tile_arr.shape[1]

    results = []
    for row_in_tile, col_in_tile, patch_id in patch_indices:
        r0 = row_in_tile * PATCH_SIZE
        r1 = r0 + PATCH_SIZE
        c0 = col_in_tile * PATCH_SIZE
        c1 = c0 + PATCH_SIZE

        if r1 > tile_height or c1 > tile_width:
            continue

        patch_arr = np.empty(
            (PATCH_SIZE, PATCH_SIZE), dtype=tile_arr.dtype
        )
        for name in tile_arr.dtype.names:
            patch_arr[name] = tile_arr[name][r0:r1, c0:c1]

        pw = tile_west + (c0 / tile_width) * (tile_east - tile_west)
        pe = tile_west + (c1 / tile_width) * (tile_east - tile_west)
        pn = tile_north - (r0 / tile_height) * (tile_north - tile_south)
        ps = tile_north - (r1 / tile_height) * (tile_north - tile_south)

        results.append((patch_id, patch_arr, (pw, ps, pe, pn)))

    return results


def save_patch_geotiff(
    arr: np.ndarray,
    output_path: str,
    west: float, south: float, east: float, north: float,
) -> None:
    """Save a structured numpy array as a multi-band GeoTIFF."""
    n_bands = len(arr.dtype.names)
    height, width = arr.shape
    transform = from_bounds(west, south, east, north, width, height)

    data = np.zeros((n_bands, height, width), dtype=np.float32)
    for i, band_name in enumerate(arr.dtype.names):
        data[i] = arr[band_name]

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=height, width=width, count=n_bands,
        dtype="float32", crs="EPSG:4326", transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data)
        for i, band_name in enumerate(arr.dtype.names):
            dst.set_band_description(i + 1, band_name)


def save_empty_geotiff(
    output_path: str,
    west: float, south: float, east: float, north: float,
) -> None:
    """Write a zero-filled GeoTIFF for months with no data."""
    n_bands = len(FEATURE_BANDS)
    transform = from_bounds(west, south, east, north, PATCH_SIZE, PATCH_SIZE)
    data = np.zeros((n_bands, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=PATCH_SIZE, width=PATCH_SIZE, count=n_bands,
        dtype="float32", crs="EPSG:4326", transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data)
        for i, b in enumerate(FEATURE_BANDS):
            dst.set_band_description(i + 1, b)


def patch_bounds(
    centroid_lon: float, centroid_lat: float,
) -> tuple[float, float, float, float]:
    """Compute (west, south, east, north) for a patch."""
    half_deg_lat = (PATCH_SIZE * SCALE / 2) / 111_320.0
    mid_lat_rad = math.radians(abs(centroid_lat))
    half_deg_lon = (PATCH_SIZE * SCALE / 2) / (111_320.0 * math.cos(mid_lat_rad))
    return (
        centroid_lon - half_deg_lon,
        centroid_lat - half_deg_lat,
        centroid_lon + half_deg_lon,
        centroid_lat + half_deg_lat,
    )


def build_monthly_composites(s2_masked: ee.ImageCollection) -> list[dict]:
    """Build 12 monthly composites with FEATURE_BANDS."""
    from configs.settings import MONTHLY_RANGES
    from gee.features import add_all_indices

    results = []
    for month_num, start_str, end_str in MONTHLY_RANGES:
        monthly = s2_masked.filterDate(start_str, end_str)
        count = monthly.size().getInfo()
        if count == 0:
            results.append({"month": month_num, "composite": None, "image_count": 0})
        else:
            median = monthly.median().select(FEATURE_BANDS[:6])
            composite = add_all_indices(median)
            composite = (
                composite
                .set("system:time_start", ee.Date(start_str).millis())
                .set("month", month_num)
            )
            results.append({
                "month": month_num,
                "composite": composite,
                "image_count": count,
            })
    return results
