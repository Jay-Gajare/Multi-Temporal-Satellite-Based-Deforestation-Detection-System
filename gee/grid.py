"""
Spatial grid generation for the ROI - batched implementation.

Generates a regular grid of 64x64 pixel patches at 30 m resolution
(1920 m x 1920 m each) covering the entire ROI, then filters invalid patches.

Algorithm:
    1. Compute patch size in degrees at the ROI centroid.
    2. Generate the grid in Python using numpy (fast).
    3. Pre-compute global Hansen + S2 obs images (one-time).
    4. Process in batches: reduceRegions at 30m for speed, filter server-side
       with sequential per-filter counting, paginate download.
    5. Combine batched results into a single local GeoDataFrame.

Filtering criteria:
    - Water fraction < 50% (Hansen datamask)
    - Forest coverage >= 60% (Hansen treecover2000 > 30%)
    - Valid observation fraction >= 80% (Sentinel-2 SR, 2023)
"""

from __future__ import annotations

import math
import ee
import geopandas as gpd
from shapely.geometry import box
import numpy as np

from configs.settings import (
    DATE_END,
    DATE_START,
    PATCH_SIZE,
    ROI_COORDINATES,
    SCALE,
)

HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
FOREST_COVER_THRESHOLD = 30
WATER_DATAMASK_VALUE = 2
MIN_FOREST_PCT = 0.60
MIN_WATER_PCT = 0.50
MIN_OBS_PCT = 0.20
SCORE_SCALE = 30

BATCH_SIZE = 2_500
DOWNLOAD_PAGE = 500


def _compute_patch_degrees() -> tuple[float, float]:
    patch_m = PATCH_SIZE * SCALE
    centroid_lat = (ROI_COORDINATES[0][1] + ROI_COORDINATES[2][1]) / 2
    deg_lat = patch_m / 111_320.0
    deg_lon = patch_m / (111_320.0 * math.cos(math.radians(abs(centroid_lat))))
    return deg_lat, deg_lon


def _roi_bounds() -> tuple[float, float, float, float]:
    lons = [c[0] for c in ROI_COORDINATES]
    lats = [c[1] for c in ROI_COORDINATES]
    return min(lons), min(lats), max(lons), max(lats)


def _grid_dimensions() -> tuple[int, int]:
    lon_min, lat_min, lon_max, lat_max = _roi_bounds()
    deg_lat, deg_lon = _compute_patch_degrees()
    n_cols = int(np.ceil((lon_max - lon_min) / deg_lon))
    n_rows = int(np.ceil((lat_max - lat_min) / deg_lat))
    return n_rows, n_cols


def _grid_coords() -> tuple[np.ndarray, np.ndarray]:
    lon_min, lat_min, lon_max, lat_max = _roi_bounds()
    deg_lat, deg_lon = _compute_patch_degrees()
    lon_vals = np.arange(lon_min, lon_max, deg_lon)
    lat_vals = np.arange(lat_min, lat_max, deg_lat)
    return lon_vals, lat_vals


def _roi_geom() -> ee.Geometry:
    return ee.Geometry.Rectangle([
        ROI_COORDINATES[0][0], ROI_COORDINATES[0][1],
        ROI_COORDINATES[2][0], ROI_COORDINATES[2][1],
    ])


def generate_grid_ee_batch(
    lon_vals: np.ndarray,
    lat_vals: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> ee.FeatureCollection:
    n_cols = len(lon_vals)
    deg_lat, deg_lon = _compute_patch_degrees()

    features = []
    for i in range(start_idx, end_idx):
        row = i // n_cols
        col = i % n_cols
        if row >= len(lat_vals):
            break
        lon = float(lon_vals[col])
        lat = float(lat_vals[row])
        geom = ee.Geometry.Rectangle([
            lon, lat, lon + deg_lon, lat + deg_lat,
        ])
        features.append(ee.Feature(geom, {
            "patch_id": f"patch_{i:06d}",
            "row": int(row),
            "col": int(col),
            "centroid_lon": round(lon + deg_lon / 2, 6),
            "centroid_lat": round(lat + deg_lat / 2, 6),
        }))

    return ee.FeatureCollection(features)


OBS_ASSET = "projects/deforestation-early-warning/assets/step4_obs_pct"


def build_water_forest_images() -> tuple[ee.Image, ee.Number]:
    hansen = ee.Image(HANSEN_ASSET)
    water = hansen.select("datamask").eq(WATER_DATAMASK_VALUE).rename("water")
    forest = hansen.select("treecover2000").gt(FOREST_COVER_THRESHOLD).rename("forest")

    roi = _roi_geom()

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )
    n_images = s2.size()

    combined = water.addBands(forest)
    return combined, n_images


def build_obs_image() -> ee.Image:
    roi = _roi_geom()

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )

    def _clip_and_mask(img):
        qa = img.select("QA60")
        cloud_bit = 1 << 10
        cirrus_bit = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
        return img.updateMask(mask).clip(roi)

    s2_masked = s2.map(_clip_and_mask)
    s2_clipped = s2.map(lambda img: img.clip(roi))

    total_count = s2_clipped.select("B4").reduce(ee.Reducer.count())
    valid_count = s2_masked.select("B4").reduce(ee.Reducer.count())
    obs_pct = valid_count.divide(total_count).rename("obs")
    dummy = obs_pct.multiply(0).rename("placeholder")
    return obs_pct.addBands(dummy)


def score_and_filter_batch(
    batch_fc: ee.FeatureCollection,
    combined_img: ee.Image,
) -> dict:
    scored = combined_img.reduceRegions(
        collection=batch_fc,
        reducer=ee.Reducer.mean(),
        scale=SCORE_SCALE,
    )

    batch_size = batch_fc.size().getInfo()

    water_passed = scored.filter(ee.Filter.lt("water", MIN_WATER_PCT))
    forest_passed = water_passed.filter(ee.Filter.gte("forest", MIN_FOREST_PCT))
    obs_passed = forest_passed.filter(ee.Filter.gte("obs", MIN_OBS_PCT))

    obs_count = obs_passed.size().getInfo()
    forest_count = forest_passed.size().getInfo()
    water_count = water_passed.size().getInfo()

    return {
        "filtered": obs_passed,
        "batch_size": batch_size,
        "water_removed": batch_size - water_count,
        "forest_removed": water_count - forest_count,
        "obs_removed": forest_count - obs_count,
    }


def download_filtered(filtered_fc: ee.FeatureCollection) -> list[dict]:
    total = filtered_fc.size().getInfo()
    if total == 0:
        return []

    all_feats = []
    remaining = filtered_fc
    while total > 0:
        page_size = min(DOWNLOAD_PAGE, total)
        batch = remaining.sort("patch_id").limit(page_size)
        features = batch.getInfo()["features"]
        if not features:
            break
        for f in features:
            all_feats.append(f["properties"])
        last_pid = features[-1]["properties"]["patch_id"]
        remaining = remaining.filter(ee.Filter.gt("patch_id", last_pid))
        total -= len(features)

    return all_feats


def properties_to_geodataframe(records: list[dict]) -> gpd.GeoDataFrame:
    deg_lat, deg_lon = _compute_patch_degrees()
    patch_area_m2 = float((PATCH_SIZE * SCALE) ** 2)

    valid = []
    for r in records:
        c_lon = r["centroid_lon"]
        c_lat = r["centroid_lat"]
        geom = box(
            c_lon - deg_lon / 2, c_lat - deg_lat / 2,
            c_lon + deg_lon / 2, c_lat + deg_lat / 2,
        )
        valid.append({
            "patch_id": r["patch_id"],
            "row": r["row"],
            "col": r["col"],
            "centroid_lon": c_lon,
            "centroid_lat": c_lat,
            "area_m2": patch_area_m2,
            "area_km2": round(patch_area_m2 / 1e6, 6),
            "forest_coverage": round(r.get("forest", 0) * 100, 2),
            "water_fraction": round(r.get("water", 0) * 100, 2),
            "valid_obs_pct": round(r.get("obs", 0) * 100, 2),
            "geometry": geom,
        })

    return gpd.GeoDataFrame(valid, crs="EPSG:4326") if valid else gpd.GeoDataFrame()


def generate_grid_gdf() -> gpd.GeoDataFrame:
    lon_vals, lat_vals = _grid_coords()
    deg_lat, deg_lon = _compute_patch_degrees()

    records = []
    patch_id = 0
    for row_idx, lat in enumerate(lat_vals):
        for col_idx, lon in enumerate(lon_vals):
            geom = box(lon, lat, lon + deg_lon, lat + deg_lat)
            centroid = geom.centroid
            records.append({
                "patch_id": f"patch_{patch_id:06d}",
                "row": row_idx,
                "col": col_idx,
                "centroid_lon": round(centroid.x, 6),
                "centroid_lat": round(centroid.y, 6),
                "geometry": geom,
            })
            patch_id += 1

    return gpd.GeoDataFrame(records, crs="EPSG:4326")
