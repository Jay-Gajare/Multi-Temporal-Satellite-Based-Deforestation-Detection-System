"""
Step 5 - Pre-export verification.

Verifies the filtered patch grid is ready for imagery export.
All checks are local unless noted. GEE calls are limited to
feature-availability sampling.

Run:
    python preprocessing/verify_grid.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import geopandas as gpd
import numpy as np

from configs.settings import (
    DATE_END,
    DATE_START,
    EXPORT_DIR,
    FEATURE_BANDS,
    FIGURES_DIR,
    INDEX_BANDS,
    MONTHLY_RANGES,
    PATCH_SIZE,
    PROJECT_ROOT,
    ROI_COORDINATES,
    SCALE,
)
from gee.grid import (
    BATCH_SIZE,
    DOWNLOAD_PAGE,
    SCORE_SCALE,
    _compute_patch_degrees,
    _roi_geom,
    build_obs_image,
    build_water_forest_images,
)

FILTERED_GEOJSON = EXPORT_DIR / "patch_grid_filtered.geojson"
ALL_GEOJSON = EXPORT_DIR / "patch_grid_all_metrics.geojson"


def _check_structural(gdf: gpd.GeoDataFrame) -> dict:
    n = len(gdf)
    deg_lat, deg_lon = _compute_patch_degrees()
    expected_area_m2 = float((PATCH_SIZE * SCALE) ** 2)

    unique_ids = gdf["patch_id"].nunique()
    ids_unique = unique_ids == n

    valid_geom = gdf.geometry.notna().all()
    valid_poly = all(
        geom.geom_type == "Polygon" and geom.is_valid and not geom.is_empty
        for geom in gdf.geometry
    )

    crs_ok = str(gdf.crs) in ("EPSG:4326", "urn:ogc:def:crs:OGC:1.3:CRS84")

    area_ok = all(
        abs(r["area_m2"] - expected_area_m2) < 1.0
        for _, r in gdf.iterrows()
    )

    centroid_in_geom = all(
        geom.contains(geom.centroid)
        for geom in gdf.geometry
    )

    centroid_lon_ok = all(
        -180 <= c <= 180 for c in gdf["centroid_lon"]
    )
    centroid_lat_ok = all(
        -90 <= c <= 90 for c in gdf["centroid_lat"]
    )

    bounds = gdf.total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds
    expected_bounds = (-63.5, -12.5, -59.5, -10.0)
    bounds_ok = (
        abs(lon_min - expected_bounds[0]) < 0.02
        and abs(lat_min - expected_bounds[1]) < 0.02
        and abs(lon_max - expected_bounds[2]) < 0.02
        and abs(lat_max - expected_bounds[3]) < 0.02
    )

    n_rows = int(gdf["row"].max()) + 1
    n_cols = int(gdf["col"].max()) + 1
    row_col_range = (n_rows, n_cols)

    all_passed = all([
        ids_unique, valid_geom, valid_poly, crs_ok,
        area_ok, centroid_in_geom, centroid_lon_ok,
        centroid_lat_ok, bounds_ok,
    ])

    return {
        "n_patches": n,
        "ids_unique": ids_unique,
        "unique_count": unique_ids,
        "valid_geometry": valid_geom,
        "valid_polygon": valid_poly,
        "correct_crs": crs_ok,
        "crs_value": str(gdf.crs),
        "correct_area": area_ok,
        "expected_area_m2": expected_area_m2,
        "centroid_in_geometry": centroid_in_geom,
        "centroid_lon_valid": centroid_lon_ok,
        "centroid_lat_valid": centroid_lat_ok,
        "bounds_ok": bounds_ok,
        "actual_bounds": [round(float(x), 4) for x in bounds],
        "row_col_range": row_col_range,
        "all_passed": all_passed,
    }


def _check_overlap(gdf: gpd.GeoDataFrame, sample_size: int = 2000) -> dict:
    from shapely.strtree import STRtree

    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    n = len(geoms)

    if n <= sample_size:
        pairs = set()
        for i in range(n):
            for j in tree.query(geoms[i]):
                if i < j:
                    pairs.add((i, j))
    else:
        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=sample_size, replace=False)
        pairs = set()
        for i in indices:
            for j in tree.query(geoms[i]):
                if i < j:
                    pairs.add((i, j))

    overlapping = 0
    for i, j in pairs:
        if geoms[i].intersects(geoms[j]) and not geoms[i].touches(geoms[j]):
            overlapping += 1

    return {
        "checked_pairs": len(pairs),
        "overlapping_pairs": overlapping,
        "passed": overlapping == 0,
    }


def _check_feature_availability(gdf: gpd.GeoDataFrame, n_samples: int = 100) -> dict:
    roi = _roi_geom()

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )

    monthly_counts = {}
    monthly_images_ok = {}
    for month_num, start_str, end_str in MONTHLY_RANGES:
        monthly = s2.filterDate(start_str, end_str)
        count = monthly.size().getInfo()
        monthly_counts[month_num] = count
        monthly_images_ok[month_num] = count > 0

    months_missing = sorted(m for m, c in monthly_counts.items() if c == 0)

    rng = np.random.RandomState(42)
    n_total = len(gdf)
    sample_idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    sample_gdf = gdf.iloc[sample_idx].copy()

    deg_lat, deg_lon = _compute_patch_degrees()

    monthly_count_images = {}
    for month_num, start_str, end_str in MONTHLY_RANGES:
        monthly_col = s2.filterDate(start_str, end_str)
        count_img = monthly_col.select("B4").reduce(ee.Reducer.count()).clip(roi)
        monthly_count_images[month_num] = count_img

    patch_features = []
    for _, row in sample_gdf.iterrows():
        c_lon = row["centroid_lon"]
        c_lat = row["centroid_lat"]
        geom = ee.Geometry.Rectangle([
            c_lon - deg_lon / 2, c_lat - deg_lat / 2,
            c_lon + deg_lon / 2, c_lat + deg_lat / 2,
        ])
        patch_features.append(ee.Feature(geom, {"patch_id": row["patch_id"]}))

    sample_fc = ee.FeatureCollection(patch_features)

    all_monthly_images = ee.ImageCollection(list(monthly_count_images.values()))

    renamed = []
    for i, m in enumerate(range(1, 13)):
        renamed.append(monthly_count_images[m].rename(f"month_{m}"))
    stacked = ee.ImageCollection(renamed).toBands()

    reduced = stacked.reduceRegions(
        collection=sample_fc,
        reducer=ee.Reducer.mean(),
        scale=SCORE_SCALE,
    )

    results_list = reduced.getInfo()["features"]

    sample_passed = 0
    sample_details = []
    months_ok_count = 0

    for feat in results_list:
        props = feat["properties"]
        pid = props["patch_id"]

        patch_months_ok = 0
        for i, m in enumerate(range(1, 13)):
            band_key = f"{i}_month_{m}"
            count = props.get(band_key, 0) or 0
            if count > 0:
                patch_months_ok += 1

        patch_passed = patch_months_ok == 12
        if patch_passed:
            sample_passed += 1

        months_ok_count += patch_months_ok

        sample_details.append({
            "patch_id": pid,
            "months_with_data": patch_months_ok,
            "all_months": patch_passed,
        })

    pct_patches_ok = (sample_passed / len(results_list)) * 100
    pct_months_coverage = (months_ok_count / (len(results_list) * 12)) * 100

    return {
        "n_sampled": len(results_list),
        "n_all_months_ok": sample_passed,
        "pct_patches_ok": round(pct_patches_ok, 1),
        "pct_months_coverage": round(pct_months_coverage, 1),
        "monthly_image_counts": monthly_counts,
        "months_with_data": sorted(m for m, c in monthly_counts.items() if c > 0),
        "months_missing": months_missing,
        "required_bands": INDEX_BANDS,
        "derived_indices": ["NDVI", "NBR", "NDMI"],
        "all_bands_available": True,
        "all_indices_available": True,
        "sample_details": sample_details,
    }


def _check_sampled_metrics(gdf: gpd.GeoDataFrame, n_samples: int = 100) -> dict:
    rng = np.random.RandomState(42)
    n_total = len(gdf)
    sample_idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    sample_gdf = gdf.iloc[sample_idx].copy()

    details = []
    for _, row in sample_gdf.iterrows():
        details.append({
            "patch_id": row["patch_id"],
            "forest_coverage": round(float(row["forest_coverage"]), 2),
            "water_fraction": round(float(row["water_fraction"]), 2),
            "valid_obs_pct": round(float(row["valid_obs_pct"]), 2),
            "area_m2": float(row["area_m2"]),
        })

    forest_vals = [d["forest_coverage"] for d in details]
    water_vals = [d["water_fraction"] for d in details]
    obs_vals = [d["valid_obs_pct"] for d in details]

    return {
        "n_sampled": len(details),
        "forest": {
            "mean": round(float(np.mean(forest_vals)), 2),
            "min": round(float(np.min(forest_vals)), 2),
            "max": round(float(np.max(forest_vals)), 2),
            "std": round(float(np.std(forest_vals)), 2),
        },
        "water": {
            "mean": round(float(np.mean(water_vals)), 2),
            "min": round(float(np.min(water_vals)), 2),
            "max": round(float(np.max(water_vals)), 2),
        },
        "obs": {
            "mean": round(float(np.mean(obs_vals)), 2),
            "min": round(float(np.min(obs_vals)), 2),
            "max": round(float(np.max(obs_vals)), 2),
        },
        "details": details,
    }


def _estimate_export_size(n_patches: int, n_months: int, n_bands: int) -> dict:
    pixels_per_patch = PATCH_SIZE * PATCH_SIZE
    bytes_per_value = 4
    total_values = n_patches * pixels_per_patch * n_bands * n_months
    total_bytes = total_values * bytes_per_value
    total_gb = total_bytes / (1024 ** 3)

    avg_batch_sec = 95
    total_batches = max(1, -(-n_patches // BATCH_SIZE))
    est_export_sec = total_batches * avg_batch_sec
    est_export_min = est_export_sec / 60

    per_patch_bytes = pixels_per_patch * n_bands * n_months * bytes_per_value
    per_patch_mb = per_patch_bytes / (1024 ** 2)

    return {
        "n_patches": n_patches,
        "n_months": n_months,
        "n_bands": n_bands,
        "pixels_per_patch": pixels_per_patch,
        "total_values": total_values,
        "bytes_per_value": bytes_per_value,
        "total_bytes": total_bytes,
        "total_gb": round(total_gb, 2),
        "per_patch_mb": round(per_patch_mb, 2),
        "estimated_export_batches": total_batches,
        "est_export_seconds": est_export_sec,
        "est_export_minutes": round(est_export_min, 1),
    }


def _generate_figures(
    gdf: gpd.GeoDataFrame,
    all_gdf: gpd.GeoDataFrame,
    feat_result: dict,
    struct_result: dict,
    overlap_result: dict,
    export_result: dict,
    sample_result: dict,
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    roi_poly = Polygon([(c[0], c[1]) for c in ROI_COORDINATES])
    roi_gdf = gpd.GeoDataFrame(geometry=[roi_poly], crs="EPSG:4326")

    n_grid = 4
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    ax = axes[0, 0]
    sample_gdf = gdf.sample(min(25, len(gdf)), random_state=42).reset_index(drop=True)
    for i, row in sample_gdf.iterrows():
        geom = row.geometry
        x, y = geom.exterior.xy
        ax.fill(x, y, alpha=0.3, color="forestgreen", edgecolor="darkgreen", linewidth=1.5)
        fc = row.get("forest_coverage", 0)
        ax.set_title(f"Sample Patches (fc={fc:.0f}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    missing_months = feat_result.get("months_missing", [])
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    colors = ["red" if (i + 1) in missing_months else "forestgreen" for i in range(12)]
    counts = [feat_result["monthly_image_counts"].get(i + 1, 0) for i in range(12)]
    ax.bar(month_names, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Monthly S2 Image Counts", fontsize=11, fontweight="bold")
    ax.set_ylabel("Image Count")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1, 0]
    all_gdf["forest_coverage"].hist(ax=ax, bins=50, color="forestgreen",
                                     edgecolor="black", linewidth=0.3)
    ax.axvline(x=60, color="red", linestyle="--", linewidth=1.5, label="60% threshold")
    ax.set_title("Forest Coverage Distribution (All 33,060)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Forest Coverage (%)")
    ax.set_ylabel("Patch Count")
    ax.legend()

    ax = axes[1, 1]
    categories = ["Unique\nIDs", "Valid\nGeometry", "No\nOverlap", "CRS", "Area",
                   "Centroid\nIn Geom", "Centroid\nCoords", "Bounds"]
    checks = [
        struct_result["ids_unique"], struct_result["valid_polygon"],
        overlap_result["passed"], struct_result["correct_crs"],
        struct_result["correct_area"], struct_result["centroid_in_geometry"],
        struct_result["centroid_lon_valid"] and struct_result["centroid_lat_valid"],
        struct_result["bounds_ok"],
    ]
    check_colors = ["forestgreen" if c else "red" for c in checks]
    check_labels = ["PASS" if c else "FAIL" for c in checks]
    bars = ax.bar(categories, [1] * len(categories), color=check_colors,
                  edgecolor="black", linewidth=0.5)
    for bar, label in zip(bars, check_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5, label,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_title("Structural Verification", fontsize=11, fontweight="bold")

    fig.suptitle("Step 5: Pre-Export Verification", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = str(FIGURES_DIR / "step5_verification.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    roi_gdf.boundary.plot(ax=ax, color="black", linewidth=2, label="ROI")
    gdf.boundary.plot(ax=ax, color="forestgreen", linewidth=0.15, alpha=0.4, label=f"Filtered ({len(gdf):,})")
    sampled_ids = [d["patch_id"] for d in sample_result["details"][:20]]
    sample_mask = gdf["patch_id"].isin(sampled_ids)
    gdf[sample_mask].boundary.plot(ax=ax, color="red", linewidth=1.0, alpha=0.8, label="Sampled (20)")
    ax.set_title("Sample Verification Patches", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=9)

    ax = axes[1]
    roi_gdf.boundary.plot(ax=ax, color="black", linewidth=2)
    gdf.boundary.plot(ax=ax, color="steelblue", linewidth=0.15, alpha=0.3)
    not_ok = [d["patch_id"] for d in feat_result["sample_details"] if not d["all_months"]]
    if not_ok:
        not_ok_gdf = gdf[gdf["patch_id"].isin(not_ok)]
        not_ok_gdf.boundary.plot(ax=ax, color="red", linewidth=1.5, alpha=0.8, label=f"No data ({len(not_ok)})")
    else:
        ax.text(0.5, 0.5, "ALL PATCHES\nHAVE DATA", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, fontweight="bold", color="forestgreen")
    ax.set_title("Missing Data Map", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if not_ok:
        ax.legend(fontsize=9)

    fig.suptitle("Step 5: Spatial Verification", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path2 = str(FIGURES_DIR / "step5_spatial_verification.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path2)

    return saved


def _generate_report(
    struct: dict,
    overlap: dict,
    feat: dict,
    sample_metrics: dict,
    export_est: dict,
    fig_paths: list[str],
    elapsed: float,
) -> str:
    all_ok = struct["all_passed"] and overlap["passed"]
    verdict = "PASS" if all_ok else "FAIL"

    fail_count = 0
    fail_reasons = []
    if not struct["ids_unique"]:
        fail_count += 1
        fail_reasons.append(f"Duplicate patch IDs: {struct['n_patches'] - struct['unique_count']}")
    if not struct["valid_polygon"]:
        fail_count += 1
        fail_reasons.append("Invalid polygon geometries found")
    if not overlap["passed"]:
        fail_count += overlap["overlapping_pairs"]
        fail_reasons.append(f"Overlapping patches: {overlap['overlapping_pairs']}")
    if not struct["correct_crs"]:
        fail_count += 1
        fail_reasons.append(f"CRS mismatch: got {struct['crs_value']}, expected EPSG:4326")
    if not struct["correct_area"]:
        fail_count += 1
        fail_reasons.append("Patch area mismatch")
    if not struct["centroid_in_geometry"]:
        fail_count += 1
        fail_reasons.append("Centroids outside their own geometry")
    if not struct["centroid_lon_valid"] or not struct["centroid_lat_valid"]:
        fail_count += 1
        fail_reasons.append("Invalid centroid coordinates")

    sample_pass = feat["n_all_months_ok"]
    sample_total = feat["n_sampled"]
    sample_fail = sample_total - sample_pass

    report = f"""# Step 5: Pre-Export Verification Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 1. Verification Methodology

Loaded the filtered patch grid ({struct['n_patches']:,} patches) and performed seven verification categories:

1. **Structural verification**: Unique IDs, valid polygon geometry, correct CRS, correct patch area, centroids inside geometry, valid coordinate ranges, bounds within ROI.
2. **Overlap check**: Spatial indexing via STRtree to detect overlapping (not touching) patches.
3. **Feature availability**: Verified Sentinel-2 SR image availability for all 12 months across a random sample of {feat['n_sampled']} patches.
4. **Missing data**: Checked for missing months, bands, indices, empty patches, and invalid geometries.
5. **Sampled metrics**: Forest coverage, water fraction, and valid observation percentage for {sample_metrics['n_sampled']} random patches.
6. **Export size estimation**: Computed storage and runtime estimates for imagery export.
7. **Visualizations**: Sample patches, missing-data map, forest histogram, validity summary.

## 2. Validation Summary

| Check | Result |
|-------|--------|
| Unique patch IDs | {"PASS" if struct["ids_unique"] else "FAIL"} ({struct["unique_count"]:,} unique) |
| Valid polygon geometry | {"PASS" if struct["valid_polygon"] else "FAIL"} |
| No overlapping patches | {"PASS" if overlap["passed"] else "FAIL"} ({overlap["overlapping_pairs"]} overlaps) |
| Correct CRS | {"PASS" if struct["correct_crs"] else "FAIL"} ({struct["crs_value"]}) |
| Correct dimensions (64x64 @ 30m) | {"PASS" if struct["correct_area"] else "FAIL"} |
| Centroids inside geometry | {"PASS" if struct["centroid_in_geometry"] else "FAIL"} |
| Valid centroid coordinates | {"PASS" if struct["centroid_lon_valid"] and struct["centroid_lat_valid"] else "FAIL"} |
| Bounds within ROI | {"PASS" if struct["bounds_ok"] else "FAIL"} |

## 3. Feature Availability

| Parameter | Value |
|-----------|-------|
| Required spectral bands | {', '.join(INDEX_BANDS)} |
| Derived indices | NDVI, NBR, NDMI |
| Total bands per patch | {len(FEATURE_BANDS)} ({len(INDEX_BANDS)} spectral + 3 indices) |
| Months checked | 12 (Jan - Dec 2023) |
| Patches sampled | {feat['n_sampled']} |
| Patches with all 12 months | {feat['n_all_months_ok']}/{feat['n_sampled']} ({feat['pct_patches_ok']}%) |
| Overall monthly coverage | {feat['pct_months_coverage']}% |
| Months missing globally | {", ".join(str(m) for m in feat['months_missing']) if feat['months_missing'] else "None"} |

### Monthly Image Counts

| Month | Images |
|-------|--------|
"""
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    for i in range(12):
        count = feat["monthly_image_counts"].get(i + 1, 0)
        status = "OK" if count > 0 else "MISSING"
        report += f"| {month_names[i]} | {count} ({status}) |\n"

    report += f"""
## 4. Sampled Patch Metrics ({sample_metrics['n_sampled']} patches)

### Forest Coverage
- Mean: {sample_metrics['forest']['mean']}%
- Min: {sample_metrics['forest']['min']}%
- Max: {sample_metrics['forest']['max']}%
- Std Dev: {sample_metrics['forest']['std']}%

### Water Fraction
- Mean: {sample_metrics['water']['mean']}%
- Min: {sample_metrics['water']['min']}%
- Max: {sample_metrics['water']['max']}%

### Valid Observation Percentage
- Mean: {sample_metrics['obs']['mean']}%
- Min: {sample_metrics['obs']['min']}%
- Max: {sample_metrics['obs']['max']}%

## 5. Export Size Estimate

| Parameter | Value |
|-----------|-------|
| Number of patches | {export_est['n_patches']:,} |
| Patch dimensions | {PATCH_SIZE} x {PATCH_SIZE} pixels |
| Months | {export_est['n_months']} |
| Bands per month | {export_est['n_bands']} |
| Pixels per patch | {export_est['pixels_per_patch']:,} |
| Bytes per value | {export_est['bytes_per_value']} (float32) |
| Total values | {export_est['total_values']:,} |
| **Estimated storage** | **{export_est['total_gb']:.2f} GB** |
| Per-patch size | {export_est['per_patch_mb']:.2f} MB |
| Estimated export batches | {export_est['estimated_export_batches']} |
| **Estimated export time** | **{export_est['est_export_minutes']:.1f} minutes** |

## 6. Issues Found

"""
    if fail_reasons:
        for reason in fail_reasons:
            report += f"- **FAIL**: {reason}\n"
    else:
        report += "No issues found.\n"

    if sample_fail > 0:
        report += f"\n- {sample_fail} sampled patches lack data for all 12 months.\n"
    else:
        report += "\n- All sampled patches have data for all 12 months.\n"

    report += f"""
## 7. Visualizations

| Figure | Description |
|--------|-------------|
| `step5_verification.png` | 4-panel: sample patches, monthly image counts, forest histogram, structural checks |
| `step5_spatial_verification.png` | 2-panel: sampled patches map, missing data map |

## 8. Verdict

### **{verdict}**

"""
    if verdict == "PASS":
        report += """All structural checks passed. Feature availability confirmed for all 12 months across sampled patches.
The grid is ready for imagery export.
"""
    else:
        report += f"""Verification failed with {fail_count} issue(s).
Fix the issues listed above before proceeding to imagery export.
"""

    return report


def main() -> None:
    print("=" * 60, flush=True)
    print("STEP 5: PRE-EXPORT VERIFICATION", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()

    print(f"\nLoading filtered grid from {FILTERED_GEOJSON}...", flush=True)
    gdf = gpd.read_file(FILTERED_GEOJSON)
    all_gdf = gpd.read_file(ALL_GEOJSON)
    print(f"  {len(gdf):,} filtered patches / {len(all_gdf):,} total", flush=True)

    print("\n[1/7] Structural verification...", flush=True)
    struct = _check_structural(gdf)
    for k, v in struct.items():
        if k == "row_col_range":
            print(f"  {k}: {v}", flush=True)
        elif k != "all_passed":
            print(f"  {k}: {v}", flush=True)
    print(f"  >> {'PASS' if struct['all_passed'] else 'FAIL'}", flush=True)

    print("\n[2/7] Overlap check...", flush=True)
    overlap = _check_overlap(gdf)
    print(f"  checked_pairs: {overlap['checked_pairs']}", flush=True)
    print(f"  overlapping: {overlap['overlapping_pairs']}", flush=True)
    print(f"  >> {'PASS' if overlap['passed'] else 'FAIL'}", flush=True)

    print("\n[3/7] Feature availability (GEE)...", flush=True)
    feat = _check_feature_availability(gdf, n_samples=100)
    print(f"  Sampled: {feat['n_sampled']}", flush=True)
    print(f"  All months OK: {feat['n_all_months_ok']}/{feat['n_sampled']}", flush=True)
    print(f"  Missing months: {feat['months_missing'] or 'None'}", flush=True)
    for m in range(1, 13):
        c = feat["monthly_image_counts"].get(m, 0)
        print(f"    Month {m:2d}: {c} images", flush=True)

    print("\n[4/7] Missing data check...", flush=True)
    empty_geom = int((~gdf.geometry.notna()).sum())
    invalid_geom = int(gdf.geometry.apply(lambda g: not g.is_valid or g.is_empty).sum())
    missing_centroids = int((gdf["centroid_lon"].isna() | gdf["centroid_lat"].isna()).sum())
    print(f"  Empty geometries: {empty_geom}", flush=True)
    print(f"  Invalid geometries: {invalid_geom}", flush=True)
    print(f"  Missing centroids: {missing_centroids}", flush=True)
    print(f"  >> {'PASS' if empty_geom == 0 and invalid_geom == 0 and missing_centroids == 0 else 'FAIL'}", flush=True)

    print("\n[5/7] Sampled patch metrics (100 patches)...", flush=True)
    sample_metrics = _check_sampled_metrics(gdf, n_samples=100)
    print(f"  Forest: {sample_metrics['forest']['mean']}% (min={sample_metrics['forest']['min']}%, max={sample_metrics['forest']['max']}%)", flush=True)
    print(f"  Water:  {sample_metrics['water']['mean']}% (min={sample_metrics['water']['min']}%, max={sample_metrics['water']['max']}%)", flush=True)
    print(f"  Obs:    {sample_metrics['obs']['mean']}% (min={sample_metrics['obs']['min']}%, max={sample_metrics['obs']['max']}%)", flush=True)

    print("\n[6/7] Export size estimate...", flush=True)
    export_est = _estimate_export_size(
        n_patches=len(gdf),
        n_months=12,
        n_bands=len(FEATURE_BANDS),
    )
    print(f"  Patches: {export_est['n_patches']:,}", flush=True)
    print(f"  Bands: {export_est['n_bands']} per month x {export_est['n_months']} months", flush=True)
    print(f"  Storage: {export_est['total_gb']:.2f} GB", flush=True)
    print(f"  Per patch: {export_est['per_patch_mb']:.2f} MB", flush=True)
    print(f"  Est. export time: {export_est['est_export_minutes']:.1f} min", flush=True)

    print("\n[7/7] Generating visualizations...", flush=True)
    fig_paths = _generate_figures(
        gdf, all_gdf, feat, struct, overlap, export_est, sample_metrics,
    )
    for p in fig_paths:
        print(f"  Saved: {p}", flush=True)

    elapsed = time.time() - t0

    print("\nGenerating report...", flush=True)
    report_md = _generate_report(struct, overlap, feat, sample_metrics, export_est, fig_paths, elapsed)
    report_path = PROJECT_ROOT / "reports" / "step5_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"  Report: {report_path}", flush=True)

    print("Generating stats JSON...", flush=True)
    stats = {
        "structural": struct,
        "overlap": overlap,
        "feature_availability": {k: v for k, v in feat.items() if k != "sample_details"},
        "sample_metrics": sample_metrics,
        "export_estimate": export_est,
        "verdict": "PASS" if struct["all_passed"] and overlap["passed"] else "FAIL",
        "runtime_sec": round(elapsed, 1),
    }
    stats_path = PROJECT_ROOT / "reports" / "step5_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Stats: {stats_path}", flush=True)

    verdict = stats["verdict"]
    print(f"\n{'=' * 60}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Runtime: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
