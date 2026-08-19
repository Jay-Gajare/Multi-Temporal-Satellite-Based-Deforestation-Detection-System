"""
Step 4 - Compute patch metrics and filter invalid patches.

Loads the generated grid GeoJSON, computes forest coverage, water coverage,
and valid observation percentage using GEE, then filters invalid patches.

Run:
    python preprocessing/generate_grid.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import geopandas as gpd

from configs.settings import (
    EXPORT_DIR,
    PROJECT_ROOT,
    ROI_COORDINATES,
    SCALE,
    PATCH_SIZE,
)
from gee.grid import (
    BATCH_SIZE,
    DOWNLOAD_PAGE,
    HANSEN_ASSET,
    MIN_FOREST_PCT,
    MIN_OBS_PCT,
    MIN_WATER_PCT,
    SCORE_SCALE,
    _compute_patch_degrees,
    _roi_geom,
    build_obs_image,
    build_water_forest_images,
)

GEOJSON_PATH = EXPORT_DIR / "patch_grid.geojson"


def _gdf_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        bounds = geom.bounds
        ee_geom = ee.Geometry.Rectangle(list(bounds))
        features.append(ee.Feature(ee_geom, {
            "patch_id": row["patch_id"],
            "row": int(row["row"]),
            "col": int(row["col"]),
            "centroid_lon": row["centroid_lon"],
            "centroid_lat": row["centroid_lat"],
        }))
    return ee.FeatureCollection(features)


def _download_fc(fc: ee.FeatureCollection) -> list[dict]:
    total = fc.size().getInfo()
    if total == 0:
        return []

    all_feats = []
    remaining = fc
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


def main() -> None:
    viz_only = "--viz-only" in sys.argv

    print("=" * 60, flush=True)
    print("STEP 4: COMPUTE METRICS & FILTER PATCHES", flush=True)
    print("=" * 60, flush=True)

    ee.Initialize(project="deforestation-early-warning")
    t0 = time.time()

    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not viz_only:
        print(f"\nLoading {GEOJSON_PATH}...", flush=True)
        gdf = gpd.read_file(GEOJSON_PATH)
        print(f"  {len(gdf):,} patches loaded", flush=True)

        deg_lat, deg_lon = _compute_patch_degrees()
        n_total = len(gdf)

        # ── Pass 1: water + forest ─────────────────────────────────────
        print("\n[1/3] Computing water + forest coverage...", flush=True)
        wf_img, n_images = build_water_forest_images()
        n_img_val = n_images.getInfo()
        print(f"  S2 images (ROI, study period): {n_img_val}", flush=True)

        n_batches = max(1, -(-n_total // BATCH_SIZE))
        print(f"  Processing {n_total:,} patches in {n_batches} batches...", flush=True)

        wf_lookup = {}
        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_total)
            bt = time.time()

            chunk = gdf.iloc[start:end]
            fc = _gdf_to_ee_fc(chunk)
            scored = wf_img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=SCORE_SCALE,
            )
            batch_records = _download_fc(scored)
            for r in batch_records:
                wf_lookup[r["patch_id"]] = r

            print(f"  Batch {b+1}/{n_batches}: {len(batch_records)} scored ({time.time()-bt:.1f}s)", flush=True)

        # ── Pass 2: valid observations ─────────────────────────────────
        print("\n[2/3] Computing valid observation percentage...", flush=True)
        obs_img = build_obs_image()

        obs_batch_size = 500
        n_obs_batches = max(1, -(-n_total // obs_batch_size))
        print(f"  Processing in {n_obs_batches} batches (size={obs_batch_size})...", flush=True)

        all_scores = []
        for b in range(n_obs_batches):
            start = b * obs_batch_size
            end = min(start + obs_batch_size, n_total)
            bt = time.time()

            chunk = gdf.iloc[start:end]
            fc = _gdf_to_ee_fc(chunk)
            scored = obs_img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=SCORE_SCALE,
            )
            batch_records = _download_fc(scored)
            all_scores.extend(batch_records)

            print(f"  Batch {b+1}/{n_obs_batches}: {len(batch_records)} scored ({time.time()-bt:.1f}s)", flush=True)

        obs_lookup = {r["patch_id"]: r.get("obs", 0) for r in all_scores}

        # ── Merge + filter ─────────────────────────────────────────────
        print("\n[3/3] Merging metrics & filtering...", flush=True)

        patch_area_m2 = float((PATCH_SIZE * SCALE) ** 2)

        records = []
        for _, row in gdf.iterrows():
            pid = row["patch_id"]
            wf = wf_lookup.get(pid, {})
            obs_val = obs_lookup.get(pid, 0)

            water_frac = round(wf.get("water", 0) * 100, 2)
            forest_cov = round(wf.get("forest", 0) * 100, 2)
            obs_pct = round(obs_val * 100, 2)

            records.append({
                "patch_id": pid,
                "row": int(row["row"]),
                "col": int(row["col"]),
                "centroid_lon": row["centroid_lon"],
                "centroid_lat": row["centroid_lat"],
                "area_m2": patch_area_m2,
                "area_km2": round(patch_area_m2 / 1e6, 6),
                "forest_coverage": forest_cov,
                "water_fraction": water_frac,
                "valid_obs_pct": obs_pct,
                "geometry": row.geometry,
            })

        all_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

        n_before = len(all_gdf)

        water_fail = all_gdf["water_fraction"] >= MIN_WATER_PCT * 100
        forest_fail = all_gdf["forest_coverage"] < MIN_FOREST_PCT * 100
        obs_fail = all_gdf["valid_obs_pct"] < MIN_OBS_PCT * 100
        invalid = water_fail | forest_fail | obs_fail

        removed_gdf = all_gdf[invalid].copy()
        filtered_gdf = all_gdf[~invalid].copy()

        n_water = int(water_fail.sum())
        n_forest = int((~water_fail & forest_fail).sum())
        n_obs = int((~water_fail & ~forest_fail & obs_fail).sum())
        n_final = len(filtered_gdf)

        print(f"\n  Filtering results:")
        print(f"    Water removed (>= {MIN_WATER_PCT*100:.0f}%):     {n_water:,}")
        print(f"    Forest removed (< {MIN_FOREST_PCT*100:.0f}%):    {n_forest:,}")
        print(f"    Obs removed (< {MIN_OBS_PCT*100:.0f}%):        {n_obs:,}")
        print(f"    Total removed:          {n_before - n_final:,}")
        print(f"    Final valid patches:    {n_final:,}")

        # ── Save ───────────────────────────────────────────────────
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        filtered_path = str(EXPORT_DIR / "patch_grid_filtered.geojson")
        filtered_gdf.to_file(filtered_path, driver="GeoJSON")

        filtered_csv = str(EXPORT_DIR / "patch_grid_filtered.csv")
        filtered_gdf.drop(columns="geometry").to_csv(filtered_csv, index=False)

        all_path = str(EXPORT_DIR / "patch_grid_all_metrics.geojson")
        all_gdf.to_file(all_path, driver="GeoJSON")

        print(f"\n  Saved:")
        print(f"    All patches with metrics: {all_path}")
        print(f"    Filtered grid (valid):    {filtered_path}")
        print(f"    Filtered CSV:             {filtered_csv}")
    else:
        filtered_path = str(EXPORT_DIR / "patch_grid_filtered.geojson")
        all_path = str(EXPORT_DIR / "patch_grid_all_metrics.geojson")
        print(f"\n  Loading existing data...", flush=True)
        filtered_gdf = gpd.read_file(filtered_path)
        all_gdf = gpd.read_file(all_path)
        print(f"  All: {len(all_gdf):,}  Filtered: {len(filtered_gdf):,}", flush=True)

    # ── Visualizations ─────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon

    roi_poly = Polygon([(c[0], c[1]) for c in ROI_COORDINATES])
    roi_gdf = gpd.GeoDataFrame(geometry=[roi_poly], crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(14, 10))
    roi_gdf.boundary.plot(ax=ax, color="black", linewidth=2)
    ax.set_title("Region of Interest (Rondonia, Brazil)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "step4_roi.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure: ROI              -> {FIGURES_DIR / 'step4_roi.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax = axes[0]
    roi_gdf.boundary.plot(ax=ax, color="black", linewidth=2, label="ROI")
    all_gdf.boundary.plot(ax=ax, color="steelblue", linewidth=0.1, alpha=0.3)
    ax.set_title(f"Candidate Grid ({len(all_gdf):,} patches)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    ax = axes[1]
    roi_gdf.boundary.plot(ax=ax, color="black", linewidth=2, label="ROI")
    filtered_gdf.boundary.plot(ax=ax, color="forestgreen", linewidth=0.2, alpha=0.5)
    ax.set_title(f"Filtered Grid ({len(filtered_gdf):,} patches)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    fig.suptitle("Step 4: Grid Overlay", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "step4_grid_overlay.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure: Grid overlay     -> {FIGURES_DIR / 'step4_grid_overlay.png'}")

    fig, ax = plt.subplots(figsize=(14, 10))
    roi_gdf.boundary.plot(ax=ax, color="black", linewidth=2)
    filtered_gdf.boundary.plot(ax=ax, color="steelblue", linewidth=0.2, alpha=0.5)
    sample = filtered_gdf.sample(min(200, len(filtered_gdf)), random_state=42)
    for _, row in sample.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(
            row["patch_id"].replace("patch_", ""),
            (centroid.x, centroid.y),
            fontsize=4, ha="center", va="center",
            color="darkred", fontweight="bold",
        )
    ax.set_title("Patch ID Labels (random sample)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "step4_patch_ids.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure: Patch IDs        -> {FIGURES_DIR / 'step4_patch_ids.png'}")

    n = 25
    sample = filtered_gdf.sample(min(n, len(filtered_gdf)), random_state=42).reset_index(drop=True)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 16))
    fig.suptitle("Random Sample of 25 Patches", fontsize=15, fontweight="bold", y=0.99)
    axes_flat = axes.flatten() if nrows > 1 else axes
    for i, ax in enumerate(axes_flat):
        if i < len(sample):
            row_data = sample.iloc[i]
            geom = row_data.geometry
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.3, color="forestgreen", edgecolor="darkgreen", linewidth=1.5)
            fc_val = row_data.get("forest_coverage", 0)
            obs_val = row_data.get("valid_obs_pct", 0)
            ax.set_title(f"{row_data['patch_id']}\nfc={fc_val:.0f}% obs={obs_val:.0f}%", fontsize=8)
            ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(FIGURES_DIR / "step4_random_25_patches.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure: 25 patches       -> {FIGURES_DIR / 'step4_random_25_patches.png'}")

    elapsed = time.time() - t0
    print(f"\nDone ({elapsed:.1f}s)")
    print(f"  All patches with metrics: {all_path}")
    print(f"  Filtered grid (valid):    {filtered_path}")


if __name__ == "__main__":
    main()
