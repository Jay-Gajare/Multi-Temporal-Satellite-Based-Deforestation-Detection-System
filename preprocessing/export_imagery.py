"""
Step 6 — Imagery export from verified spatial grid.

Extracts monthly Sentinel-2 composites (9 bands) as GeoTIFFs for each
valid patch using tile-based computePixels extraction.

Resume strategy:
  1. Scan exports/patches/ on disk — verify each patch (12 files, readable,
     correct dims/bands/dtype)
  2. Delete incomplete patch directories (partial exports from prior runs)
  3. Skip patches that are already complete
  4. Checkpoint after every tile

Run:
    python preprocessing/export_imagery.py
    python preprocessing/export_imagery.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import numpy as np
import pandas as pd
import rasterio

from configs.settings import (
    EXPORT_DIR,
    FEATURE_BANDS,
    PATCH_SIZE,
    PROJECT_ROOT,
    SCALE,
)
from gee.cloud_mask import apply_cloud_mask
from gee.datasets import load_sentinel2_raw
from gee.export_helpers import (
    TILE_PATCHES,
    TILE_SIZE,
    build_monthly_composites,
    crop_tile_to_patches,
    extract_tile,
    patch_bounds,
    save_empty_geotiff,
    save_patch_geotiff,
)

PATCHES_DIR = EXPORT_DIR / "patches"
CHECKPOINT_PATH = EXPORT_DIR / "checkpoint.json"
METADATA_PATH = EXPORT_DIR / "patch_metadata.csv"
RETRY_LIMIT = 3
RETRY_DELAY = 5


# ---------------------------------------------------------------------------
# Disk scanning / patch validation
# ---------------------------------------------------------------------------

def _verify_patch_on_disk(patch_dir: Path) -> bool:
    """Return True if patch_dir contains 12 valid, readable GeoTIFFs."""
    if not patch_dir.is_dir():
        return False
    for m in range(1, 13):
        fpath = patch_dir / f"month_{m:02d}.tif"
        if not fpath.exists() or fpath.stat().st_size < 1000:
            return False
        try:
            with rasterio.open(str(fpath)) as src:
                if src.count != len(FEATURE_BANDS):
                    return False
                if src.height != PATCH_SIZE or src.width != PATCH_SIZE:
                    return False
                if src.dtypes[0] != "float32":
                    return False
        except Exception:
            return False
    return True


def scan_existing_patches() -> tuple[set[str], list[str]]:
    """
    Scan exports/patches/ and classify every directory.

    Returns (completed_patch_ids, incomplete_patch_ids).
    """
    completed: set[str] = set()
    incomplete: list[str] = []

    if not PATCHES_DIR.exists():
        return completed, incomplete

    for d in sorted(PATCHES_DIR.iterdir()):
        if not d.is_dir():
            continue
        pid = d.name
        if _verify_patch_on_disk(d):
            completed.add(pid)
        else:
            incomplete.append(pid)

    return completed, incomplete


def delete_incomplete_patches(incomplete: list[str]) -> int:
    """Delete incomplete patch directories. Returns count deleted."""
    deleted = 0
    for pid in incomplete:
        d = PATCHES_DIR / pid
        if d.exists():
            shutil.rmtree(d)
            deleted += 1
    return deleted


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {
        "completed_patches": [],
        "failed_patches": [],
        "last_tile_idx": -1,
        "started_at": None,
        "last_updated": None,
    }


def save_checkpoint(ckpt: dict) -> None:
    ckpt["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f)


# ---------------------------------------------------------------------------
# Grid / tile helpers
# ---------------------------------------------------------------------------

def load_grid() -> pd.DataFrame:
    import geopandas as gpd
    return gpd.read_file(EXPORT_DIR / "patch_grid_filtered.geojson")


def compute_deg_per_pixel() -> tuple[float, float]:
    deg_lat = (PATCH_SIZE * SCALE) / 111_320.0
    mid_lat = 11.25
    deg_lon = (PATCH_SIZE * SCALE) / (111_320.0 * math.cos(math.radians(mid_lat)))
    return deg_lon, deg_lat


def build_tiles(gdf: pd.DataFrame) -> list[dict]:
    """Group patches into 8x8 tiles based on grid row/col."""
    deg_lon, deg_lat = compute_deg_per_pixel()

    tiles: dict[tuple[int, int], dict] = {}
    for _, row in gdf.iterrows():
        r = int(row["row"])
        c = int(row["col"])
        tile_r, tile_c = r // TILE_PATCHES, c // TILE_PATCHES
        tile_key = (tile_r, tile_c)
        row_in_tile = r % TILE_PATCHES
        col_in_tile = c % TILE_PATCHES

        if tile_key not in tiles:
            tile_west = float(row["centroid_lon"]) - col_in_tile * deg_lon - deg_lon / 2
            tile_north = float(row["centroid_lat"]) + row_in_tile * deg_lat + deg_lat / 2
            tiles[tile_key] = {"tile_key": tile_key, "patches": []}

        tiles[tile_key]["patches"].append((
            row_in_tile, col_in_tile,
            row["patch_id"],
            float(row["centroid_lon"]),
            float(row["centroid_lat"]),
        ))

    tile_list = sorted(tiles.values(), key=lambda t: t["tile_key"])

    for tile in tile_list:
        patches = tile["patches"]
        all_lons = [p[3] for p in patches]
        all_lats = [p[4] for p in patches]
        min_col = min(p[1] for p in patches)
        max_col = max(p[1] for p in patches)
        min_row = min(p[0] for p in patches)
        max_row = max(p[0] for p in patches)

        center_lon = (min(all_lons) + max(all_lons)) / 2
        center_lat = (min(all_lats) + max(all_lats)) / 2
        half_width = (max_col - min_col + 1) * deg_lon / 2 + deg_lon / 2
        half_height = (max_row - min_row + 1) * deg_lat / 2 + deg_lat / 2

        tile["west"] = center_lon - half_width
        tile["east"] = center_lon + half_width
        tile["north"] = center_lat + half_height
        tile["south"] = center_lat - half_height
        tile["tile_id"] = f"tile_{tile['tile_key'][0]:03d}_{tile['tile_key'][1]:03d}"

    return tile_list


def _patch_bounds_from_tile(
    tile: dict, row_in_tile: int, col_in_tile: int,
) -> tuple[float, float, float, float]:
    deg_lon, deg_lat = compute_deg_per_pixel()
    pw = tile["west"] + col_in_tile * deg_lon
    pe = pw + deg_lon
    pn = tile["north"] - row_in_tile * deg_lat
    ps = pn - deg_lat
    return pw, ps, pe, pn


# ---------------------------------------------------------------------------
# Tile export
# ---------------------------------------------------------------------------

def export_tile(
    tile: dict,
    composites: list[dict],
    completed_set: set[str],
) -> tuple[str, int, list[str]]:
    """
    Export all months for all patches in a tile that aren't already done.

    Returns (tile_id, n_newly_completed, failed_patch_ids).
    """
    needed = [
        (r, c, pid)
        for r, c, pid, _, _ in tile["patches"]
        if pid not in completed_set
    ]

    if not needed:
        return tile["tile_id"], 0, []

    all_ok_patches: set[str] = set()
    all_failed: list[str] = []

    for month_data in composites:
        month_num = month_data["month"]

        if month_data["composite"] is None:
            for _, _, patch_id in needed:
                patch_dir = PATCHES_DIR / patch_id
                patch_dir.mkdir(parents=True, exist_ok=True)
                month_path = patch_dir / f"month_{month_num:02d}.tif"
                if not month_path.exists():
                    pw, ps, pe, pn = _patch_bounds_from_tile(
                        tile, *[x for x in tile["patches"] if x[2] == patch_id][0][:2]
                    )
                    save_empty_geotiff(str(month_path), pw, ps, pe, pn)
            continue

        tile_arr = None
        for attempt in range(RETRY_LIMIT):
            try:
                tile_arr = extract_tile(
                    month_data["composite"],
                    tile["west"], tile["south"],
                    tile["east"], tile["north"],
                )
                break
            except Exception as e:
                if attempt < RETRY_LIMIT - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    for _, _, pid in needed:
                        all_failed.append(pid)
                    return tile["tile_id"], 0, all_failed

        cropped = crop_tile_to_patches(
            tile_arr,
            tile["west"], tile["south"],
            tile["east"], tile["north"],
            [(r, c, pid) for r, c, pid in needed],
        )

        for patch_id, patch_arr, (pw, ps, pe, pn) in cropped:
            patch_dir = PATCHES_DIR / patch_id
            patch_dir.mkdir(parents=True, exist_ok=True)
            month_path = patch_dir / f"month_{month_num:02d}.tif"
            if not month_path.exists() or month_path.stat().st_size < 1000:
                save_patch_geotiff(patch_arr, str(month_path), pw, ps, pe, pn)
            all_ok_patches.add(patch_id)

    newly_completed = [
        pid for _, _, pid in needed
        if pid in all_ok_patches and _verify_patch_on_disk(PATCHES_DIR / pid)
    ]
    failed = [
        pid for _, _, pid in needed
        if pid not in all_ok_patches or not _verify_patch_on_disk(PATCHES_DIR / pid)
    ]

    return tile["tile_id"], len(newly_completed), failed


# ---------------------------------------------------------------------------
# Metadata / verification
# ---------------------------------------------------------------------------

def generate_metadata(gdf: pd.DataFrame) -> None:
    rows = []
    for _, row in gdf.iterrows():
        pid = row["patch_id"]
        patch_dir = PATCHES_DIR / pid
        if patch_dir.exists() and _verify_patch_on_disk(patch_dir):
            status = "exported"
        elif patch_dir.exists():
            status = "partial"
        else:
            status = "pending"
        rows.append({
            "patch_id": pid,
            "row": int(row["row"]),
            "column": int(row["col"]),
            "latitude": row["centroid_lat"],
            "longitude": row["centroid_lon"],
            "forest_coverage": row.get("forest_coverage", None),
            "water_fraction": row.get("water_fraction", None),
            "valid_obs_pct": row.get("valid_obs_pct", None),
            "export_status": status,
        })

    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def verify_exports(gdf: pd.DataFrame) -> dict:
    total = len(gdf)
    exported = 0
    failed = 0
    total_size = 0
    band_counts: set[int] = set()
    dtypes: set[str] = set()
    crs_set: set[str] = set()
    shapes: set[tuple[int, int]] = set()
    empty_count = 0
    corrupt_count = 0

    for _, row in gdf.iterrows():
        pid = row["patch_id"]
        patch_dir = PATCHES_DIR / pid
        if not patch_dir.exists() or not _verify_patch_on_disk(patch_dir):
            failed += 1
            continue

        patch_ok = True
        for m in range(1, 13):
            fpath = patch_dir / f"month_{m:02d}.tif"
            try:
                with rasterio.open(str(fpath)) as src:
                    if src.count != len(FEATURE_BANDS) or src.height != PATCH_SIZE or src.width != PATCH_SIZE:
                        patch_ok = False
                        break
                    band_counts.add(src.count)
                    dtypes.add(src.dtypes[0])
                    crs_set.add(str(src.crs))
                    shapes.add((src.height, src.width))
                    total_size += fpath.stat().st_size
                    data = src.read(1)
                    if np.all(data == 0):
                        empty_count += 1
            except Exception:
                corrupt_count += 1
                patch_ok = False
                break

        if patch_ok:
            exported += 1
        else:
            failed += 1

    return {
        "total_patches": total,
        "exported": exported,
        "failed": failed,
        "empty_patches": empty_count,
        "corrupt_files": corrupt_count,
        "total_size_bytes": total_size,
        "total_size_gb": round(total_size / (1024**3), 2),
        "band_counts": list(band_counts),
        "dtypes": list(dtypes),
        "crs_values": list(crs_set),
        "spatial_shapes": list(shapes),
        "all_correct_bands": band_counts <= {len(FEATURE_BANDS)},
        "all_correct_dtype": dtypes == {"float32"},
        "all_correct_crs": "EPSG:4326" in crs_set or len(crs_set) == 0,
        "all_correct_shape": shapes <= {(PATCH_SIZE, PATCH_SIZE)} or len(shapes) == 0,
    }


def generate_report(
    gdf: pd.DataFrame, verify: dict, runtime: float,
    n_tiles_total: int, n_tiles_done: int, avg_speed: float, ckpt: dict,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    failed_patches = ckpt.get("failed_patches", [])

    passed = (
        verify["exported"] > 0
        and verify["all_correct_bands"]
        and verify["all_correct_dtype"]
        and verify["all_correct_crs"]
        and verify["all_correct_shape"]
        and verify["corrupt_files"] == 0
    )

    return f"""# Step 6: Imagery Export Report

Generated: {now}

## 1. Export Configuration

| Parameter | Value |
|-----------|-------|
| Bands per patch | {len(FEATURE_BANDS)} ({', '.join(FEATURE_BANDS)}) |
| Patch dimensions | {PATCH_SIZE} x {PATCH_SIZE} pixels |
| Resolution | {SCALE} m |
| Datatype | Float32 |
| Temporal range | 12 months (Jan-Dec 2023) |
| Tile size | {TILE_PATCHES}x{TILE_PATCHES} patches ({TILE_SIZE}x{TILE_SIZE} pixels) |
| Retry limit | {RETRY_LIMIT} |
| Storage location | `{PATCHES_DIR}` |

## 2. Export Summary

| Metric | Value |
|--------|-------|
| Total valid patches | {verify['total_patches']:,} |
| Successfully exported | {verify['exported']:,} ({verify['exported']/max(verify['total_patches'],1)*100:.1f}%) |
| Failed exports | {verify['failed']:,} |
| Tiles processed | {n_tiles_done}/{n_tiles_total} |
| Average export speed | {avg_speed:.1f} patches/min |
| Total dataset size | {verify['total_size_gb']:.2f} GB |
| Per-patch avg file size | {verify['total_size_bytes']/max(verify['exported']*12,1)/1024:.1f} KB |
| Runtime | {runtime:.1f} min ({runtime/60:.1f} hrs) |

## 3. Quality Verification

| Check | Result |
|-------|--------|
| Correct band count ({len(FEATURE_BANDS)}) | {'PASS' if verify['all_correct_bands'] else 'FAIL'} |
| Correct dtype (Float32) | {'PASS' if verify['all_correct_dtype'] else 'FAIL'} |
| Correct CRS (EPSG:4326) | {'PASS' if verify['all_correct_crs'] else 'FAIL'} |
| Correct dimensions ({PATCH_SIZE}x{PATCH_SIZE}) | {'PASS' if verify['all_correct_shape'] else 'FAIL'} |
| No corrupted GeoTIFFs | {'PASS' if verify['corrupt_files'] == 0 else 'FAIL'} ({verify['corrupt_files']} found) |
| Empty (zero-filled) patches | {verify['empty_patches']} |

## 4. File Structure

```
exports/patches/
  {{patch_id}}/
    month_01.tif  (9 bands, Float32, {PATCH_SIZE}x{PATCH_SIZE})
    month_02.tif
    ...
    month_12.tif
```

## 5. Failed Patches

{f"Count: {len(failed_patches)}" if failed_patches else "None"}

## 6. Verdict

### {'**PASS**' if passed else '**FAIL**'}

{'All checks passed.' if passed else f'Issues: {verify["failed"]} failed, {verify["corrupt_files"]} corrupt.'}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: Export imagery")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 6: IMAGERY EXPORT (tile-based)")
    print("=" * 60)

    gdf = load_grid()
    print(f"\nLoaded {len(gdf):,} filtered patches")

    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Verify-only mode
    # ---------------------------------------------------------------
    if args.verify_only:
        print("\nVerifying existing exports...")
        verify = verify_exports(gdf)
        _print_verify(verify)
        report = generate_report(gdf, verify, 0, 0, 0, 0, {"failed_patches": []})
        (PROJECT_ROOT / "reports" / "step6_report.md").write_text(report)
        stats = {
            "export": {"exported": verify["exported"], "total": verify["total_patches"]},
            "verdict": "PASS" if verify["exported"] > 0 else "FAIL",
        }
        (PROJECT_ROOT / "reports" / "step6_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"\n  Report: reports/step6_report.md")
        return

    ee.Initialize(project="deforestation-early-warning")

    # ---------------------------------------------------------------
    # Phase 1: Build composites
    # ---------------------------------------------------------------
    t0 = time.time()
    print("\n[1/6] Building monthly composites on GEE...")
    raw_col = load_sentinel2_raw()
    masked_col = apply_cloud_mask(raw_col)
    composites = build_monthly_composites(masked_col)
    success = [c for c in composites if c["composite"] is not None]
    print(f"  Composites: {len(success)}/12 months ({time.time()-t0:.1f}s)")
    for c in composites:
        print(f"    Month {c['month']:2d}: {c['image_count']:4d} images"
              + ("" if c["composite"] else " [NO DATA]"))

    # ---------------------------------------------------------------
    # Phase 2: Build tile grid
    # ---------------------------------------------------------------
    print("\n[2/6] Building tile grid...")
    tiles = build_tiles(gdf)
    print(f"  Tiles: {len(tiles)} (each {TILE_PATCHES}x{TILE_PATCHES} = {TILE_SIZE}x{TILE_SIZE} px)")
    total_patches_in_tiles = sum(len(t["patches"]) for t in tiles)
    print(f"  Patches in tiles: {total_patches_in_tiles:,}")

    # ---------------------------------------------------------------
    # Phase 3: Scan disk, clean up, determine what to export
    # ---------------------------------------------------------------
    print("\n[3/6] Scanning existing exports on disk...")
    completed_set, incomplete_ids = scan_existing_patches()
    print(f"  Complete patches on disk:   {len(completed_set):,}")
    print(f"  Incomplete patches on disk: {len(incomplete_ids):,}")

    if incomplete_ids:
        print(f"\n  Deleting {len(incomplete_ids)} incomplete patch directories...")
        deleted = delete_incomplete_patches(incomplete_ids)
        print(f"  Deleted {deleted} directories")

    remaining = total_patches_in_tiles - len(completed_set)
    print(f"\n  Patches already done:  {len(completed_set):,}")
    print(f"  Patches remaining:     {remaining:,}")

    # ---------------------------------------------------------------
    # Phase 4: Export tiles
    # ---------------------------------------------------------------
    print("\n[4/6] Exporting tiles...")
    tiles_todo = [
        t for t in tiles
        if any(pid not in completed_set for _, _, pid, _, _ in t["patches"])
    ]
    n_tiles = len(tiles_todo)

    ckpt = {
        "completed_patches": list(completed_set),
        "failed_patches": [],
        "last_tile_idx": -1,
        "started_at": datetime.now().isoformat(),
        "last_updated": None,
    }
    save_checkpoint(ckpt)

    if remaining > 0:
        t_start = time.time()
        total_ok = 0
        total_fail = 0
        all_failed: list[str] = []

        for i, tile in enumerate(tiles_todo):
            tile_id, n_ok, failed = export_tile(tile, composites, completed_set)
            total_ok += n_ok
            total_fail += len(failed)
            all_failed.extend(failed)

            for _, _, pid, _, _ in tile["patches"]:
                if _verify_patch_on_disk(PATCHES_DIR / pid):
                    completed_set.add(pid)

            ckpt["completed_patches"] = list(completed_set)
            ckpt["failed_patches"] = list(set(all_failed))
            ckpt["last_tile_idx"] = i
            save_checkpoint(ckpt)

            elapsed = time.time() - t_start
            speed = total_ok / (elapsed / 60) if elapsed > 0 else 0
            eta = (remaining - total_ok) / speed if speed > 0 else 0
            pct_done = (i + 1) / n_tiles * 100
            print(
                f"  [{i+1}/{n_tiles}] {pct_done:.1f}% ({tile_id}): "
                f"+{n_ok} ok, {len(failed)} fail | "
                f"Cumul: {total_ok:,} ok, {total_fail} fail | "
                f"{speed:.1f} patches/min, ETA {eta:.0f} min"
            )

        print(f"\n  Export done: {total_ok:,} new patches in {time.time()-t_start:.0f}s")
    else:
        print("\n  All patches already complete — nothing to export.")

    # ---------------------------------------------------------------
    # Phase 5: Generate metadata
    # ---------------------------------------------------------------
    print("\n[5/6] Generating metadata...")
    generate_metadata(gdf)
    print(f"  Saved: {METADATA_PATH}")

    # ---------------------------------------------------------------
    # Phase 6: Final verification + report
    # ---------------------------------------------------------------
    print("\n[6/6] Verifying exports...")
    verify = verify_exports(gdf)
    _print_verify(verify)

    runtime = (time.time() - t0) / 60
    avg_speed = verify["exported"] / runtime if runtime > 0 else 0

    report = generate_report(gdf, verify, runtime, len(tiles), len(tiles_todo), avg_speed, ckpt)
    report_path = PROJECT_ROOT / "reports" / "step6_report.md"
    report_path.write_text(report)

    stats = {
        "export": {
            "total_patches": verify["total_patches"],
            "exported": verify["exported"],
            "failed": verify["failed"],
            "total_size_gb": verify["total_size_gb"],
            "total_tiles": len(tiles),
            "tiles_processed": len(tiles_todo),
            "avg_speed_patches_per_min": round(avg_speed, 1),
            "runtime_minutes": round(runtime, 1),
        },
        "verification": {
            "correct_bands": verify["all_correct_bands"],
            "correct_dtype": verify["all_correct_dtype"],
            "correct_crs": verify["all_correct_crs"],
            "correct_shape": verify["all_correct_shape"],
            "corrupt_files": verify["corrupt_files"],
            "empty_patches": verify["empty_patches"],
        },
        "config": {
            "bands": FEATURE_BANDS,
            "patch_size": PATCH_SIZE,
            "scale_m": SCALE,
            "dtype": "float32",
            "tile_patches": TILE_PATCHES,
            "tile_size_px": TILE_SIZE,
        },
        "verdict": "PASS" if verify["exported"] > 0 and verify["corrupt_files"] == 0 else "FAIL",
    }
    stats_path = PROJECT_ROOT / "reports" / "step6_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Exported : {verify['exported']:,}/{verify['total_patches']:,}")
    print(f"  Failed   : {verify['failed']:,}")
    print(f"  Size     : {verify['total_size_gb']:.2f} GB")
    print(f"  Speed    : {avg_speed:.1f} patches/min")
    print(f"  Runtime  : {runtime:.1f} min")
    print(f"  Verdict  : {stats['verdict']}")
    print(f"  Report   : {report_path}")
    print("=" * 60)


def _print_verify(verify: dict) -> None:
    print(f"  Exported: {verify['exported']:,}/{verify['total_patches']:,}")
    print(f"  Size: {verify['total_size_gb']:.2f} GB")
    print(f"  Bands: {verify['all_correct_bands']}")
    print(f"  Dtype: {verify['all_correct_dtype']}")
    print(f"  CRS: {verify['all_correct_crs']}")
    print(f"  Shape: {verify['all_correct_shape']}")
    print(f"  Corrupt: {verify['corrupt_files']}")


if __name__ == "__main__":
    main()
