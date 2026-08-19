"""
Finalize dataset: balanced selection from exported patches.
"""
import json
import os
import random
import math
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio

from configs.settings import EXPORT_DIR, FEATURE_BANDS, PATCH_SIZE

PATCHES_DIR = EXPORT_DIR / "patches"
SELECTION_PATH = EXPORT_DIR / "selected_patches.json"
METADATA_PATH = EXPORT_DIR / "final_patch_metadata.csv"
TARGET = 5000
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("FINALIZE DATASET")
    print("=" * 60)

    # Load filtered grid
    print("\n[1/7] Loading filtered grid...")
    with open(EXPORT_DIR / "patch_grid_filtered.geojson") as f:
        gj = json.load(f)

    pid_map = {}
    for feat in gj["features"]:
        p = feat["properties"]
        pid_map[p["patch_id"]] = p
    print(f"  Grid patches: {len(pid_map):,}")

    # Scan exported patches
    print("\n[2/7] Scanning exported patches on disk...")
    exported = []
    incomplete_ids = []
    for d in sorted(os.listdir(PATCHES_DIR)):
        full = PATCHES_DIR / d
        if not full.is_dir():
            continue
        tifs = list(full.glob("*.tif"))
        if len(tifs) != 12:
            incomplete_ids.append(d)
            continue
        if d not in pid_map:
            continue
        p = pid_map[d].copy()
        p["patch_id"] = d
        empty = 0
        for m in range(1, 13):
            fpath = full / f"month_{m:02d}.tif"
            if fpath.exists() and fpath.stat().st_size < 3000:
                empty += 1
        p["empty_months"] = empty
        p["months_available"] = 12 - empty
        exported.append(p)

    print(f"  Complete patches: {len(exported):,}")
    print(f"  Incomplete patches: {len(incomplete_ids)}")
    if incomplete_ids:
        print(f"  Incomplete IDs: {', '.join(incomplete_ids)}")

    # Spatial stratification
    print(f"\n[3/7] Balanced selection (target={TARGET})...")
    min_row = min(p["row"] for p in exported)
    max_row = max(p["row"] for p in exported)
    n_bands = 8
    band_size = max(1, (max_row - min_row + 1) // n_bands)

    bands = {}
    for p in exported:
        band = min(p["row"] // band_size, n_bands - 1)
        if band not in bands:
            bands[band] = []
        bands[band].append(p)

    selected = []
    for b in sorted(bands.keys()):
        pool = bands[b]
        alloc = max(1, round(TARGET * len(pool) / len(exported)))

        class_low = [p for p in pool if p["forest_coverage"] < 80]
        class_high = [p for p in pool if p["forest_coverage"] >= 80]

        ratio_low = len(class_low) / len(pool) if pool else 0
        n_low = max(1, round(alloc * ratio_low)) if class_low else 0
        n_high = alloc - n_low

        pick_low = random.sample(class_low, min(n_low, len(class_low))) if class_low else []
        pick_high = random.sample(class_high, min(n_high, len(class_high))) if class_high else []
        band_selected = pick_low + pick_high
        selected.extend(band_selected)

        r_min = min(p["row"] for p in pool)
        r_max = max(p["row"] for p in pool)
        fc_mean = np.mean([p["forest_coverage"] for p in band_selected]) if band_selected else 0
        print(f"  Band {b} (row {r_min:3d}-{r_max:3d}): {len(band_selected):4d}/{len(pool):5d} (fc={fc_mean:.1f}%)")

    print(f"\n  Selected: {len(selected)}")

    # Verify every selected patch (fast: existence + size check)
    print(f"\n[4/7] Verifying {len(selected)} selected patches...")
    verified = []
    corrupt = []
    for p in selected:
        pid = p["patch_id"]
        patch_dir = PATCHES_DIR / pid
        ok = True
        for m in range(1, 13):
            fpath = patch_dir / f"month_{m:02d}.tif"
            if not fpath.exists() or fpath.stat().st_size < 1000:
                ok = False
                break
        if ok:
            verified.append(p)
        else:
            corrupt.append(pid)

    # Deep-verify 5% sample with rasterio
    sample_n = max(5, len(selected) // 20)
    sample = random.sample(selected, min(sample_n, len(selected)))
    deep_ok = 0
    deep_fail = 0
    for p in sample:
        pid = p["patch_id"]
        patch_dir = PATCHES_DIR / pid
        try:
            with rasterio.open(str(patch_dir / "month_01.tif")) as src:
                if src.count == len(FEATURE_BANDS) and src.height == PATCH_SIZE and src.width == PATCH_SIZE and src.dtypes[0] == "float32":
                    deep_ok += 1
                else:
                    deep_fail += 1
        except Exception:
            deep_fail += 1
    print(f"  Deep-verified {len(sample)} sample: {deep_ok} OK, {deep_fail} FAIL")

    print(f"  Verified OK: {len(verified)}")
    print(f"  Corrupt/rejected: {len(corrupt)}")
    if corrupt:
        print(f"  Rejected IDs: {', '.join(corrupt[:20])}")

    selected = verified
    print(f"\n  Final dataset: {len(selected)} patches")

    # Build metadata
    print(f"\n[5/7] Generating final_patch_metadata.csv...")
    selected_ids = set(p["patch_id"] for p in selected)
    all_ids = set(pid_map.keys())

    rows = []
    for pid in sorted(all_ids):
        if pid not in pid_map:
            continue
        p = pid_map[pid]
        is_exported = pid in set(e["patch_id"] for e in exported)
        is_selected = pid in selected_ids

        # Tile assignment
        tile_r = p["row"] // 8
        tile_c = p["col"] // 8
        tile_id = f"tile_{tile_r:03d}_{tile_c:03d}"

        # Determine status
        if is_selected:
            status = "selected"
        elif is_exported:
            status = "exported_not_selected"
        else:
            status = "not_exported"

        rows.append({
            "patch_id": pid,
            "latitude": round(p["centroid_lat"], 6),
            "longitude": round(p["centroid_lon"], 6),
            "row": int(p["row"]),
            "column": int(p["col"]),
            "tile_id": tile_id,
            "forest_percentage": round(p["forest_coverage"], 2),
            "water_percentage": round(p["water_fraction"], 4),
            "observation_count": round(p["valid_obs_pct"], 2),
            "months_available": p.get("months_available", 12),
            "status": status,
            "selected_for_training": is_selected,
        })

    fieldnames = [
        "patch_id", "latitude", "longitude", "row", "column", "tile_id",
        "forest_percentage", "water_percentage", "observation_count",
        "months_available", "status", "selected_for_training",
    ]
    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_selected = sum(1 for r in rows if r["selected_for_training"])
    n_exported = sum(1 for r in rows if r["status"] in ("selected", "exported_not_selected"))
    print(f"  Total rows: {len(rows):,}")
    print(f"  Exported: {n_exported:,}")
    print(f"  Selected: {n_selected:,}")
    print(f"  Saved: {METADATA_PATH}")

    # Save selection
    with open(SELECTION_PATH, "w") as f:
        json.dump({
            "target": TARGET,
            "selected_ids": sorted(selected_ids),
            "stats": {
                "total_selected": len(selected),
                "fc_mean": float(np.mean([p["forest_coverage"] for p in selected])),
                "fc_std": float(np.std([p["forest_coverage"] for p in selected])),
                "months_mean": float(np.mean([p["months_available"] for p in selected])),
                "row_min": min(p["row"] for p in selected),
                "row_max": max(p["row"] for p in selected),
                "lat_min": min(p["centroid_lat"] for p in selected),
                "lat_max": max(p["centroid_lat"] for p in selected),
                "lon_min": min(p["centroid_lon"] for p in selected),
                "lon_max": max(p["centroid_lon"] for p in selected),
            },
        }, f, indent=2)

    # Compute final stats
    print(f"\n[6/7] Computing final statistics...")
    sel_fc = [p["forest_coverage"] for p in selected]
    sel_months = [p["months_available"] for p in selected]
    sel_lats = [p["centroid_lat"] for p in selected]
    sel_lons = [p["centroid_lon"] for p in selected]

    stats = {
        "total_exported_complete": len(exported),
        "total_selected": len(selected),
        "rejected_corrupt": len(corrupt),
        "incomplete_on_disk": len(incomplete_ids),
        "not_exported": len(all_ids) - len(exported),
        "fc_mean": float(np.mean(sel_fc)),
        "fc_median": float(np.median(sel_fc)),
        "fc_std": float(np.std(sel_fc)),
        "fc_min": float(np.min(sel_fc)),
        "fc_max": float(np.max(sel_fc)),
        "fc_class_60_80": sum(1 for v in sel_fc if v < 80),
        "fc_class_80_100": sum(1 for v in sel_fc if v >= 80),
        "months_mean": float(np.mean(sel_months)),
        "months_min": int(np.min(sel_months)),
        "months_max": int(np.max(sel_months)),
        "lat_min": float(np.min(sel_lats)),
        "lat_max": float(np.max(sel_lats)),
        "lon_min": float(np.min(sel_lons)),
        "lon_max": float(np.max(sel_lons)),
        "row_min": min(p["row"] for p in selected),
        "row_max": max(p["row"] for p in selected),
        "monthly_availability": {},
        "storage_selected_mb": round(len(selected) * 1.06, 1),
        "storage_selected_gb": round(len(selected) * 1.06 / 1024, 2),
        "storage_full_grid_mb": round(len(all_ids) * 1.06, 1),
        "storage_full_grid_gb": round(len(all_ids) * 1.06 / 1024, 2),
        "storage_savings_mb": round((len(all_ids) - len(selected)) * 1.06, 1),
        "storage_savings_gb": round((len(all_ids) - len(selected)) * 1.06 / 1024, 2),
    }

    # Monthly availability for selected
    for m in range(1, 13):
        count = 0
        for p in selected:
            fpath = PATCHES_DIR / p["patch_id"] / f"month_{m:02d}.tif"
            if fpath.exists() and fpath.stat().st_size >= 3000:
                count += 1
        stats["monthly_availability"][f"month_{m:02d}"] = {
            "count": count,
            "pct": round(count / len(selected) * 100, 1),
        }

    with open(EXPORT_DIR / "final_dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved: exports/final_dataset_stats.json")

    # Summary
    print(f"\n[7/7] Summary")
    print("=" * 60)
    print(f"  Selected: {len(selected):,} patches")
    print(f"  Forest class 60-80%: {stats['fc_class_60_80']:,} ({stats['fc_class_60_80']/len(selected)*100:.1f}%)")
    print(f"  Forest class 80-100%: {stats['fc_class_80_100']:,} ({stats['fc_class_80_100']/len(selected)*100:.1f}%)")
    print(f"  Mean forest coverage: {stats['fc_mean']:.1f}%")
    print(f"  Mean months available: {stats['months_mean']:.1f}/12")
    print(f"  Storage: {stats['storage_selected_gb']:.1f} GB selected (saves {stats['storage_savings_gb']:.1f} GB vs full grid)")
    print(f"  Rows: {stats['row_min']}-{stats['row_max']}")
    print(f"  Latitude: {stats['lat_min']:.4f} to {stats['lat_max']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
