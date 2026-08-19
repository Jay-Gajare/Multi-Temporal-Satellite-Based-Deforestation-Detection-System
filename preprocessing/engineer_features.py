"""
Step 3 — Feature engineering on monthly composites.

Computes NDVI, NBR, NDMI on every monthly composite, validates ranges,
generates map and histogram visualizations, and compares dry/wet seasons.

Run:
    python preprocessing/engineer_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
from configs.settings import (
    DRY_SEASON_MONTHS,
    FEATURE_BANDS,
    FIGURES_DIR,
    PROJECT_ROOT,
    WET_SEASON_MONTHS,
)
from gee.cloud_mask import apply_cloud_mask
from gee.composites import monthly_composites
from gee.datasets import get_roi, get_viz_region, load_sentinel2_raw
from gee.validate import seasonal_comparison, validate_composite
from gee.visualize import (
    make_histograms,
    make_index_maps,
    make_validation_table,
)


def main() -> None:
    roi = get_roi()
    viz_region = get_viz_region()

    print("=" * 60)
    print("STEP 3: FEATURE ENGINEERING & VALIDATION")
    print("=" * 60)

    # ── 1. Rebuild composites (from Step 2 pipeline) ────────────────
    print("\n[1/7] Loading cloud-masked Sentinel-2 collection...")
    raw_col = load_sentinel2_raw()
    raw_count = raw_col.size().getInfo()
    masked_col = apply_cloud_mask(raw_col)
    masked_count = masked_col.size().getInfo()
    print(f"       Raw: {raw_count}  ->  Masked: {masked_count}")

    # ── 2. Generate monthly composites with indices ─────────────────
    print("\n[2/7] Generating monthly composites with spectral indices...")
    composites = monthly_composites(masked_col)
    success = [c for c in composites if c["composite"] is not None]
    print(f"       Composites generated: {len(success)}/12")
    print(f"       Bands per composite:  {FEATURE_BANDS}")

    # ── 3. Validate each composite ──────────────────────────────────
    print("\n[3/7] Validating index ranges per month...")
    validation_results = []
    all_passed = True

    for entry in composites:
        if entry["composite"] is None:
            print(f"       Month {entry['month']:2d}: SKIPPED (no images)")
            continue

        result = validate_composite(entry["composite"], entry["month"])
        validation_results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        ndvi_r = result["validation"]["NDVI"]
        nbr_r = result["validation"]["NBR"]
        ndmi_r = result["validation"]["NDMI"]

        ndvi_range = f"[{ndvi_r['min_actual']:.3f}, {ndvi_r['max_actual']:.3f}]"
        nbr_range = f"[{nbr_r['min_actual']:.3f}, {nbr_r['max_actual']:.3f}]"
        ndmi_range = f"[{ndmi_r['min_actual']:.3f}, {ndmi_r['max_actual']:.3f}]"

        print(f"       Month {entry['month']:2d}: NDVI={ndvi_range}  "
              f"NBR={nbr_range}  NDMI={ndmi_range}  [{status}]")

        if not result["passed"]:
            all_passed = False

    if all_passed:
        print("\n       >> ALL VALIDATIONS PASSED")
    else:
        print("\n       >> SOME VALIDATIONS FAILED — see details above")

    # ── 4. Seasonal comparison ──────────────────────────────────────
    print("\n[4/7] Dry vs wet season comparison...")
    seasonal = seasonal_comparison(composites)

    dry = seasonal["dry_season"]
    wet = seasonal["wet_season"]
    print(f"       Dry season ({len(DRY_SEASON_MONTHS)} months, "
          f"{dry['count']} composites):")
    for band, val in dry["means"].items():
        print(f"         {band}: {val:.4f}" if val else f"         {band}: N/A")

    print(f"       Wet season ({len(WET_SEASON_MONTHS)} months, "
          f"{wet['count']} composites):")
    for band, val in wet["means"].items():
        print(f"         {band}: {val:.4f}" if val else f"         {band}: N/A")

    # ── 5. Visualizations — pick a representative dry month ─────────
    print("\n[5/7] Generating index maps and histograms...")
    best_dry = max(
        [c for c in success if c["month"] in DRY_SEASON_MONTHS],
        key=lambda e: e["image_count"],
    )
    month_label = f"{best_dry['start'][:7]}"
    composite = best_dry["composite"]

    maps_path = str(FIGURES_DIR / "step3_index_maps.png")
    make_index_maps(composite, viz_region, month_label, save_path=maps_path)
    print(f"       Index maps saved:  {maps_path}")

    hist_path = str(FIGURES_DIR / "step3_histograms.png")
    make_histograms(composite, viz_region, month_label, save_path=hist_path)
    print(f"       Histograms saved:  {hist_path}")

    # ── 6. Validation table ─────────────────────────────────────────
    print("\n[6/7] Generating validation table...")
    table_path = str(FIGURES_DIR / "step3_validation_table.png")
    make_validation_table(validation_results, save_path=table_path)
    print(f"       Table saved:       {table_path}")

    # ── 7. Save JSON statistics ─────────────────────────────────────
    print("\n[7/7] Saving statistics...")
    stats = {
        "total_composites": len(success),
        "validation_passed": all_passed,
        "months_validated": len(validation_results),
        "monthly_stats": [],
        "seasonal_comparison": {
            "dry_season": {
                "months": DRY_SEASON_MONTHS,
                "count": dry["count"],
                "means": {k: round(v, 4) if v else None for k, v in dry["means"].items()},
            },
            "wet_season": {
                "months": WET_SEASON_MONTHS,
                "count": wet["count"],
                "means": {k: round(v, 4) if v else None for k, v in wet["means"].items()},
            },
        },
        "figures": {
            "index_maps": maps_path,
            "histograms": hist_path,
            "validation_table": table_path,
        },
    }

    for vr in validation_results:
        entry = {
            "month": vr["month"],
            "passed": vr["passed"],
            "stats": {band: {k: round(v, 4) if v is not None else None
                             for k, v in s.items()}
                      for band, s in vr["stats"].items()
                      if band in ["NDVI", "NBR", "NDMI"]},
            "validation": vr["validation"],
        }
        stats["monthly_stats"].append(entry)

    stats_path = str(PROJECT_ROOT / "reports" / "step3_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"       Stats saved: {stats_path}")

    # ── Final summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Composites    : {len(success)}/12")
    print(f"  Features      : {len(FEATURE_BANDS)} bands per composite")
    print(f"  Validation    : {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"  Index maps    : {maps_path}")
    print(f"  Histograms    : {hist_path}")
    print(f"  Valid. table  : {table_path}")

    return stats


if __name__ == "__main__":
    main()
