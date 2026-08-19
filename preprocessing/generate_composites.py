"""
Step 2 — Cloud masking, monthly composites, visualisation, and statistics.

Run:
    python preprocessing/generate_composites.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
from configs.settings import (
    CLOUD_PROB_THRESHOLD,
    FIGURES_DIR,
    MONTHLY_RANGES,
    PROJECT_ROOT,
    S2_BANDS,
    YEAR,
)
from gee.cloud_mask import apply_cloud_mask
from gee.composites import monthly_composites
from gee.datasets import get_roi, get_viz_region, load_sentinel2_raw
from gee.visualize import make_figure


def main() -> None:
    roi = get_roi()
    viz_region = get_viz_region()

    # ── 1. Load raw collection ──────────────────────────────────────
    print("=" * 60)
    print("STEP 2: CLOUD MASKING & MONTHLY COMPOSITES")
    print("=" * 60)

    raw_col = load_sentinel2_raw()
    raw_count = raw_col.size().getInfo()
    print(f"\n[1/5] Raw S2 images (pre-mask): {raw_count}")

    # ── 2. Apply s2cloudless cloud masking ──────────────────────────
    masked_col = apply_cloud_mask(raw_col)
    masked_count = masked_col.size().getInfo()
    removed = raw_count - masked_count
    pct_removed = (removed / raw_count * 100) if raw_count > 0 else 0

    print(f"[2/5] Cloud-masked images:  {masked_count}")
    print(f"       Images removed:       {removed} ({pct_removed:.1f}%)")
    print(f"       Cloud prob threshold: < {CLOUD_PROB_THRESHOLD}")

    # ── 3. Generate monthly composites ──────────────────────────────
    composites = monthly_composites(masked_col)
    success = [c for c in composites if c["composite"] is not None]
    failed = [c for c in composites if c["composite"] is None]

    print(f"\n[3/5] Monthly composites generated: {len(success)}/12")
    if failed:
        failed_months = [c["month"] for c in failed]
        print(f"       FAILED months: {failed_months}")

    # ── 4. Per-month statistics ─────────────────────────────────────
    month_stats = []
    for entry in composites:
        m = entry["month"]
        label = entry["start"][:7]  # "2023-01"
        n = entry["image_count"]
        month_stats.append({"month": m, "label": label, "images": n})
        print(f"       {label}: {n:4d} images")

    # ── 5. Visualisation — pick a month with most images ────────────
    print(f"\n[4/5] Generating visualisation...")
    best = max(success, key=lambda e: e["image_count"])
    best_month = best["month"]
    month_label = f"{best['start'][:7]}"

    # Get individual images for this month (unmasked for "original" panel)
    raw_month = raw_col.filterDate(best["start"], best["end"])
    masked_month = masked_col.filterDate(best["start"], best["end"])

    # Pick the scene with lowest cloud % as the showcase
    raw_list = raw_month.sort("CLOUDY_PIXEL_PERCENTAGE").toList(1)
    original_img = ee.Image(raw_list.get(0))

    # For the masked panel, use the first available masked image
    masked_list = masked_month.sort("CLOUDY_PIXEL_PERCENTAGE").toList(1)
    masked_img = ee.Image(masked_list.get(0))

    composite_img = best["composite"]

    fig_path = str(FIGURES_DIR / "step2_processing_pipeline.png")
    fig = make_figure(
        original=original_img,
        masked=masked_img,
        composite=composite_img,
        region=viz_region,
        month_label=month_label,
        save_path=fig_path,
    )
    print(f"       Saved: {fig_path}")

    # ── 6. Save JSON statistics ─────────────────────────────────────
    stats = {
        "raw_image_count": raw_count,
        "masked_image_count": masked_count,
        "images_removed": removed,
        "cloud_pct_removed": round(pct_removed, 2),
        "cloud_prob_threshold": CLOUD_PROB_THRESHOLD,
        "months_generated": len(success),
        "months_failed": [c["month"] for c in failed],
        "month_breakdown": month_stats,
        "best_viz_month": month_label,
        "visualization_path": fig_path,
    }
    stats_path = str(PROJECT_ROOT / "reports" / "step2_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"       Stats saved: {stats_path}")

    print(f"\n[5/5] Done.")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Raw images     : {raw_count}")
    print(f"  Masked images  : {masked_count}")
    print(f"  Removed        : {removed} ({pct_removed:.1f}%)")
    print(f"  Composites     : {len(success)}/12 months")
    print(f"  Figure         : {fig_path}")
    print(f"  Stats JSON     : {stats_path}")

    return stats


if __name__ == "__main__":
    main()
