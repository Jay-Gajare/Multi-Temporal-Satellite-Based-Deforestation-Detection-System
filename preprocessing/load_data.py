"""
Step 1 – Verify GEE datasets and ROI.

Run:
    python preprocessing/load_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee                              # noqa: E402  – triggers gee/__init__.py
from configs.settings import (          # noqa: E402
    DATE_END,
    DATE_START,
    HANSEN_BANDS,
    PROJECT_ROOT,
    ROI_CENTER,
    ROI_COORDINATES,
    S2_BANDS,
    SCALE,
)
from gee.datasets import (             # noqa: E402
    get_roi,
    load_hansen,
    load_sentinel2,
)


def main() -> None:
    # ── 1. ROI ──────────────────────────────────────────────────────
    roi = get_roi()
    print("=" * 60)
    print("ROI")
    print("=" * 60)
    print(f"  Coordinates : {ROI_COORDINATES}")
    print(f"  Center      : {ROI_CENTER}")
    print(f"  ee.Geometry : {roi.getInfo()}")

    # ── 2. Hansen GFC ──────────────────────────────────────────────
    hansen = load_hansen()
    hansen_info = hansen.getInfo()
    hansen_bands = [b["id"] for b in hansen_info["bands"]]
    print("\n" + "=" * 60)
    print("HANSEN GLOBAL FOREST CHANGE")
    print("=" * 60)
    print(f"  Dataset     : UMD/hansen/global_forest_change_2025_v1_13")
    print(f"  Bands       : {hansen_bands}")
    print(f"  Band count  : {len(hansen_bands)}")

    # ── 3. Sentinel-2 ──────────────────────────────────────────────
    s2_col = load_sentinel2()
    s2_count = s2_col.size().getInfo()
    s2_first = ee.Image(s2_col.first()).getInfo()
    s2_bands_loaded = [b["id"] for b in s2_first["bands"]]

    print("\n" + "=" * 60)
    print("SENTINEL-2 SR HARMONISED")
    print("=" * 60)
    print(f"  Dataset     : COPERNICUS/S2_SR_HARMONIZED")
    print(f"  Date range  : {DATE_START} to {DATE_END}")
    print(f"  Images found: {s2_count}")
    print(f"  Bands loaded: {s2_bands_loaded}")
    print(f"  Scale       : {SCALE} m")
    if s2_count > 0:
        prop = s2_first["properties"]
        print(f"  First image date : {prop.get('system:time_start')}")
        print(f"  Cloud %          : {prop.get('CLOUDY_PIXEL_PERCENTAGE', 'N/A')}")

    # ── 4. Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  ROI area (approx): {(ROI_COORDINATES[2][0]-ROI_COORDINATES[0][0]):.1f} x {(ROI_COORDINATES[2][1]-ROI_COORDINATES[0][1]):.1f} degrees")
    print(f"  Hansen bands OK  : {set(HANSEN_BANDS).issubset(set(hansen_bands))}")
    print(f"  S2 bands OK      : {set(S2_BANDS).issubset(set(s2_bands_loaded))}")
    print(f"  S2 image count   : {s2_count}")
    print("\nAll checks passed – data pipeline ready for next step.")


if __name__ == "__main__":
    main()
