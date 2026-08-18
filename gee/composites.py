"""
Monthly median composite generation from cloud-masked Sentinel-2 imagery.

Step 3 update: composites now include spectral indices (NDVI, NBR, NDMI)
computed on the median composite itself.
"""

from __future__ import annotations

import ee

from configs.settings import FEATURE_BANDS, INDEX_BANDS, MONTHLY_RANGES
from gee.features import add_all_indices


def monthly_composites(s2_masked: ee.ImageCollection) -> list[dict]:
    """
    Generate one median composite per month, with spectral indices attached.

    For each month:
        1. Filter the cloud-masked collection to that month
        2. Compute per-band median (on INDEX_BANDS: B2, B3, B4, B8, B11, B12)
        3. Compute NDVI, NBR, NDMI on the median composite
        4. If zero images exist, return None for that month

    Returns a list of dicts:
        [{"month": 1, "start": "2023-01-01", "end": "2023-01-31",
          "composite": ee.Image or None, "image_count": int}, ...]

    Each composite contains FEATURE_BANDS:
        B2, B3, B4, B8, B11, B12, NDVI, NBR, NDMI
    """
    results = []

    for month_num, start_str, end_str in MONTHLY_RANGES:
        monthly = s2_masked.filterDate(start_str, end_str)
        count = monthly.size().getInfo()

        if count == 0:
            results.append({
                "month": month_num,
                "start": start_str,
                "end": end_str,
                "composite": None,
                "image_count": 0,
            })
        else:
            # Median of spectral bands only
            median = monthly.median().select(INDEX_BANDS)

            # Compute indices on the composite (not per-image, to avoid
            # NaN propagation from partially-masked single scenes)
            composite = add_all_indices(median)
            composite = (
                composite
                .set("system:time_start", ee.Date(start_str).millis())
                .set("month", month_num)
                .set("image_count", count)
            )
            results.append({
                "month": month_num,
                "start": start_str,
                "end": end_str,
                "composite": composite,
                "image_count": count,
            })

    return results
