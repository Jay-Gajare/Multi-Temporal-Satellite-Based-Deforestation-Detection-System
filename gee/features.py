"""
Spectral index computation for Sentinel-2 composites.

All functions operate on ee.Image objects and return new ee.Image objects
with the computed index added as a band. No side effects.

Mathematical definitions:

    NDVI = (B8 - B4) / (B8 + B4)
        Normalized Difference Vegetation Index.
        Measures vegetation greenness / photosynthetic activity.
        Range: [-1, 1].  Dense forest ~ 0.7-0.9; bare soil ~ 0.1-0.2.

    NBR  = (B8 - B12) / (B8 + B12)
        Normalized Burn Ratio.
        Sensitive to moisture content and vegetation health.
        Range: [-1, 1].  Healthy forest ~ 0.5-0.8; burned/cleared ~ -0.2-0.1.

    NDMI = (B8 - B11) / (B8 + R11)
        Normalized Difference Moisture Index.
        Detects vegetation water content; sensitive to drought stress.
        Range: [-1, 1].  Well-watered ~ 0.3-0.6; dry/bare ~ -0.1-0.1.
"""

from __future__ import annotations

import ee


def add_ndvi(image: ee.Image) -> ee.Image:
    """
    Compute NDVI = (NIR - Red) / (NIR + Red).

    Uses Sentinel-2 bands B8 (NIR, 10m) and B4 (Red, 10m).
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def add_nbr(image: ee.Image) -> ee.Image:
    """
    Compute NBR = (NIR - SWIR2) / (NIR + SWIR2).

    Uses Sentinel-2 bands B8 (NIR, 10m) and B12 (SWIR2, 20m).
    """
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    return image.addBands(nbr)


def add_ndmi(image: ee.Image) -> ee.Image:
    """
    Compute NDMI = (NIR - SWIR1) / (NIR + SWIR1).

    Uses Sentinel-2 bands B8 (NIR, 10m) and B11 (SWIR1, 20m).
    """
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    return image.addBands(ndmi)


def add_all_indices(image: ee.Image) -> ee.Image:
    """
    Compute and attach NDVI, NBR, and NDMI to the image.

    Returns the original image with three additional bands:
        - NDVI: Normalized Difference Vegetation Index
        - NBR:  Normalized Burn Ratio
        - NDMI: Normalized Difference Moisture Index
    """
    image = add_ndvi(image)
    image = add_nbr(image)
    image = add_ndmi(image)
    return image
