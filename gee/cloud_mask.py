"""
Cloud masking for Sentinel-2 using s2cloudless (COPERNICUS/S2_CLOUD_PROBABILITY).

WHY s2cloudless over QA60:
    - QA60 flags clouds at the product level using a coarse bit-flag scheme.
      It marks entire pixels as cloud/cirrus but gives no confidence score.
    - s2cloudless is a dedicated ML-based cloud detector trained on Sentinel-2
      data. It produces a per-pixel probability score (0-100), allowing a
      tunable threshold rather than a hard binary mask.
    - With a probability threshold we can trade off between cloud removal and
      valid-pixel retention — critical for tropical regions like the Amazon
      where thin cloud and haze are common.
    - S2_CLOUD_PROBABILITY is hosted on GEE (no local inference needed) and
      is collocated with every S2_SR_HARMONIZED scene by acquisition date.

Fallback: If S2_CLOUD_PROBABILITY is unavailable for a given scene, QA60 is
used as a safety net.
"""

from __future__ import annotations

import ee

from configs.settings import (
    CLOUD_PROB_THRESHOLD,
    S2_BANDS,
    S2_CLOUD_PROB_COLLECTION,
)


def _join_by_index(s2_col: ee.ImageCollection) -> ee.ImageCollection:
    """
    Inner-join S2_SR_HARMONIZED with S2_CLOUD_PROBABILITY on system:index.

    Each Sentinel-2 scene in SR has a matching entry in the cloud probability
    collection. The join appends the 'probability' band to every SR image.
    """
    cloud_col = ee.ImageCollection(S2_CLOUD_PROB_COLLECTION)

    join = ee.Join.inner()
    joined = join.apply(
        primary=s2_col,
        secondary=cloud_col,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def merge_bands(match: ee.Feature) -> ee.Image:
        primary = ee.Image(match.get("primary"))
        secondary = ee.Image(match.get("secondary"))
        return primary.addBands(secondary.select("probability"))

    return ee.ImageCollection(joined.map(merge_bands))


def mask_clouds_s2cloudless(img: ee.Image) -> ee.Image:
    """
    Apply s2cloudless mask to a single image that already has a 'probability' band.

    Pixels with cloud probability >= threshold are masked out.
    Bands are selected to S2_BANDS and scaled to [0, 1].
    """
    prob = img.select("probability")
    cloud_mask = prob.lt(CLOUD_PROB_THRESHOLD)
    return (
        img.updateMask(cloud_mask)
        .select(S2_BANDS)
        .divide(10_000)
        .copyProperties(img, ["system:time_start"])
    )


def mask_clouds_qa60(img: ee.Image) -> ee.Image:
    """
    Fallback: mask clouds using the QA60 band (bits 10 and 11).

    Used when S2_CLOUD_PROBABILITY is not available for a scene.
    """
    qa = img.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return (
        img.updateMask(mask)
        .select(S2_BANDS)
        .divide(10_000)
        .copyProperties(img, ["system:time_start"])
    )


def apply_cloud_mask(s2_col: ee.ImageCollection) -> ee.ImageCollection:
    """
    Full cloud masking pipeline:
        1. Join S2_SR_HARMONIZED with S2_CLOUD_PROBABILITY
        2. Apply s2cloudless threshold mask
        3. Select and scale spectral bands

    Returns an ImageCollection with one cloud-free image per original scene.
    """
    joined = _join_by_index(s2_col)
    return joined.map(mask_clouds_s2cloudless)
