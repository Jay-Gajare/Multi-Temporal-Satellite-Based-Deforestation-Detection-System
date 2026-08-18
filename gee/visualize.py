"""
Visualization helpers for Sentinel-2 imagery and spectral indices.

Produces:
    - 2x2 RGB / index map panels
    - Histograms for NDVI, NBR, NDMI
"""

from __future__ import annotations

import ee
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from configs.settings import (
    FIGURES_DIR,
    INDEX_VALID_MAX,
    INDEX_VALID_MIN,
    VIS_GAMMA,
    VIS_MAX,
    VIS_MIN,
)


# ── Internal helpers ──────────────────────────────────────────────────

def _ee_image_to_rgb(
    image: ee.Image,
    bands: list[str],
    region: ee.Geometry,
    vis_params: dict,
) -> np.ndarray:
    """Download an EE image thumbnail as an RGB numpy array (512px, 60m)."""
    thumb = image.getThumbURL({
        "bands": bands,
        "min": vis_params.get("min", VIS_MIN),
        "max": vis_params.get("max", VIS_MAX),
        "gamma": vis_params.get("gamma", VIS_GAMMA),
        "dimensions": "512",
        "region": region,
        "format": "png",
    })
    import urllib.request
    from io import BytesIO
    from PIL import Image
    with urllib.request.urlopen(thumb) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img)


def _ee_image_to_single_band(
    image: ee.Image,
    band: str,
    region: ee.Geometry,
    vis_min: float,
    vis_max: float,
    palette: list[str],
) -> np.ndarray:
    """Download a single-band EE image as a colored RGB numpy array."""
    thumb = image.select(band).getThumbURL({
        "min": vis_min,
        "max": vis_max,
        "palette": palette,
        "dimensions": "512",
        "region": region,
        "format": "png",
    })
    import urllib.request
    from io import BytesIO
    from PIL import Image
    with urllib.request.urlopen(thumb) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img)


def _histogram_from_image(
    image: ee.Image,
    band: str,
    region: ee.Geometry,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a histogram of a single band over a region using GEE."""
    hist = image.select(band).reduceRegion(
        reducer=ee.Reducer.fixedHistogram(INDEX_VALID_MIN, INDEX_VALID_MAX, n_bins),
        geometry=region,
        scale=100,
        maxPixels=1e10,
    ).getInfo()

    data = hist.get(band, [])
    if not data:
        return np.array([]), np.array([])
    arr = np.array(data)
    return arr[:, 0], arr[:, 1]


# ── Index visualization palettes ──────────────────────────────────────

NDVI_PALETTE = [
    "#d73027", "#fc8d59", "#fee08b",
    "#d9ef8b", "#91cf60", "#1a9850",
]
NBR_PALETTE = [
    "#d73027", "#fc8d59", "#fee08b",
    "#d9ef8b", "#91cf60", "#0571b0",
]
NDMI_PALETTE = [
    "#d73027", "#fc8d59", "#fee08b",
    "#d9ef8b", "#91cf60", "#2166ac",
]


# ── Public API ────────────────────────────────────────────────────────

def make_index_maps(
    composite: ee.Image,
    region: ee.Geometry,
    month_label: str,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Generate a 2x2 panel:
        1. RGB composite (B4, B3, B2)
        2. NDVI map
        3. NBR map
        4. NDMI map
    """
    vis_rgb = {"min": VIS_MIN, "max": VIS_MAX, "gamma": VIS_GAMMA}

    arr_rgb = _ee_image_to_rgb(composite, ["B4", "B3", "B2"], region, vis_rgb)
    arr_ndvi = _ee_image_to_single_band(composite, "NDVI", region, -0.2, 0.9, NDVI_PALETTE)
    arr_nbr = _ee_image_to_single_band(composite, "NBR", region, -0.2, 0.9, NBR_PALETTE)
    arr_ndmi = _ee_image_to_single_band(composite, "NDMI", region, -0.2, 0.7, NDMI_PALETTE)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"Spectral Index Maps - {month_label}", fontsize=16, fontweight="bold", y=0.98)

    panels = [
        (axes[0, 0], arr_rgb, "RGB Composite (B4, B3, B2)", None),
        (axes[0, 1], arr_ndvi, "NDVI", NDVI_PALETTE),
        (axes[1, 0], arr_nbr, "NBR", NBR_PALETTE),
        (axes[1, 1], arr_ndmi, "NDMI", NDMI_PALETTE),
    ]

    for ax, arr, title, pal in panels:
        ax.imshow(arr)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def make_histograms(
    composite: ee.Image,
    region: ee.Geometry,
    month_label: str,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Generate histograms for NDVI, NBR, NDMI distributions.

    Each subplot shows the histogram with a vertical dashed line at 0.
    """
    index_bands = ["NDVI", "NBR", "NDMI"]
    colors = ["#1a9850", "#d73027", "#2166ac"]
    titles = [
        "NDVI Distribution",
        "NBR Distribution",
        "NDMI Distribution",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Index Distributions - {month_label}", fontsize=15, fontweight="bold")

    for ax, band, color, title in zip(axes, index_bands, colors, titles):
        bins, counts = _histogram_from_image(composite, band, region)
        if len(bins) > 0:
            ax.bar(bins, counts, width=(INDEX_VALID_MAX - INDEX_VALID_MIN) / len(bins),
                   color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(band, fontsize=11)
        ax.set_ylabel("Pixel Count", fontsize=11)
        ax.set_xlim(INDEX_VALID_MIN, INDEX_VALID_MAX)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def make_validation_table(
    validation_results: list[dict],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Generate a table figure showing per-month index statistics and pass/fail.
    """
    months = [f"{r['month']:02d}" for r in validation_results]
    index_bands = ["NDVI", "NBR", "NDMI"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_title("Feature Validation Summary", fontsize=14, fontweight="bold", pad=20)

    col_labels = ["Month"] + [f"{b} min" for b in index_bands] + [f"{b} max" for b in index_bands] + ["Status"]
    cell_text = []
    cell_colors = []

    for r in validation_results:
        row = [f"{r['month']:02d}"]
        row_colors = ["#f0f0f0"]
        all_ok = True
        for band in index_bands:
            v = r["validation"][band]
            row.append(f"{v['min_actual']:.3f}" if v['min_actual'] is not None else "N/A")
            row_colors.append("#d4edda" if v["in_range"] else "#f8d7da")
        for band in index_bands:
            v = r["validation"][band]
            row.append(f"{v['max_actual']:.3f}" if v['max_actual'] is not None else "N/A")
            row_colors.append("#d4edda" if v["in_range"] else "#f8d7da")
        status = "PASS" if r["passed"] else "FAIL"
        row.append(status)
        row_colors.append("#d4edda" if r["passed"] else "#f8d7da")
        cell_text.append(row)
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#dee2e6"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    if save_path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig
