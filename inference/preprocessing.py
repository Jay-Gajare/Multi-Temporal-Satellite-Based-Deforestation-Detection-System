"""
Preprocessing module: GeoTIFF I/O, spectral index computation, input validation.

Matches the exact pipeline used during training:
  - Raw float32 values (no additional normalization)
  - 9 bands per month: B2, B3, B4, B8, B11, B12, NDVI, NBR, NDMI
  - temporal_stack: 12 months × 9 bands = 108 channels
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import torch

from inference.utils import (
    N_BANDS, N_MONTHS, PATCH_SIZE, EXPECTED_CHANNELS,
    S2_BANDS,
)

logger = logging.getLogger("inference.preprocessing")


# ---------------------------------------------------------------------------
# Spectral index computation (must match GEE export exactly)
# ---------------------------------------------------------------------------

def compute_spectral_indices(bands_9: np.ndarray) -> np.ndarray:
    """Compute NDVI, NBR, NDMI from 9-band array and return 12-band array.

    Parameters
    ----------
    bands_9 : ndarray of shape (9, H, W)
        Bands in order: B2, B3, B4, B8, B11, B12, NDVI_old, NBR_old, NDMI_old.
        If indices are already computed (non-zero), they are kept as-is.

    Returns
    -------
    ndarray of shape (12, H, W)
        B2, B3, B4, B8, B11, B12, NDVI, NBR, NDMI
    """
    b2 = bands_9[0].astype(np.float32)
    b3 = bands_9[1].astype(np.float32)
    b4 = bands_9[2].astype(np.float32)
    b8 = bands_9[3].astype(np.float32)
    b11 = bands_9[4].astype(np.float32)
    b12 = bands_9[5].astype(np.float32)

    denom_ndvi = b8 + b4
    ndvi = np.where(denom_ndvi > 0, (b8 - b4) / denom_ndvi, 0.0).astype(np.float32)

    denom_nbr = b8 + b12
    nbr = np.where(denom_nbr > 0, (b8 - b12) / denom_nbr, 0.0).astype(np.float32)

    denom_ndmi = b8 + b11
    ndmi = np.where(denom_ndmi > 0, (b8 - b11) / denom_ndmi, 0.0).astype(np.float32)

    return np.stack([b2, b3, b4, b8, b11, b12, ndvi, nbr, ndmi], axis=0)


# ---------------------------------------------------------------------------
# GeoTIFF reading
# ---------------------------------------------------------------------------

def read_geotiff(path: Path) -> tuple[np.ndarray, dict]:
    """Read a GeoTIFF and return (data, metadata).

    Returns
    -------
    data : ndarray of shape (C, H, W), float32
    meta : dict with keys: crs, resolution, width, height, count, dtype
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")
    if path.stat().st_size < 100:
        raise ValueError(f"File too small ({path.stat().st_size} bytes), likely corrupted: {path}")

    with rasterio.open(str(path)) as src:
        data = src.read().astype(np.float32)
        meta = {
            "crs": str(src.crs),
            "resolution": src.res,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "bounds": src.bounds,
        }
    return data, meta


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class InputError(Exception):
    """Raised when input fails validation."""
    pass


def validate_geotiff(path: Path, expected_bands: int | None = None) -> dict:
    """Validate a single GeoTIFF file.

    Parameters
    ----------
    path : Path to the GeoTIFF
    expected_bands : Expected number of bands (None = skip band count check)

    Returns
    -------
    dict with validation results

    Raises
    ------
    InputError if validation fails
    """
    path = Path(path)
    if not path.exists():
        raise InputError(f"File not found: {path}")
    if path.stat().st_size < 100:
        raise InputError(f"File too small ({path.stat().st_size} bytes), likely corrupted: {path}")

    data, meta = read_geotiff(path)

    errors: list[str] = []

    # CRS check (warn but don't fail — some data may use different CRS)
    if meta["crs"] and "EPSG:4326" not in meta["crs"] and "WGS 84" not in meta["crs"]:
        logger.warning("CRS is %s (expected EPSG:4326) — proceeding anyway", meta["crs"])

    # Resolution check (warn)
    res_x, res_y = meta["resolution"]
    if res_x > 0.1 or res_y > 0.1:
        logger.warning("Resolution %.4f × %.4f degrees — expected ~0.00027 (30m)", res_x, res_y)

    # Band count
    if expected_bands is not None and meta["count"] != expected_bands:
        errors.append(f"Expected {expected_bands} bands, got {meta['count']}")

    # Dimensions
    if meta["width"] != PATCH_SIZE or meta["height"] != PATCH_SIZE:
        errors.append(f"Expected {PATCH_SIZE}×{PATCH_SIZE}, got {meta['width']}×{meta['height']}")

    # NaN/Inf check
    if np.any(np.isnan(data)):
        errors.append("Contains NaN values")
    if np.any(np.isinf(data)):
        errors.append("Contains Inf values")

    if errors:
        raise InputError(f"Validation failed for {path.name}: {'; '.join(errors)}")

    return {
        "path": str(path),
        "valid": True,
        "bands": meta["count"],
        "width": meta["width"],
        "height": meta["height"],
        "crs": meta["crs"],
        "dtype": meta["dtype"],
    }


def validate_monthly_folder(folder: Path) -> dict:
    """Validate a folder containing 12 monthly GeoTIFFs.

    Expected naming: month_01.tif ... month_12.tif
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise InputError(f"Not a directory: {folder}")

    results: list[dict] = []
    missing: list[int] = []
    corrupted: list[int] = []

    for m in range(1, N_MONTHS + 1):
        fname = f"month_{m:02d}.tif"
        fpath = folder / fname
        if not fpath.exists():
            missing.append(m)
            continue
        try:
            v = validate_geotiff(fpath, expected_bands=N_BANDS)
            results.append(v)
        except InputError as e:
            corrupted.append(m)
            results.append({"path": str(fpath), "valid": False, "error": str(e)})

    if missing:
        raise InputError(
            f"Missing months {missing} in {folder.name}. "
            f"Expected {N_MONTHS} files: month_01.tif .. month_12.tif"
        )
    if corrupted:
        raise InputError(f"Corrupted month files {corrupted} in {folder.name}")

    return {"folder": str(folder), "valid": True, "n_months": len(results)}


def validate_patch_dir(patch_dir: Path) -> dict:
    """Validate an exported patch directory."""
    return validate_monthly_folder(patch_dir)


# ---------------------------------------------------------------------------
# Temporal stacking (must match training pipeline exactly)
# ---------------------------------------------------------------------------

def load_temporal_stack_from_monthly(folder: Path) -> np.ndarray:
    """Load 12 monthly GeoTIFFs and stack into (108, 64, 64) array.

    Each month file must have exactly 9 bands (B2, B3, B4, B8, B11, B12, NDVI, NBR, NDMI).

    Returns
    -------
    ndarray of shape (108, 64, 64), float32
    """
    folder = Path(folder)
    months_data: list[np.ndarray] = []

    for m in range(1, N_MONTHS + 1):
        fpath = folder / f"month_{m:02d}.tif"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing month {m:02d}: {fpath}")

        data, meta = read_geotiff(fpath)

        # Handle band count
        if data.shape[0] == N_BANDS:
            # Already has spectral indices — use as-is
            months_data.append(data)
        elif data.shape[0] == 6:
            # Only raw bands — compute indices
            logger.info("Month %02d: 6 bands, computing spectral indices", m)
            months_data.append(compute_spectral_indices(data))
        else:
            raise ValueError(
                f"Month {m:02d}: expected {N_BANDS} or 6 bands, got {data.shape[0]}"
            )

    stack = np.concatenate(months_data, axis=0).astype(np.float32)
    logger.info("Temporal stack: shape=%s, range=[%.3f, %.3f]", stack.shape, stack.min(), stack.max())
    return stack


def load_temporal_stack_from_geotiff(path: Path) -> np.ndarray:
    """Load a single GeoTIFF with all 108 channels already stacked.

    Returns
    -------
    ndarray of shape (108, 64, 64), float32
    """
    data, meta = read_geotiff(path)

    if data.shape[0] == EXPECTED_CHANNELS:
        return data.astype(np.float32)

    if data.shape[0] == N_BANDS:
        # Single month — zero-pad remaining months
        logger.warning(
            "Single month GeoTIFF (%d channels) — padding %d months with zeros",
            data.shape[0], N_MONTHS - 1,
        )
        pad = np.zeros((N_BANDS * (N_MONTHS - 1), data.shape[1], data.shape[2]), dtype=np.float32)
        return np.concatenate([data, pad], axis=0)

    raise ValueError(
        f"Expected {EXPECTED_CHANNELS} or {N_BANDS} bands, got {data.shape[0]}"
    )


def load_temporal_stack_from_patch(patch_dir: Path) -> np.ndarray:
    """Load from an exported patch directory (month_01.tif .. month_12.tif).

    Returns
    -------
    ndarray of shape (108, 64, 64), float32
    """
    return load_temporal_stack_from_monthly(patch_dir)


# ---------------------------------------------------------------------------
# Tensor conversion (matches training exactly)
# ---------------------------------------------------------------------------

def to_tensor(stack: np.ndarray) -> torch.Tensor:
    """Convert numpy temporal stack to PyTorch tensor.

    No normalization — matches the training pipeline which feeds raw float32.

    Parameters
    ----------
    stack : ndarray of shape (108, 64, 64)

    Returns
    -------
    Tensor of shape (1, 108, 64, 64) — batch dimension added
    """
    return torch.from_numpy(stack).unsqueeze(0)


def detect_input_type(path: Path) -> str:
    """Detect whether path is a single GeoTIFF, monthly folder, or patch directory.

    Returns
    -------
    One of: "geotiff", "monthly_folder", "patch_dir"
    """
    path = Path(path)

    if path.is_file() and path.suffix.lower() == ".tif":
        return "geotiff"

    if path.is_dir():
        # Check for month_XX.tif files
        has_monthly = all((path / f"month_{m:02d}.tif").exists() for m in range(1, 13))
        if has_monthly:
            return "patch_dir"

        # Check for any .tif files (could be a folder of monthly files)
        tif_files = list(path.glob("*.tif"))
        if len(tif_files) > 0:
            return "monthly_folder"

    raise InputError(
        f"Cannot determine input type for: {path}. "
        f"Expected a .tif file, a directory with month_01.tif..month_12.tif, "
        f"or a patch directory."
    )


def load_input(path: Path) -> np.ndarray:
    """Unified input loader: detects type and returns (108, 64, 64) temporal stack.

    Parameters
    ----------
    path : Path to GeoTIFF, monthly folder, or patch directory

    Returns
    -------
    ndarray of shape (108, 64, 64), float32
    """
    input_type = detect_input_type(path)
    logger.info("Detected input type: %s — %s", input_type, path)

    if input_type == "geotiff":
        return load_temporal_stack_from_geotiff(path)
    elif input_type == "patch_dir":
        return load_temporal_stack_from_patch(path)
    else:
        return load_temporal_stack_from_monthly(path)
