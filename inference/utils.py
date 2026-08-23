"""
Utility helpers: logging setup, device detection, I/O, project paths.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = PROJECT_ROOT / "exports"
PATCHES_DIR = EXPORTS_DIR / "patches"
SPLITS_DIR = EXPORTS_DIR / "splits"
LABELS_CSV = EXPORTS_DIR / "patch_labels.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"

# Sentinel-2 band names in export order
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NBR", "NDMI"]
N_BANDS = len(S2_BANDS)
N_MONTHS = 12
PATCH_SIZE = 64
EXPECTED_CHANNELS = N_BANDS * N_MONTHS  # 108

DEFAULT_THRESHOLD = 0.50
DEFAULT_DROPOUT = 0.3


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger and return the inference logger."""
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)
    return logging.getLogger("inference")


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logging.getLogger("inference").info("GPU detected: %s (%.1f GB VRAM)", name, vram)
    else:
        dev = torch.device("cpu")
        cores = psutil.cpu_count()
        logging.getLogger("inference").info("Using CPU (%d cores)", cores)
    return dev


def get_system_info() -> dict:
    """Collect system information for benchmarking."""
    info: dict = {
        "cpu_count": psutil.cpu_count(),
        "cpu_freq_mhz": None,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 1),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    freq = psutil.cpu_freq()
    if freq:
        info["cpu_freq_mhz"] = round(freq.max, 0)
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1)
    return info


def get_process_memory_gb() -> float:
    """Return current process RSS in GB."""
    return round(psutil.Process().memory_info().rss / (1024 ** 3), 3)


@contextmanager
def timer() -> Generator[dict, None, None]:
    """Context manager that records elapsed time in a dict.

    Usage:
        with timer() as t:
            do_work()
        print(t["elapsed"])  # seconds
    """
    result = {"elapsed": 0.0}
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - t0


def save_json(data: dict, path: Path) -> None:
    """Write dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
