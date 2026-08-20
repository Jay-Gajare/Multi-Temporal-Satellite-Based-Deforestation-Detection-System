"""
PyTorch Dataset for Sentinel-2 deforestation patches.

Supports two backends:
  - GeoTIFF: reads from exports/patches/{patch_id}/month_{01..12}.tif (slow)
  - Memmap: reads from exports/cache/{strategy}_data.npy (fast, ~100× faster I/O)

Supports three temporal input strategies:
  - single_month: one month selected as single-band input
  - average: mean across all available months
  - temporal_stack: all months stacked as channels
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FEATURE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NBR", "NDMI"]


class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for binary deforestation classification.

    Backend selection:
      - use_cache=True (default if cache exists): reads from numpy memmap files
        in exports/cache/. Requires running `python preprocessing/build_cache.py` first.
      - use_cache=False: reads 12 individual GeoTIFF files per sample (slow).
    """

    def __init__(
        self,
        split_csv: Path,
        labels_csv: Path,
        patches_dir: Path,
        temporal_strategy: str = "temporal_stack",
        best_month: int = 7,
        transform: torch.nn.Module | None = None,
        use_cache: bool | None = None,
    ) -> None:
        split_df = pd.read_csv(split_csv)
        labels_df = pd.read_csv(labels_csv)
        labels_map = dict(zip(labels_df["patch_id"], labels_df["label"]))

        self.patches_dir = Path(patches_dir)
        self.temporal_strategy = temporal_strategy
        self.best_month = best_month
        self.transform = transform

        self.n_bands = 9
        self.n_months = 12
        self.patch_size = 64

        # Determine backend
        cache_dir = self.patches_dir.parent / "cache"
        cache_data_path = cache_dir / f"{temporal_strategy}_data.npy"
        cache_index_path = cache_dir / "patch_index.json"

        if use_cache is None:
            use_cache = cache_data_path.exists() and cache_index_path.exists()

        self.use_cache = use_cache
        self._memmap: np.memmap | None = None
        self.patch_ids: list[str] = []
        self.labels: list[int] = []
        self._sample_map: dict[int, int] = {}  # our_idx → memmap_row

        if use_cache:
            self._init_cache(split_df, labels_map, cache_dir)
        else:
            self._init_geotiff(split_df, labels_map)

    def _init_cache(self, split_df: pd.DataFrame, labels_map: dict, cache_dir: Path) -> None:
        """Initialize from prebuilt memmap cache."""
        cache_index_path = cache_dir / "patch_index.json"
        with open(cache_index_path) as f:
            index_entries = json.load(f)

        split_pids = set(split_df["patch_id"].values)
        entry_by_pid = {e["patch_id"]: e for e in index_entries}

        memmap_path = cache_dir / f"{self.temporal_strategy}_data.npy"
        self._memmap = np.lib.format.open_memmap(str(memmap_path), mode="r")

        our_idx = 0
        for _, row in split_df.iterrows():
            pid = row["patch_id"]
            if pid not in entry_by_pid:
                continue
            entry = entry_by_pid[pid]
            label = int(labels_map.get(pid, entry["label"]))
            self.patch_ids.append(pid)
            self.labels.append(label)
            self._sample_map[our_idx] = entry["row"]
            our_idx += 1

        logger.info(
            "Cache backend: %d samples, memmap shape=%s, strategy=%s",
            len(self.patch_ids), list(self._memmap.shape), self.temporal_strategy,
        )

    def _init_geotiff(self, split_df: pd.DataFrame, labels_map: dict) -> None:
        """Initialize from individual GeoTIFF files."""
        for _, row in split_df.iterrows():
            pid = row["patch_id"]
            label = int(labels_map[pid])
            patch_dir = self.patches_dir / pid
            if self._check_patch(patch_dir):
                self.patch_ids.append(pid)
                self.labels.append(label)

    @staticmethod
    def _check_patch(patch_dir: Path) -> bool:
        if not patch_dir.is_dir():
            return False
        for m in range(1, 13):
            fpath = patch_dir / f"month_{m:02d}.tif"
            if not fpath.exists() or fpath.stat().st_size < 1000:
                return False
        return True

    def _read_month_geotiff(self, patch_dir: Path, month: int) -> np.ndarray:
        """Read one month's GeoTIFF → (9, 64, 64) float32."""
        import rasterio
        fpath = patch_dir / f"month_{month:02d}.tif"
        with rasterio.open(str(fpath)) as src:
            return src.read().astype(np.float32)

    def _load_patch_geotiff(self, idx: int) -> np.ndarray:
        """Load from individual GeoTIFFs (original slow path)."""
        pid = self.patch_ids[idx]
        patch_dir = self.patches_dir / pid

        if self.temporal_strategy == "single_month":
            return self._read_month_geotiff(patch_dir, self.best_month)

        if self.temporal_strategy == "average":
            months_data = [self._read_month_geotiff(patch_dir, m) for m in range(1, 13)]
            stack = np.stack(months_data, axis=0)
            return stack.mean(axis=0)

        months_data = [self._read_month_geotiff(patch_dir, m) for m in range(1, 13)]
        return np.concatenate(months_data, axis=0)

    def _load_patch_cache(self, idx: int) -> np.ndarray:
        """Load from memmap (fast: single contiguous read)."""
        row = self._sample_map[idx]
        return np.array(self._memmap[row], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.use_cache:
            arr = self._load_patch_cache(idx)
        else:
            arr = self._load_patch_geotiff(idx)

        tensor = torch.from_numpy(arr)
        label = self.labels[idx]

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

    @property
    def in_channels(self) -> int:
        if self.temporal_strategy == "single_month":
            return self.n_bands
        if self.temporal_strategy == "average":
            return self.n_bands
        return self.n_bands * self.n_months  # 108

    def class_counts(self) -> tuple[int, int]:
        """Return (n_negative, n_positive)."""
        neg = sum(1 for l in self.labels if l == 0)
        pos = sum(1 for l in self.labels if l == 1)
        return neg, pos
