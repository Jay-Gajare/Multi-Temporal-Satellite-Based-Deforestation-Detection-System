"""
Convert GeoTIFF patches to a single .npy file for fast training I/O.

Creates exports/cache/{strategy}_data.npy — a dense numpy array (N, C, H, W).
Memory: temporal_stack → ~678 MB for 5,001 samples (safe for 7.8 GB RAM).

Also writes exports/cache/patch_index.json with row→patch_id mapping.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_cache")

PROJECT = Path(__file__).resolve().parent.parent
EXPORTS = PROJECT / "exports"
PATCHES = EXPORTS / "patches"
CACHE = EXPORTS / "cache"

N_BANDS = 9
N_MONTHS = 12
PATCH_SIZE = 64


def read_geotiff(path: Path) -> np.ndarray:
    with rasterio.open(str(path)) as src:
        return src.read().astype(np.float32)


def build_cache(strategy: str = "temporal_stack") -> None:
    cache_dir = CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(EXPORTS / "patch_labels.csv")
    labels_map = dict(zip(labels_df["patch_id"], labels_df["label"]))

    all_splits = []
    for split in ["train", "val", "test"]:
        csv_path = EXPORTS / "splits" / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["_split"] = split
            all_splits.append(df)
    split_df = pd.concat(all_splits, ignore_index=True)

    valid_rows = []
    for _, row in split_df.iterrows():
        pid = row["patch_id"]
        patch_dir = PATCHES / pid
        if not patch_dir.is_dir():
            continue
        ok = True
        for m in range(1, N_MONTHS + 1):
            fpath = patch_dir / f"month_{m:02d}.tif"
            if not fpath.exists() or fpath.stat().st_size < 1000:
                ok = False
                break
        if ok:
            valid_rows.append(row)

    n = len(valid_rows)
    logger.info("Found %d valid patches", n)

    if strategy == "temporal_stack":
        c = N_BANDS * N_MONTHS
    else:
        c = N_BANDS

    data = np.zeros((n, c, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    logger.info("Allocated array: %s (%.1f MB)", data.shape, data.nbytes / 1e6)

    index_entries = []
    t0 = time.time()

    for i, row in enumerate(tqdm(valid_rows, desc="Converting")):
        pid = row["patch_id"]
        patch_dir = PATCHES / pid
        label = int(labels_map.get(pid, -1))
        split_name = row["_split"]

        if strategy == "temporal_stack":
            months = [read_geotiff(patch_dir / f"month_{m:02d}.tif") for m in range(1, N_MONTHS + 1)]
            data[i] = np.concatenate(months, axis=0)
        elif strategy == "average":
            months = [read_geotiff(patch_dir / f"month_{m:02d}.tif") for m in range(1, N_MONTHS + 1)]
            data[i] = np.stack(months, axis=0).mean(axis=0)
        elif strategy == "single_month":
            data[i] = read_geotiff(patch_dir / "month_07.tif")

        index_entries.append({"row": i, "patch_id": pid, "label": label, "split": split_name})

    elapsed = time.time() - t0
    logger.info("Conversion done in %.1fs (%.1f samples/s)", elapsed, n / elapsed)

    out_path = cache_dir / f"{strategy}_data.npy"
    np.save(str(out_path), data)
    saved_size = out_path.stat().st_size / 1e6
    logger.info("Saved %s (%.1f MB on disk)", out_path.name, saved_size)

    with open(cache_dir / "patch_index.json", "w") as f:
        json.dump(index_entries, f, indent=2)
    logger.info("Saved patch_index.json (%d entries)", len(index_entries))

    del data


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="temporal_stack")
    args = p.parse_args()
    build_cache(args.strategy)
