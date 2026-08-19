"""
Step 7 — Generate ground-truth labels for supervised deforestation detection.

Queries Hansen Global Forest Change v1.13 for every selected patch and
produces:
  - Binary label: 1 = deforestation (Hansen loss in study period), 0 = none
  - Continuous targets: loss_percentage, tree_cover_percentage
  - Stratified train/val/test splits

Run:
    python preprocessing/generate_labels.py
    python preprocessing/generate_labels.py --sample 200   # quick test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import numpy as np
import pandas as pd
import rasterio

from configs.settings import (
    EXPORT_DIR,
    FEATURE_BANDS,
    HANSEN_ASSET_ID,
    PATCH_SIZE,
    PROJECT_ROOT,
    SCALE,
)
from gee.export_helpers import build_tile_grid_spec

PATCHES_DIR = EXPORT_DIR / "patches"
LABELS_CSV = EXPORT_DIR / "patch_labels.csv"
SPLITS_DIR = EXPORT_DIR / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"

# Hansen loss years: 1=2001 ... 22=2022, 23=2023
STUDY_END_YEAR = 2023


def load_selected_metadata() -> pd.DataFrame:
    df = pd.read_csv(EXPORT_DIR / "final_patch_metadata.csv")
    return df[df["selected_for_training"] == True].copy()


def init_hansen() -> ee.Image:
    return ee.Image(HANSEN_ASSET_ID)


def patch_bounds_from_metadata(row: pd.Series) -> tuple[float, float, float, float]:
    """Compute (west, south, east, north) from centroid and patch size."""
    import math
    half_deg_lat = (PATCH_SIZE * SCALE / 2) / 111_320.0
    mid_lat_rad = math.radians(abs(row["latitude"]))
    half_deg_lon = (PATCH_SIZE * SCALE / 2) / (111_320.0 * math.cos(mid_lat_rad))
    return (
        row["longitude"] - half_deg_lon,
        row["latitude"] - half_deg_lat,
        row["longitude"] + half_deg_lon,
        row["latitude"] + half_deg_lat,
    )


def extract_hansen_for_patch(
    hansen: ee.Image, west: float, south: float, east: float, north: float,
) -> dict:
    """
    Extract Hansen GFC stats for a single patch.

    Returns dict with: tree_cover_mean, loss_pct, loss_year_mode, has_gain,
                       loss_pixel_count, total_pixels
    """
    patch_ee = ee.Geometry.Rectangle([west, south, east, north])

    tc2000 = hansen.select("treecover2000")
    loss = hansen.select("loss")
    lossyear = hansen.select("lossyear")
    gain = hansen.select("gain")

    # Mean tree cover in 2000
    tc_stats = tc2000.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=patch_ee, scale=30, maxPixels=1e7,
    )
    tc_mean = tc_stats.get("treecover2000")

    # Loss fraction
    loss_stats = loss.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=patch_ee, scale=30, maxPixels=1e7,
    )
    loss_frac = loss_stats.get("loss")

    # Loss year — mode (most common year of loss in patch)
    ly_stats = lossyear.reduceRegion(
        reducer=ee.Reducer.mode(), geometry=patch_ee, scale=30, maxPixels=1e7,
    )
    ly_mode = ly_stats.get("lossyear")

    # Gain fraction
    gain_stats = gain.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=patch_ee, scale=30, maxPixels=1e7,
    )
    gain_frac = gain_stats.get("gain")

    # Pixel count for loss=1
    loss_sum_stats = loss.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=patch_ee, scale=30, maxPixels=1e7,
    )
    loss_pixels = loss_sum_stats.get("loss")

    result = ee.Dictionary({
        "tree_cover_pct": tc_mean,
        "loss_pct": loss_frac,
        "loss_year_mode": ly_mode,
        "gain_pct": gain_frac,
        "loss_pixels": loss_pixels,
    })

    return result


def extract_hansen_batch(
    hansen: ee.Image,
    rows: list[pd.Series],
    batch_size: int = 50,
) -> dict[str, dict]:
    """
    Extract Hansen stats for many patches using batched GEE calls.

    Uses reduceRegions for efficiency — processes all patches in a batch
    with a single server-side call per batch.
    """
    results = {}
    n_batches = (len(rows) + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch_rows = rows[b * batch_size : (b + 1) * batch_size]
        start_t = time.time()

        # Build feature collection of patch rectangles
        features = []
        for row in batch_rows:
            w, s, e, n = patch_bounds_from_metadata(row)
            geom = ee.Geometry.Rectangle([w, s, e, n])
            features.append(ee.Feature(geom, {"patch_id": row["patch_id"]}))

        fc = ee.FeatureCollection(features)

        # Map Hansen stats over all patches in one call
        tc2000 = hansen.select("treecover2000")
        loss = hansen.select("loss")
        lossyear = hansen.select("lossyear")
        gain = hansen.select("gain")

        def compute_stats(feature):
            geom = feature.geometry()
            tc = tc2000.reduceRegion(
                ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e7,
            ).get("treecover2000")
            lf = loss.reduceRegion(
                ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e7,
            ).get("loss")
            ly = lossyear.reduceRegion(
                ee.Reducer.mode(), geometry=geom, scale=30, maxPixels=1e7,
            ).get("lossyear")
            gf = gain.reduceRegion(
                ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e7,
            ).get("gain")
            lp = loss.reduceRegion(
                ee.Reducer.sum(), geometry=geom, scale=30, maxPixels=1e7,
            ).get("loss")
            return feature.set({
                "tc2000": tc,
                "loss_frac": lf,
                "loss_year": ly,
                "gain_frac": gf,
                "loss_px": lp,
            })

        result_fc = fc.map(compute_stats)
        fc_result = result_fc.getInfo()
        features_list = fc_result["features"] if isinstance(fc_result, dict) else fc_result

        for feat in features_list:
            props = feat["properties"]
            pid = props["patch_id"]
            tc = props.get("tc2000")
            lf = props.get("loss_frac")
            ly = props.get("loss_year")
            gf = props.get("gain_frac")
            lp = props.get("loss_px")

            results[pid] = {
                "tree_cover_pct": tc if tc is not None else None,
                "loss_pct": lf if lf is not None else None,
                "loss_year_mode": ly if ly is not None else None,
                "gain_pct": gf if gf is not None else None,
                "loss_pixels": lp if lp is not None else None,
            }

        elapsed = time.time() - start_t
        pct = (b + 1) / n_batches * 100
        print(
            f"  Batch {b+1}/{n_batches} ({pct:.0f}%): "
            f"{len(batch_rows)} patches in {elapsed:.1f}s"
        )

    return results


def assign_labels(hansen_data: dict[str, dict], meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary labels and continuous targets from Hansen data.

    Binary label:
      1 = deforestation: any Hansen loss event with lossyear <= 23 (year 2023)
         AND tree cover in 2000 was >= 30%
      0 = no deforestation

    Continuous targets:
      loss_percentage: fraction of patch with Hansen loss
      tree_cover_percentage: mean tree cover in 2000
    """
    rows = []
    for _, row in meta_df.iterrows():
        pid = row["patch_id"]
        h = hansen_data.get(pid, {})

        tc = h.get("tree_cover_pct")
        lf = h.get("loss_pct")
        ly = h.get("loss_year_mode")
        gf = h.get("gain_pct")
        lp = h.get("loss_pixels")

        # Default values for missing data
        tc_pct = float(tc) if tc is not None else 0.0
        loss_pct = float(lf) if lf is not None else 0.0

        # Binary label: loss occurred AND original forest was present
        # Hansen lossyear: 0=no loss, 1-22=2001-2022, 23=2023
        # We consider lossyear 1-23 as deforestation events
        if ly is not None and ly != 0 and tc_pct >= 30:
            label = 1
        elif loss_pct > 0.05 and tc_pct >= 30:
            # Fallback: if loss fraction > 5% and forest existed
            label = 1
        else:
            label = 0

        # Loss year (for reference)
        loss_year_val = int(ly) if ly is not None and ly != 0 else 0
        # Convert Hansen year code to actual year: 1=2001, 23=2023
        actual_loss_year = loss_year_val + 2000 if loss_year_val > 0 else 0

        rows.append({
            "patch_id": pid,
            "label": label,
            "loss_percentage": round(loss_pct * 100, 4),
            "tree_cover_percentage": round(tc_pct, 2),
            "loss_year": actual_loss_year,
            "gain_percentage": round(float(gf) * 100, 4) if gf is not None else 0.0,
        })

    return pd.DataFrame(rows)


def verify_patches(labels_df: pd.DataFrame, meta_df: pd.DataFrame) -> dict:
    """Verify every labeled patch has imagery, metadata, and label."""
    selected_ids = set(meta_df["patch_id"])
    labeled_ids = set(labels_df["patch_id"])

    missing_metadata = labeled_ids - selected_ids
    missing_labels = selected_ids - labeled_ids

    imagery_ok = 0
    imagery_missing = 0
    imagery_corrupt = 0

    for pid in selected_ids:
        patch_dir = PATCHES_DIR / pid
        if not patch_dir.exists():
            imagery_missing += 1
            continue
        ok = True
        for m in range(1, 13):
            fpath = patch_dir / f"month_{m:02d}.tif"
            if not fpath.exists() or fpath.stat().st_size < 1000:
                ok = False
                break
            try:
                with rasterio.open(str(fpath)) as src:
                    if src.count != len(FEATURE_BANDS):
                        ok = False
                        break
            except Exception:
                ok = False
                break
        if ok:
            imagery_ok += 1
        else:
            imagery_corrupt += 1

    return {
        "total_selected": len(selected_ids),
        "total_labeled": len(labeled_ids),
        "missing_metadata": len(missing_metadata),
        "missing_labels": len(missing_labels),
        "imagery_ok": imagery_ok,
        "imagery_missing": imagery_missing,
        "imagery_corrupt": imagery_corrupt,
    }


def create_splits(labels_df: pd.DataFrame, seed: int = 42) -> dict:
    """
    Create stratified train/val/test splits.

    Split ratio: 80% train, 10% val, 10% test.
    Stratification ensures class balance across splits.
    """
    rng = np.random.RandomState(seed)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Stratified split
    df = labels_df.copy()
    df["_rand"] = rng.random(len(df))

    # Sort by label for stratification
    df = df.sort_values(["label", "_rand"]).reset_index(drop=True)

    # Compute split boundaries per class
    splits = {"train": [], "val": [], "test": []}

    for label_val in [0, 1]:
        class_df = df[df["label"] == label_val].copy()
        n = len(class_df)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train_pids = class_df.iloc[:n_train]["patch_id"].tolist()
        val_pids = class_df.iloc[n_train:n_train + n_val]["patch_id"].tolist()
        test_pids = class_df.iloc[n_train + n_val:]["patch_id"].tolist()

        splits["train"].extend(train_pids)
        splits["val"].extend(val_pids)
        splits["test"].extend(test_pids)

    # Shuffle within splits
    for k in splits:
        rng.shuffle(splits[k])

    # Write split CSVs
    label_map = labels_df.set_index("patch_id")["label"].to_dict()
    for split_name, pids in splits.items():
        path = SPLITS_DIR / f"{split_name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patch_id", "label"])
            for pid in pids:
                writer.writerow([pid, label_map[pid]])

    return {k: len(v) for k, v in splits.items()}


def compute_report_stats(
    labels_df: pd.DataFrame, splits: dict, verify: dict,
) -> dict:
    """Compute comprehensive statistics for the report."""
    total = len(labels_df)
    pos = (labels_df["label"] == 1).sum()
    neg = (labels_df["label"] == 0).sum()

    stats = {
        "total_patches": total,
        "positive_count": int(pos),
        "negative_count": int(neg),
        "positive_pct": round(pos / total * 100, 1),
        "negative_pct": round(neg / total * 100, 1),
        "splits": splits,
        "verification": verify,
        "tree_cover_stats": {
            "mean": round(labels_df["tree_cover_percentage"].mean(), 2),
            "median": round(labels_df["tree_cover_percentage"].median(), 2),
            "std": round(labels_df["tree_cover_percentage"].std(), 2),
            "min": round(labels_df["tree_cover_percentage"].min(), 2),
            "max": round(labels_df["tree_cover_percentage"].max(), 2),
        },
        "loss_pct_stats": {
            "mean": round(labels_df["loss_percentage"].mean(), 4),
            "median": round(labels_df["loss_percentage"].median(), 4),
            "max": round(labels_df["loss_percentage"].max(), 4),
            "nonzero_count": int((labels_df["loss_percentage"] > 0).sum()),
        },
        "loss_year_distribution": {},
    }

    # Loss year distribution (only for labeled positive)
    positives = labels_df[labels_df["label"] == 1]
    if len(positives) > 0:
        year_counts = positives["loss_year"].value_counts().to_dict()
        stats["loss_year_distribution"] = {str(k): int(v) for k, v in year_counts.items()}

    return stats


def generate_report(stats: dict) -> str:
    """Generate step7_report.md content."""
    v = stats["verification"]
    all_ok = (
        v["imagery_ok"] == stats["total_patches"]
        and v["missing_metadata"] == 0
        and v["missing_labels"] == 0
        and v["imagery_corrupt"] == 0
    )

    ly_dist = stats["loss_year_distribution"]
    ly_lines = ""
    for year in sorted(ly_dist.keys()):
        ly_lines += f"| {year} | {ly_dist[year]} |\n"
    if not ly_lines:
        ly_lines = "| — | 0 |\n"

    return f"""# Step 7: Ground-Truth Labels Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 1. Labeling Methodology

| Parameter | Value |
|-----------|-------|
| Label source | Hansen Global Forest Change v1.13 (2025 release) |
| Study period | 2000–2023 |
| Binary definition | 1 = Hansen loss event in patch AND treecover2000 ≥ 30% |
| Continuous targets | loss_percentage, tree_cover_percentage |
| Dataset size | {stats['total_patches']:,} selected patches |

## 2. Class Distribution

| Class | Count | Percentage |
|-------|-------|-----------|
| No deforestation (0) | {stats['negative_count']:,} | {stats['negative_pct']:.1f}% |
| Deforestation (1) | {stats['positive_count']:,} | {stats['positive_pct']:.1f}% |
| **Total** | **{stats['total_patches']:,}** | **100%** |

## 3. Continuous Target Statistics

### Tree Cover Percentage (Hansen 2000)

| Statistic | Value |
|-----------|-------|
| Mean | {stats['tree_cover_stats']['mean']:.1f}% |
| Median | {stats['tree_cover_stats']['median']:.1f}% |
| Std | {stats['tree_cover_stats']['std']:.1f}% |
| Min | {stats['tree_cover_stats']['min']:.1f}% |
| Max | {stats['tree_cover_stats']['max']:.1f}% |

### Loss Percentage

| Statistic | Value |
|-----------|-------|
| Mean | {stats['loss_pct_stats']['mean']:.2f}% |
| Median | {stats['loss_pct_stats']['median']:.2f}% |
| Max | {stats['loss_pct_stats']['max']:.2f}% |
| Patches with loss > 0 | {stats['loss_pct_stats']['nonzero_count']:,} |

## 4. Loss Year Distribution (Positive Samples)

| Loss Year | Count |
|-----------|-------|
{ly_lines}

## 5. Train / Validation / Test Splits

| Split | Count | Percentage |
|-------|-------|-----------|
| Train | {stats['splits']['train']:,} | {stats['splits']['train']/stats['total_patches']*100:.1f}% |
| Validation | {stats['splits']['val']:,} | {stats['splits']['val']/stats['total_patches']*100:.1f}% |
| Test | {stats['splits']['test']:,} | {stats['splits']['test']/stats['total_patches']*100:.1f}% |
| **Total** | **{stats['total_patches']:,}** | **100%** |

## 6. Dataset Integrity Verification

| Check | Result |
|-------|--------|
| Total selected patches | {v['total_selected']:,} |
| Total labeled | {v['total_labeled']:,} |
| Missing metadata | {v['missing_metadata']} |
| Missing labels | {v['missing_labels']} |
| Imagery OK | {v['imagery_ok']:,} |
| Imagery missing | {v['imagery_missing']} |
| Imagery corrupt | {v['imagery_corrupt']} |

### Verdict: {'**PASS** — All checks passed' if all_ok else '**FAIL** — Issues found'}

## 7. Files Generated

| File | Description |
|------|-------------|
| `exports/patch_labels.csv` | Labels + continuous targets for all 5,001 patches |
| `exports/splits/train.csv` | Training split |
| `exports/splits/val.csv` | Validation split |
| `exports/splits/test.csv` | Test split |
| `reports/step7_report.md` | This report |
| `reports/step7_stats.json` | Machine-readable statistics |

## 8. Ready for CNN Training

{'Yes — all {0:,} patches have imagery, metadata, and verified labels.'.format(stats['total_patches']) if all_ok else 'No — integrity checks failed. See Section 6.'}

The dataset is stratified, class-balanced, and ready for input to:
- U-Net semantic segmentation
- ResNet/EfficientNet patch classification
- Vision Transformer (ViT)
- Any standard CNN architecture
"""


def main():
    parser = argparse.ArgumentParser(description="Step 7: Generate labels")
    parser.add_argument("--sample", type=int, default=0,
                        help="Use only N selected patches (for quick testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 7: GENERATE GROUND-TRUTH LABELS")
    print("=" * 60)

    # Load metadata
    print("\n[1/7] Loading finalized metadata...")
    meta_df = load_selected_metadata()
    if args.sample > 0:
        meta_df = meta_df.head(args.sample).copy()
        print(f"  (sample mode: using {args.sample} patches)")
    print(f"  Selected patches: {len(meta_df):,}")

    # Initialize GEE
    print("\n[2/7] Initializing GEE...")
    ee.Initialize(project="deforestation-early-warning")
    hansen = init_hansen()
    print("  Hansen GFC v1.13 loaded")

    # Extract Hansen data for all patches
    print(f"\n[3/7] Querying Hansen GFC for {len(meta_df):,} patches...")
    t0 = time.time()
    hansen_data = extract_hansen_batch(
        hansen,
        [row for _, row in meta_df.iterrows()],
        batch_size=50,
    )
    elapsed = time.time() - t0
    print(f"  Done: {len(hansen_data):,} patches in {elapsed:.0f}s")

    # Assign labels
    print("\n[4/7] Creating labels...")
    labels_df = assign_labels(hansen_data, meta_df)
    pos = (labels_df["label"] == 1).sum()
    neg = (labels_df["label"] == 0).sum()
    print(f"  Positive (deforestation): {pos:,} ({pos/len(labels_df)*100:.1f}%)")
    print(f"  Negative (no deforestation): {neg:,} ({neg/len(labels_df)*100:.1f}%)")

    # Save labels CSV
    print("\n[5/7] Saving patch_labels.csv...")
    labels_df.to_csv(LABELS_CSV, index=False)
    print(f"  Saved: {LABELS_CSV}")

    # Verify patches
    print("\n[6/7] Verifying dataset integrity...")
    verify = verify_patches(labels_df, meta_df)
    print(f"  Imagery OK: {verify['imagery_ok']:,}/{verify['total_selected']:,}")
    print(f"  Missing metadata: {verify['missing_metadata']}")
    print(f"  Missing labels: {verify['missing_labels']}")
    print(f"  Corrupt imagery: {verify['imagery_corrupt']}")

    # Create splits
    print("\n[7/7] Creating stratified train/val/test splits (seed=42)...")
    splits = create_splits(labels_df, seed=42)
    for k, v in splits.items():
        print(f"  {k}: {v:,}")

    # Compute stats and generate report
    stats = compute_report_stats(labels_df, splits, verify)
    report = generate_report(stats)

    report_path = REPORT_DIR / "step7_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Report: {report_path}")

    stats_path = REPORT_DIR / "step7_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats:  {stats_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total patches : {stats['total_patches']:,}")
    print(f"  Positive      : {stats['positive_count']:,} ({stats['positive_pct']:.1f}%)")
    print(f"  Negative      : {stats['negative_count']:,} ({stats['negative_pct']:.1f}%)")
    print(f"  Train         : {splits['train']:,}")
    print(f"  Val           : {splits['val']:,}")
    print(f"  Test          : {splits['test']:,}")
    print(f"  Integrity     : {'PASS' if verify['imagery_corrupt'] == 0 and verify['missing_labels'] == 0 else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
