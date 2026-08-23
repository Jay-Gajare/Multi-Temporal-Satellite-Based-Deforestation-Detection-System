"""
Batch inference: predict an entire folder of patches and generate summary statistics.

Usage:
    python -m inference.predict_folder --input-dir <patches_folder> [--threshold 0.5]

Outputs:
    outputs/batch_predictions.csv  — per-patch predictions
    outputs/batch_summary.json     — aggregate statistics
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from inference.predict import DeforestationPredictor
from inference.utils import (
    PROJECT_ROOT, OUTPUTS_DIR, SPLITS_DIR, LABELS_CSV,
    PATCHES_DIR, DEFAULT_THRESHOLD,
    get_system_info, get_process_memory_gb,
    save_json, ensure_dir, setup_logging, timer,
)

logger = logging.getLogger("inference.predict_folder")


def find_patches(input_dir: Path) -> list[Path]:
    """Find all valid patch directories within input_dir.

    A valid patch directory contains month_01.tif .. month_12.tif.
    """
    input_dir = Path(input_dir)
    patches: list[Path] = []

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    # Check if input_dir itself is a patch
    has_monthly = all((input_dir / f"month_{m:02d}.tif").exists() for m in range(1, 13))
    if has_monthly:
        return [input_dir]

    # Look for subdirectories that are patches
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            if all((child / f"month_{m:02d}.tif").exists() for m in range(1, 13)):
                patches.append(child)

    # If no subdirectories, look for .tif files in the directory itself
    if not patches:
        tif_files = sorted(input_dir.glob("*.tif"))
        if tif_files:
            patches.append(input_dir)

    return patches


def predict_folder(
    input_dir: str | Path,
    model_path: str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Run inference on all patches in a directory.

    Parameters
    ----------
    input_dir : Directory containing patch subdirectories
    model_path : Path to model checkpoint
    threshold : Classification threshold
    output_dir : Override output directory
    limit : Maximum number of patches to process

    Returns
    -------
    dict with per-patch results and summary statistics
    """
    input_dir = Path(input_dir)
    out_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    ensure_dir(out_dir)

    # Find patches
    patches = find_patches(input_dir)
    if limit:
        patches = patches[:limit]
    logger.info("Found %d patches to process in %s", len(patches), input_dir)

    if not patches:
        logger.warning("No patches found in %s", input_dir)
        return {"patches": [], "summary": {"total_patches": 0}}

    # Initialize predictor
    predictor = DeforestationPredictor(model_path=model_path, threshold=threshold)

    # Collect results
    results: list[dict] = []
    total_inference_time = 0.0
    total_preprocess_time = 0.0
    total_gradcam_time = 0.0
    n_deforestation = 0
    n_forest = 0
    confidences: list[float] = []
    probabilities: list[float] = []
    inference_times: list[float] = []

    t_start = time.perf_counter()

    for i, patch_path in enumerate(patches):
        patch_id = patch_path.name
        logger.info("[%d/%d] Processing %s", i + 1, len(patches), patch_id)

        try:
            result = predictor.predict(patch_path, patch_id=patch_id)
            results.append(result)

            # Accumulate statistics
            pred = result["prediction"]
            prob = result["probability"]
            conf = result["confidence"]
            inf_time = result["inference_time_s"]

            if pred == 1:
                n_deforestation += 1
            else:
                n_forest += 1

            probabilities.append(prob)
            confidences.append(conf)
            inference_times.append(inf_time)
            total_inference_time += inf_time
            total_preprocess_time += result.get("load_time_s", 0) + result.get("validate_time_s", 0)
            total_gradcam_time += result.get("gradcam_time_s", 0)

        except Exception as e:
            logger.error("Failed to process %s: %s", patch_id, e)
            results.append({
                "patch_id": patch_id,
                "error": str(e),
                "prediction": -1,
            })

    total_time = time.perf_counter() - t_start
    n_processed = len([r for r in results if "error" not in r])
    n_failed = len(results) - n_processed

    # Build summary
    summary: dict = {
        "total_patches": len(patches),
        "processed": n_processed,
        "failed": n_failed,
        "predicted_forest": n_forest,
        "predicted_deforestation": n_deforestation,
        "forest_pct": round(n_forest / max(n_processed, 1) * 100, 1),
        "deforestation_pct": round(n_deforestation / max(n_processed, 1) * 100, 1),
        "avg_probability": round(float(np.mean(probabilities)), 4) if probabilities else 0,
        "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0,
        "avg_inference_time_s": round(float(np.mean(inference_times)), 4) if inference_times else 0,
        "total_inference_time_s": round(total_inference_time, 3),
        "total_preprocess_time_s": round(total_preprocess_time, 3),
        "total_gradcam_time_s": round(total_gradcam_time, 3),
        "total_wall_time_s": round(total_time, 3),
        "throughput_per_second": round(n_processed / max(total_time, 0.001), 2),
        "threshold": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": get_system_info(),
        "memory_gb": get_process_memory_gb(),
    }

    # Save batch predictions CSV
    csv_rows = []
    for r in results:
        csv_rows.append({
            "patch_id": r.get("patch_id", ""),
            "prediction": r.get("prediction", -1),
            "prediction_label": r.get("prediction_label", "error"),
            "probability": r.get("probability", 0),
            "confidence": r.get("confidence", 0),
            "inference_time_s": r.get("inference_time_s", 0),
            "error": r.get("error", ""),
        })
    batch_df = pd.DataFrame(csv_rows)
    batch_csv_path = out_dir / "batch_predictions.csv"
    batch_df.to_csv(batch_csv_path, index=False)
    logger.info("Batch predictions saved: %s", batch_csv_path)

    # Save summary JSON
    save_json({"summary": summary, "results": results}, out_dir / "batch_summary.json")

    logger.info(
        "Batch complete: %d patches, %d deforestation, %d forest, %.1f patches/s",
        n_processed, n_deforestation, n_forest, summary["throughput_per_second"],
    )

    return {"results": results, "summary": summary}


def main() -> None:
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Deforestation inference — batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", "-i", required=True, type=str,
                        help="Directory containing patch subdirectories")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Classification threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: outputs/)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of patches to process")
    args = parser.parse_args()

    output = predict_folder(
        input_dir=args.input_dir,
        model_path=args.model,
        threshold=args.threshold,
        output_dir=args.output_dir,
        limit=args.limit,
    )

    print(json.dumps(output["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
