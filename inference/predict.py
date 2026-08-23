"""
Single-sample inference pipeline.

Loads model, preprocesses input, runs inference, generates Grad-CAM,
and saves prediction artifacts (JSON, CSV, overlay, Grad-CAM).

Usage:
    python -m inference.predict --input <path> [--threshold 0.5] [--output-dir outputs/]
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
import torch

from inference.preprocessing import (
    InputError,
    detect_input_type,
    load_input,
    validate_geotiff,
    validate_monthly_folder,
    validate_patch_dir,
)
from inference.gradcam import generate_gradcam, save_gradcam, save_gradcam_overlay, make_rgbComposite
from inference.utils import (
    PROJECT_ROOT, MODELS_DIR, OUTPUTS_DIR, GRADCAM_DIR,
    DEFAULT_THRESHOLD, DEFAULT_DROPOUT, EXPECTED_CHANNELS,
    get_device, get_system_info, get_process_memory_gb,
    save_json, ensure_dir, setup_logging, timer,
)

from training.config import Config
from training.models import build_model

logger = logging.getLogger("inference.predict")


class DeforestationPredictor:
    """Production inference engine for deforestation detection.

    Usage:
        predictor = DeforestationPredictor()
        result = predictor.predict("path/to/input")
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the predictor.

        Parameters
        ----------
        model_path : Path to best_model.pth. If None, uses models/run_01/best_model.pth
        threshold : Classification threshold for deforestation (default 0.50)
        device : Force a specific device. If None, auto-detects.
        """
        self.threshold = threshold
        self._model_path = Path(model_path) if model_path else MODELS_DIR / "run_01" / "best_model.pth"
        self._device = torch.device(device) if device else get_device()
        self._model: torch.nn.Module | None = None
        self._model_loaded = False
        self._checkpoint_info: dict = {}

    def load_model(self) -> None:
        """Load model weights from checkpoint."""
        if self._model_loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self._model_path}")

        logger.info("Loading model from %s", self._model_path)
        t0 = time.perf_counter()

        ckpt = torch.load(str(self._model_path), map_location=self._device, weights_only=False)

        architecture = ckpt["config"]["architecture"]
        temporal_strategy = ckpt["config"].get("temporal_strategy", "temporal_stack")
        in_channels = EXPECTED_CHANNELS if temporal_strategy == "temporal_stack" else 9

        self._model = build_model(
            architecture=architecture,
            in_channels=in_channels,
            pretrained=False,
            dropout=DEFAULT_DROPOUT,
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

        load_time = time.perf_counter() - t0
        self._checkpoint_info = {
            "epoch": ckpt.get("epoch", -1),
            "best_val_f1": ckpt.get("best_val_f1", -1.0),
            "architecture": architecture,
            "temporal_strategy": temporal_strategy,
            "in_channels": in_channels,
        }
        self._model_loaded = True
        logger.info(
            "Model loaded in %.3fs — %s (epoch %d, val_f1=%.4f, device=%s)",
            load_time, architecture, self._checkpoint_info["epoch"],
            self._checkpoint_info["best_val_f1"], self._device,
        )

    def predict(self, input_path: str | Path, patch_id: str | None = None) -> dict:
        """Run full inference pipeline on a single input.

        Parameters
        ----------
        input_path : Path to GeoTIFF, monthly folder, or patch directory
        patch_id : Optional patch identifier. If None, derived from filename.

        Returns
        -------
        dict with keys: patch_id, prediction, probability, confidence,
                        threshold, timing breakdown, gradcam_path, etc.
        """
        self.load_model()
        input_path = Path(input_path)

        if patch_id is None:
            patch_id = input_path.stem if input_path.is_file() else input_path.name

        result: dict = {
            "patch_id": patch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_path": str(self._model_path),
            "device": str(self._device),
            "threshold": self.threshold,
        }

        # --- 1. Validate input ---
        with timer() as t_validate:
            input_type = detect_input_type(input_path)
            result["input_type"] = input_type
            if input_type == "geotiff":
                validate_geotiff(input_path)
            elif input_type == "patch_dir":
                validate_patch_dir(input_path)
            else:
                validate_monthly_folder(input_path)
        result["time_validate_s"] = round(t_validate["elapsed"], 6)

        # --- 2. File loading (GeoTIFF I/O from disk) ---
        with timer() as t_fileload:
            from inference.preprocessing import detect_input_type as _dit, read_geotiff, load_temporal_stack_from_monthly
            input_type_local = detect_input_type(input_path)
            months_data = []
            for m in range(1, 13):
                fpath = input_path / f"month_{m:02d}.tif"
                if fpath.exists():
                    data, _ = read_geotiff(fpath)
                    months_data.append(data)
        result["time_fileload_s"] = round(t_fileload["elapsed"], 6)

        # --- 3. Preprocessing (spectral indices + temporal stacking) ---
        with timer() as t_preproc:
            from inference.preprocessing import compute_spectral_indices
            processed_months = []
            for data in months_data:
                if data.shape[0] == 9:
                    processed_months.append(compute_spectral_indices(data))
                elif data.shape[0] == 12:
                    processed_months.append(data)
                else:
                    processed_months.append(compute_spectral_indices(data[:9]))
            stack = np.concatenate(processed_months, axis=0).astype(np.float32)
        result["time_preprocess_s"] = round(t_preproc["elapsed"], 6)
        result["stack_shape"] = list(stack.shape)

        # --- 4. Tensor conversion (numpy → torch + device transfer) ---
        with timer() as t_tensor:
            tensor = torch.from_numpy(stack).unsqueeze(0).to(self._device)
        result["time_tensor_s"] = round(t_tensor["elapsed"], 6)

        # --- 5. Model inference (forward pass + sigmoid) ---
        with timer() as t_infer, torch.no_grad():
            logits = self._model(tensor).squeeze(-1)
            probability = torch.sigmoid(logits).item()
        result["time_inference_s"] = round(t_infer["elapsed"], 6)
        result["logit"] = round(float(logits.item()), 6)
        result["probability"] = round(probability, 6)
        result["prediction"] = int(probability >= self.threshold)
        result["prediction_label"] = "deforestation" if result["prediction"] == 1 else "no_deforestation"
        result["confidence"] = round(abs(probability - 0.5) * 2, 6)

        # --- 6. Grad-CAM heatmap generation ---
        with timer() as t_cam:
            heatmap = generate_gradcam(self._model, tensor)
        result["time_gradcam_s"] = round(t_cam["elapsed"], 6)

        # --- 7. Visualization (matplotlib rendering) ---
        output_dir = ensure_dir(OUTPUTS_DIR / patch_id)
        gradcam_path = output_dir / "gradcam.png"
        overlay_path = output_dir / "prediction_overlay.png"
        rgb = make_rgbComposite(stack, month=0)

        with timer() as t_viz:
            save_gradcam_overlay(
                heatmap, rgb, gradcam_path,
                patch_id=patch_id,
                prediction=result["prediction"],
                probability=result["probability"],
                confidence=result["confidence"],
            )
            save_gradcam_overlay(
                heatmap, rgb, overlay_path,
                patch_id=patch_id,
                prediction=result["prediction"],
                probability=result["probability"],
            )
        result["time_visualize_s"] = round(t_viz["elapsed"], 6)
        result["gradcam_path"] = str(gradcam_path)
        result["overlay_path"] = str(overlay_path)

        # --- 8. File saving (JSON + CSV) ---
        with timer() as t_save:
            save_json(result, output_dir / "prediction.json")
            df = pd.DataFrame([{
                "patch_id": patch_id,
                "prediction": result["prediction"],
                "prediction_label": result["prediction_label"],
                "probability": result["probability"],
                "confidence": result["confidence"],
                "threshold": self.threshold,
                "device": str(self._device),
                "inference_time_s": result["time_inference_s"],
                "timestamp": result["timestamp"],
            }])
            df.to_csv(output_dir / "prediction.csv", index=False)
        result["time_save_s"] = round(t_save["elapsed"], 6)

        # --- Total latency ---
        result["time_total_s"] = round(
            result["time_validate_s"] + result["time_fileload_s"] + result["time_preprocess_s"]
            + result["time_tensor_s"] + result["time_inference_s"] + result["time_gradcam_s"]
            + result["time_visualize_s"] + result["time_save_s"],
            6,
        )
        result["memory_gb"] = get_process_memory_gb()

        logger.info(
            "Prediction: %s -> %s (P=%.4f, conf=%.4f) total=%.3fs [file=%.1fms preproc=%.1fms tensor=%.1fms infer=%.1fms cam=%.1fms viz=%.1fms save=%.1fms]",
            patch_id, result["prediction_label"], result["probability"],
            result["confidence"], result["time_total_s"],
            result["time_fileload_s"] * 1000, result["time_preprocess_s"] * 1000,
            result["time_tensor_s"] * 1000, result["time_inference_s"] * 1000,
            result["time_gradcam_s"] * 1000, result["time_visualize_s"] * 1000,
            result["time_save_s"] * 1000,
        )

        return result


def predict_single(
    input_path: str | Path,
    model_path: str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: str | Path | None = None,
    patch_id: str | None = None,
) -> dict:
    """Convenience function to run single-sample inference.

    Parameters
    ----------
    input_path : Path to input (GeoTIFF, monthly folder, or patch directory)
    model_path : Path to model checkpoint
    threshold : Classification threshold
    output_dir : Override output directory
    patch_id : Override patch ID

    Returns
    -------
    dict with full prediction results
    """
    if output_dir:
        global OUTPUTS_DIR
        from inference import utils
        utils.OUTPUTS_DIR = Path(output_dir)

    predictor = DeforestationPredictor(model_path=model_path, threshold=threshold)
    return predictor.predict(input_path, patch_id=patch_id)


def main() -> None:
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Deforestation inference — single sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, type=str,
                        help="Path to input: .tif file, monthly folder, or patch directory")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to model checkpoint (default: models/run_01/best_model.pth)")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Classification threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: outputs/)")
    parser.add_argument("--patch-id", type=str, default=None,
                        help="Patch identifier (default: derived from input path)")
    args = parser.parse_args()

    result = predict_single(
        input_path=args.input,
        model_path=args.model,
        threshold=args.threshold,
        output_dir=args.output_dir,
        patch_id=args.patch_id,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
