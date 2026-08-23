"""
Grad-CAM heatmap generation for deforestation model explainability.

Implements Grad-CAM from scratch using PyTorch hooks — no external dependency.
Targets the last convolutional layer (model.layer4 for ResNet, features[-1] for EfficientNet).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("inference.gradcam")


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Hooks into a target layer to capture forward activations and backward
    gradients, then computes a spatial heatmap showing which regions most
    influenced the model's prediction.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap for input.

        Parameters
        ----------
        input_tensor : Tensor of shape (1, C, H, W)

        Returns
        -------
        ndarray of shape (H, W) — heatmap in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor).squeeze(-1)
        output.backward()

        # Global average pooling of gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0, 0].cpu().numpy()

    def remove(self) -> None:
        """Remove hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def get_target_layer(model: nn.Module) -> nn.Module:
    """Get the last convolutional layer for Grad-CAM based on model architecture."""
    model_type = type(model).__name__

    if hasattr(model, "layer4"):
        # ResNet18/50
        return model.layer4
    elif hasattr(model, "features"):
        # EfficientNet — last conv block
        return model.features[-1]
    else:
        raise ValueError(f"Cannot locate target layer for model type: {model_type}")


def generate_gradcam(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single input.

    Parameters
    ----------
    model : Trained model
    input_tensor : Tensor of shape (1, C, H, W)

    Returns
    -------
    ndarray of shape (H, W) — heatmap in [0, 1]
    """
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    try:
        heatmap = gradcam.generate(input_tensor)
    finally:
        gradcam.remove()

    return heatmap


def save_gradcam(
    heatmap: np.ndarray,
    output_path: Path,
    patch_id: str = "",
    prediction: int | None = None,
    probability: float | None = None,
    confidence: float | None = None,
) -> None:
    """Save a Grad-CAM visualization with optional metadata overlay.

    Generates a figure with three panels:
      1. Raw RGB composite (bands B4/B3/B2 from month 1)
      2. Heatmap only
      3. Overlay of heatmap on RGB

    Parameters
    ----------
    heatmap : (H, W) ndarray in [0, 1]
    output_path : Where to save the PNG
    patch_id : Optional patch identifier
    prediction : Predicted class (0 or 1)
    probability : Predicted probability
    confidence : Confidence score
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Placeholder RGB (grey) — the actual patch bands aren't passed here
    # to keep this function self-contained; the caller can overlay on real data
    rgb_placeholder = np.ones((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.float32) * 0.5

    axes[0].imshow(rgb_placeholder)
    axes[0].set_title("Input (RGB placeholder)")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(rgb_placeholder)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    title_parts = ["Grad-CAM Overlay"]
    if patch_id:
        title_parts.append(f"Patch: {patch_id}")
    if prediction is not None:
        label = "Deforestation" if prediction == 1 else "No Deforestation"
        title_parts.append(f"Pred: {label}")
    if probability is not None:
        title_parts.append(f"P={probability:.3f}")
    if confidence is not None:
        title_parts.append(f"Conf={confidence:.3f}")
    axes[2].set_title("\n".join(title_parts))
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grad-CAM saved: %s", output_path)


def save_gradcam_overlay(
    heatmap: np.ndarray,
    rgb_bands: np.ndarray,
    output_path: Path,
    patch_id: str = "",
    prediction: int | None = None,
    probability: float | None = None,
    confidence: float | None = None,
) -> None:
    """Save Grad-CAM overlay on actual RGB imagery.

    Parameters
    ----------
    heatmap : (H, W) ndarray in [0, 1]
    rgb_bands : (H, W, 3) ndarray — RGB composite from the patch
    output_path : Where to save the PNG
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(rgb_bands)
    axes[0].set_title("RGB Composite")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(rgb_bands)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    title_parts = ["Grad-CAM Overlay"]
    if patch_id:
        title_parts.append(f"Patch: {patch_id}")
    if prediction is not None:
        label = "Deforestation" if prediction == 1 else "No Deforestation"
        title_parts.append(f"Pred: {label}")
    if probability is not None:
        title_parts.append(f"P={probability:.4f}")
    axes[2].set_title("\n".join(title_parts))
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grad-CAM overlay saved: %s", output_path)


def make_rgbComposite(stack: np.ndarray, month: int = 0) -> np.ndarray:
    """Extract RGB from temporal stack for visualization.

    Parameters
    ----------
    stack : (108, 64, 64) temporal stack
    month : Month index (0-based) to extract from

    Returns
    -------
    (64, 64, 3) ndarray, normalized to [0, 1] for display
    """
    offset = month * 9
    r = stack[offset + 2]  # B4 (Red)
    g = stack[offset + 1]  # B3 (Green)
    b = stack[offset + 0]  # B2 (Blue)

    rgb = np.stack([r, g, b], axis=-1)
    # Clip to typical S2 reflectance range and normalize
    rgb = np.clip(rgb / 3000.0, 0, 1).astype(np.float32)
    return rgb
