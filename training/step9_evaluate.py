"""
Step 9 — Hold-out Test Set Evaluation

Loads best_model.pth, runs inference on test split, computes comprehensive
metrics, optimizes threshold, performs error analysis, and generates Grad-CAM
visualizations.
"""
from __future__ import annotations

import json
import csv
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    average_precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from training.config import Config, PROJECT_ROOT, EXPORT_DIR, MODELS_DIR, REPORTS_DIR
from training.dataset import DeforestationDataset
from training.models import build_model

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
RUN_DIR = MODELS_DIR / "run_01"
BEST_MODEL_PATH = RUN_DIR / "best_model.pth"
TEST_CSV = EXPORT_DIR / "splits" / "test.csv"
LABELS_CSV = EXPORT_DIR / "patch_labels.csv"
OUTPUT_DIR = RUN_DIR / "test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load model
# ──────────────────────────────────────────────────────────────────────────────
logger.info("Loading best model from %s", BEST_MODEL_PATH)
ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=False)

cfg = Config()
cfg.model.architecture = ckpt["config"]["architecture"]
cfg.data.temporal_strategy = ckpt["config"]["temporal_strategy"]

model = build_model(
    architecture=cfg.model.architecture,
    in_channels=108,
    pretrained=False,
    dropout=cfg.model.dropout,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
logger.info("Model loaded (epoch %d, best_val_f1=%.4f)", ckpt["epoch"], ckpt["best_val_f1"])

# ──────────────────────────────────────────────────────────────────────────────
# 2. Build test dataset and DataLoader
# ──────────────────────────────────────────────────────────────────────────────
test_ds = DeforestationDataset(
    split_csv=TEST_CSV,
    labels_csv=LABELS_CSV,
    patches_dir=EXPORT_DIR / "patches",
    temporal_strategy=cfg.data.temporal_strategy,
    use_cache=True,
)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
logger.info("Test set: %d samples", len(test_ds))

# ──────────────────────────────────────────────────────────────────────────────
# 3. Run inference — collect logits, labels, patch_ids, probabilities
# ──────────────────────────────────────────────────────────────────────────────
all_logits = []
all_labels = []
all_patch_ids = list(test_ds.patch_ids)

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Inference"):
        logits = model(inputs).squeeze(-1)
        all_logits.append(logits.numpy())
        all_labels.append(labels.numpy())

logits_np = np.concatenate(all_logits)
labels_np = np.concatenate(all_labels)
probs_np = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid

logger.info("Inference complete: %d samples, prob range [%.4f, %.4f]",
            len(probs_np), probs_np.min(), probs_np.max())

# ──────────────────────────────────────────────────────────────────────────────
# 4. Compute comprehensive metrics at threshold=0.5
# ──────────────────────────────────────────────────────────────────────────────
def compute_all_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    pr_auc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    cm = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }

metrics_05 = compute_all_metrics(labels_np, probs_np, threshold=0.5)
logger.info("=== Test Metrics (threshold=0.5) ===")
for k, v in metrics_05.items():
    if k != "confusion_matrix":
        logger.info("  %-22s %.4f", k, v)
logger.info("  confusion_matrix: %s", metrics_05["confusion_matrix"])

# ──────────────────────────────────────────────────────────────────────────────
# 5. Threshold optimization
# ──────────────────────────────────────────────────────────────────────────────
logger.info("Optimizing threshold 0.10–0.90 ...")
thresholds = np.arange(0.10, 0.901, 0.01)
threshold_results = []
for t in thresholds:
    preds_t = (probs_np >= t).astype(int)
    f1_t = f1_score(labels_np, preds_t, zero_division=0)
    prec_t = precision_score(labels_np, preds_t, zero_division=0)
    rec_t = recall_score(labels_np, preds_t, zero_division=0)
    threshold_results.append({
        "threshold": round(float(t), 2),
        "f1": float(f1_t),
        "precision": float(prec_t),
        "recall": float(rec_t),
    })

threshold_df = pd.DataFrame(threshold_results)
best_idx = threshold_df["f1"].idxmax()
best_thresh = threshold_df.loc[best_idx]
logger.info("Best threshold=%.2f  F1=%.4f  P=%.4f  R=%.4f",
            best_thresh["threshold"], best_thresh["f1"],
            best_thresh["precision"], best_thresh["recall"])

# Recompute metrics at best threshold
metrics_best = compute_all_metrics(labels_np, probs_np, threshold=best_thresh["threshold"])

# ──────────────────────────────────────────────────────────────────────────────
# 6. Plots — confusion matrix, ROC, PR curve, threshold sweep
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, output_path, title="Confusion Matrix"):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["No Deforestation", "Deforestation"],
           yticklabels=["No Deforestation", "Deforestation"],
           xlabel="Predicted Label", ylabel="True Label", title=title)
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_arr[i, j]:,d}", ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_roc(labels, probs, output_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test Set — ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_pr_curve(labels, probs, output_path):
    precision_arr, recall_arr, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_arr, precision_arr, "r-", linewidth=2, label=f"PR (AP = {pr_auc:.3f})")
    baseline = labels.sum() / len(labels)
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Test Set — Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_threshold_sweep(threshold_df, best_thresh, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_df["threshold"], threshold_df["f1"], "b-o", linewidth=1.5,
            markersize=3, label="F1")
    ax.plot(threshold_df["threshold"], threshold_df["precision"], "g-", linewidth=1,
            alpha=0.7, label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], "r-", linewidth=1,
            alpha=0.7, label="Recall")
    ax.axvline(x=best_thresh["threshold"], color="k", linestyle="--", alpha=0.7,
               label=f"Best F1 @ t={best_thresh['threshold']:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Optimization")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

# Generate plots
plot_confusion_matrix(metrics_05["confusion_matrix"], OUTPUT_DIR / "confusion_matrix.png",
                       title="Confusion Matrix (threshold=0.50)")
plot_roc(labels_np, probs_np, OUTPUT_DIR / "roc_curve.png")
plot_pr_curve(labels_np, probs_np, OUTPUT_DIR / "precision_recall_curve.png")
plot_threshold_sweep(threshold_df, best_thresh, OUTPUT_DIR / "threshold_sweep.png")
logger.info("Plots saved to %s", OUTPUT_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Error analysis — FPs and FNs with confidence scores
# ──────────────────────────────────────────────────────────────────────────────
preds_05 = (probs_np >= 0.5).astype(int)
error_rows = []
for i in range(len(labels_np)):
    row = {
        "patch_id": all_patch_ids[i],
        "true_label": int(labels_np[i]),
        "predicted_label": int(preds_05[i]),
        "probability": round(float(probs_np[i]), 6),
        "logit": round(float(logits_np[i]), 6),
        "correct": bool(preds_05[i] == labels_np[i]),
    }
    if not row["correct"]:
        if row["true_label"] == 0 and row["predicted_label"] == 1:
            row["error_type"] = "false_positive"
        elif row["true_label"] == 1 and row["predicted_label"] == 0:
            row["error_type"] = "false_negative"
        else:
            row["error_type"] = "other"
    else:
        row["error_type"] = "correct"
    error_rows.append(row)

error_df = pd.DataFrame(error_rows)
error_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

fp_df = error_df[error_df["error_type"] == "false_positive"].sort_values("probability", ascending=False)
fn_df = error_df[error_df["error_type"] == "false_negative"].sort_values("probability", ascending=True)
correct_df = error_df[error_df["error_type"] == "correct"]

logger.info("Error analysis: %d correct, %d FP, %d FN",
            len(correct_df), len(fp_df), len(fn_df))

# Save error samples
fp_df.to_csv(OUTPUT_DIR / "false_positives.csv", index=False)
fn_df.to_csv(OUTPUT_DIR / "false_negatives.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 8. Grad-CAM implementation from scratch (no torchcam dependency)
# ──────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM for ResNet18 using layer4 (last residual block)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._fwd_hook_fn)
        self._bwd_hook = target_layer.register_full_backward_hook(self._bwd_hook_fn)

    def _fwd_hook_fn(self, module, input, output):
        self.activations = output.detach()

    def _bwd_hook_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = 1):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor).squeeze(-1)
        if target_class == 1:
            output.backward()
        else:
            (-output).backward()

        # weights = global average of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def generate_gradcam(model, input_tensor, target_class=1):
    """Generate Grad-CAM heatmap for a single input."""
    # Find last conv layer in ResNet18 → model.layer4
    target_layer = model.layer4
    gradcam = GradCAM(model, target_layer)

    input_t = input_tensor.unsqueeze(0)
    cam = gradcam.generate(input_t, target_class=target_class)
    gradcam.remove_hooks()

    return cam[0, 0]  # (H, W) — 2D heatmap


def visualize_gradcam(patch_tensor, cam, title, output_path, true_label=None, pred_label=None, prob=None):
    """Save a Grad-CAM visualization: RGB composite + heatmap overlay."""
    # Use bands B4(2), B3(1), B2(0) for RGB composite from month 0
    pt = patch_tensor.detach()
    rgb = np.stack([
        pt[2].numpy(),  # B4 (Red) - channel index 2 in first month
        pt[1].numpy(),  # B3 (Green)
        pt[0].numpy(),  # B2 (Blue)
    ], axis=-1)

    # Normalize to [0,1] for display
    rgb_vis = np.clip(rgb / 3000.0, 0, 1)  # Typical S2 range clip

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb_vis)
    axes[0].set_title("RGB Composite (Month 1)")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(rgb_vis)
    axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    info_parts = [title]
    if true_label is not None:
        info_parts.append(f"True: {'Deforest' if true_label==1 else 'No Deforest'}")
    if pred_label is not None:
        info_parts.append(f"Pred: {'Deforest' if pred_label==1 else 'No Deforest'}")
    if prob is not None:
        info_parts.append(f"Conf: {prob:.3f}")
    axes[2].set_title("\n".join(info_parts))
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 9. Generate Grad-CAM visualizations: 1 correct, 1 FP, 1 FN
# ──────────────────────────────────────────────────────────────────────────────
logger.info("Generating Grad-CAM visualizations ...")

def pick_representative(df_sub, sample_n=1):
    """Pick middle-confidence sample for Grad-CAM (most informative)."""
    if len(df_sub) == 0:
        return []
    # Sort by distance from 0.5 (most uncertain = most informative)
    df_sub = df_sub.copy()
    df_sub["uncertainty"] = abs(df_sub["probability"] - 0.5)
    df_sub = df_sub.sort_values("uncertainty", ascending=False)
    return df_sub.head(sample_n)["patch_id"].tolist()

# Get sample patch_ids
correct_samples = pick_representative(correct_df, 1)
fp_samples = pick_representative(fp_df, 1)
fn_samples = pick_representative(fn_df, 1)

# Find their indices in the dataset
patch_to_idx = {pid: i for i, pid in enumerate(all_patch_ids)}

model.eval()
for label, samples, error_type in [
    ("correct", correct_samples, "correct"),
    ("false_positive", fp_samples, "false_positive"),
    ("false_negative", fn_samples, "false_negative"),
]:
    if not samples:
        logger.warning("No %s samples found for Grad-CAM", error_type)
        continue

    pid = samples[0]
    idx = patch_to_idx[pid]
    row = error_df[error_df["patch_id"] == pid].iloc[0]

    # Load tensor
    tensor, true_label = test_ds[idx]
    tensor.requires_grad_(True)

    # Generate Grad-CAM
    cam = generate_gradcam(model, tensor, target_class=1)

    # Visualize
    title = error_type.replace("_", " ").title()
    visualize_gradcam(
        tensor, cam, title,
        OUTPUT_DIR / f"gradcam_{error_type}.png",
        true_label=true_label,
        pred_label=int(row["predicted_label"]),
        prob=float(row["probability"]),
    )
    logger.info("  Grad-CAM saved: %s (patch=%s, prob=%.3f)", error_type, pid, row["probability"])

# ──────────────────────────────────────────────────────────────────────────────
# 10. Save all results as JSON for the report
# ──────────────────────────────────────────────────────────────────────────────
results = {
    "model_checkpoint": str(BEST_MODEL_PATH),
    "checkpoint_epoch": int(ckpt["epoch"]),
    "best_val_f1": float(ckpt["best_val_f1"]),
    "test_samples": len(labels_np),
    "metrics_at_05": {k: v for k, v in metrics_05.items() if k != "confusion_matrix"},
    "confusion_matrix_05": metrics_05["confusion_matrix"],
    "metrics_at_best_threshold": {k: v for k, v in metrics_best.items() if k != "confusion_matrix"},
    "confusion_matrix_best": metrics_best["confusion_matrix"],
    "best_threshold": float(best_thresh["threshold"]),
    "threshold_analysis": threshold_results,
    "error_analysis": {
        "total_correct": int(len(correct_df)),
        "false_positives": int(len(fp_df)),
        "false_negatives": int(len(fn_df)),
        "fp_details": fp_df.to_dict(orient="records"),
        "fn_details": fn_df.to_dict(orient="records"),
    },
}

with open(OUTPUT_DIR / "test_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

logger.info("All results saved to %s", OUTPUT_DIR)
logger.info("Done.")
