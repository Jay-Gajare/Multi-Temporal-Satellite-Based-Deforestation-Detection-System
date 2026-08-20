"""
Visualization utilities for training curves, confusion matrix, ROC curve.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: list[dict], output_dir: Path) -> None:
    """Plot training and validation loss + F1 over epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_f1 = [h["train_f1"] for h in history]
    val_f1 = [h["val_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "b-", label="Train Loss")
    ax1.plot(epochs, val_loss, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_f1, "b-", label="Train F1")
    ax2.plot(epochs, val_f1, "r-", label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Training & Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: list[list[int]], output_dir: Path) -> None:
    """Plot confusion matrix heatmap."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["No Deforestation", "Deforestation"],
        yticklabels=["No Deforestation", "Deforestation"],
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix",
    )
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=16,
            )
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(all_probs: np.ndarray, all_labels: np.ndarray, output_dir: Path) -> None:
    """Plot ROC curve with AUC score."""
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_learning_rate(history: list[dict], output_dir: Path) -> None:
    """Plot learning rate schedule."""
    epochs = [h["epoch"] for h in history]
    lrs = [h["learning_rate"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, lrs, "g-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
