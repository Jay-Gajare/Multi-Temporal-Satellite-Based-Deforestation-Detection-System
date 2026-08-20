"""
Metrics computation for binary deforestation classification.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    all_probs: np.ndarray | None = None,
) -> dict:
    """
    Compute classification metrics from predictions and labels.

    Parameters
    ----------
    all_preds : ndarray of shape (N,)
        Binary predictions (0 or 1).
    all_labels : ndarray of shape (N,)
        Ground truth labels (0 or 1).
    all_probs : ndarray of shape (N,), optional
        Predicted probabilities for positive class.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc, confusion_matrix, report
    """
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)

    auc = 0.0
    if all_probs is not None and len(np.unique(all_labels)) > 1:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def preds_from_logits(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert raw logits to binary predictions via sigmoid + threshold."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= threshold).astype(int)


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits to probabilities via sigmoid."""
    return 1.0 / (1.0 + np.exp(-logits))
