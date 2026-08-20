"""
Main entry point for training.

Usage:
    python -m training.train
    python -m training.train --arch resnet50 --strategy average --epochs 30
    python -m training.train --resume models/run_01/checkpoints/epoch_010.pth
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.config import Config
from training.dataset import DeforestationDataset
from training.models import build_model
from training.trainer import Trainer
from training.visualize import (
    plot_confusion_matrix,
    plot_learning_rate,
    plot_roc_curve,
    plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train deforestation classifier")
    p.add_argument("--arch", default=None, help="resnet18 | resnet50 | efficientnet_b0")
    p.add_argument("--strategy", default=None, help="single_month | average | temporal_stack")
    p.add_argument("--best-month", type=int, default=None, help="Month 1-12 for single_month")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="Path to checkpoint .pth")
    p.add_argument("--device", default=None, help="auto | cpu | cuda")
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    # Override from CLI
    if args.arch:
        cfg.model.architecture = args.arch
    if args.strategy:
        cfg.data.temporal_strategy = args.strategy
    if args.best_month:
        cfg.data.best_month = args.best_month
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.learning_rate = args.lr
    if args.dropout:
        cfg.model.dropout = args.dropout
    if args.run_name:
        cfg.run_name = args.run_name
    if args.no_pretrained:
        cfg.model.pretrained = False
    if args.device:
        cfg.train.device = args.device

    set_seed(cfg.train.seed)
    device = cfg.train.resolved_device
    logger.info("Device: %s", device)
    logger.info("Config: arch=%s strategy=%s epochs=%d bs=%d lr=%s",
                cfg.model.architecture, cfg.data.temporal_strategy,
                cfg.train.epochs, cfg.train.batch_size, cfg.train.learning_rate)

    # ── Data ────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_ds = DeforestationDataset(
        cfg.data.train_csv, cfg.data.labels_csv, cfg.data.patches_dir,
        temporal_strategy=cfg.data.temporal_strategy,
        best_month=cfg.data.best_month,
    )
    val_ds = DeforestationDataset(
        cfg.data.val_csv, cfg.data.labels_csv, cfg.data.patches_dir,
        temporal_strategy=cfg.data.temporal_strategy,
        best_month=cfg.data.best_month,
    )
    in_channels = train_ds.in_channels
    logger.info("In-channels: %d (strategy=%s)", in_channels, cfg.data.temporal_strategy)
    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device == "cuda",
    )

    # ── Class weights ──────────────────────────────────────────────
    class_weights = None
    if cfg.train.use_class_weights:
        neg, pos = train_ds.class_counts()
        total = neg + pos
        w_neg = total / (2.0 * neg) if neg > 0 else 1.0
        w_pos = total / (2.0 * pos) if pos > 0 else 1.0
        class_weights = torch.tensor([w_pos / w_neg], dtype=torch.float32)
        logger.info("Class weights: neg=%.2f pos=%.2f (ratio=%.2f)", w_neg, w_pos, w_pos / w_neg)

    # ── Model ──────────────────────────────────────────────────────
    logger.info("Building model: %s", cfg.model.architecture)
    model = build_model(
        cfg.model.architecture, in_channels,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    )

    # ── Trainer ────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, cfg, class_weights)

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # ── Train ──────────────────────────────────────────────────────
    t0 = time.time()
    history = trainer.fit()
    total_time = time.time() - t0
    logger.info("Training complete in %.0fs (%.1f min)", total_time, total_time / 60)

    # ── Visualizations ─────────────────────────────────────────────
    logger.info("Generating visualizations...")
    if history:
        plot_training_curves(history, cfg.output_dir)
        plot_learning_rate(history, cfg.output_dir)

    # Reload best model for final evaluation
    best_path = cfg.output_dir / "best_model.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded best model from epoch %d", ckpt["epoch"])

    model.eval()
    all_logits, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs).squeeze(-1).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    preds_np = (probs_np >= 0.5).astype(int)

    from training.metrics import compute_metrics
    val_metrics = compute_metrics(preds_np, labels_np, probs_np)

    plot_confusion_matrix(val_metrics["confusion_matrix"], cfg.output_dir)
    plot_roc_curve(probs_np, labels_np, cfg.output_dir)

    # ── Save metrics ───────────────────────────────────────────────
    metrics_out = {
        "config": {
            "architecture": cfg.model.architecture,
            "temporal_strategy": cfg.data.temporal_strategy,
            "in_channels": in_channels,
            "batch_size": cfg.train.batch_size,
            "learning_rate": cfg.train.learning_rate,
            "epochs": cfg.train.epochs,
            "weight_decay": cfg.train.weight_decay,
            "dropout": cfg.model.dropout,
            "seed": cfg.train.seed,
            "device": device,
            "pretrained": cfg.model.pretrained,
        },
        "best_val_metrics": {
            k: v for k, v in val_metrics.items() if k != "report"
        },
        "training_time_seconds": round(total_time, 1),
        "total_epochs_run": len(history),
        "best_epoch": max(history, key=lambda h: h["val_f1"])["epoch"] if history else 0,
        "best_val_f1": trainer.best_val_f1,
    }
    with open(cfg.output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Architecture : {cfg.model.architecture}")
    print(f"  Strategy     : {cfg.data.temporal_strategy}")
    print(f"  In-channels  : {in_channels}")
    print(f"  Epochs       : {len(history)}")
    print(f"  Best epoch   : {metrics_out['best_epoch']}")
    print(f"  Best val F1  : {trainer.best_val_f1:.4f}")
    print(f"  Best val AUC : {val_metrics['auc']:.4f}")
    print(f"  Val accuracy : {val_metrics['accuracy']:.4f}")
    print(f"  Time         : {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Device       : {device}")
    print(f"  Output       : {cfg.output_dir}")
    print("=" * 60)

    return history, val_metrics, total_time


if __name__ == "__main__":
    main()
