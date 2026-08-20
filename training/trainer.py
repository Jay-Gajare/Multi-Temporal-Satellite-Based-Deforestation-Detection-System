"""
Training loop with AMP, early stopping, checkpointing, and resume.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import compute_metrics, preds_from_logits, probs_from_logits

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Full training loop for binary deforestation classification.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.device = torch.device(config.train.resolved_device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=class_weights.to(self.device) if class_weights is not None else None,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.train.epochs - config.train.warmup_epochs,
            eta_min=config.train.min_lr,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda" and config.train.amp))
        self.early_stopping = EarlyStopping(
            patience=config.train.early_stopping_patience,
            min_delta=config.train.early_stopping_min_delta,
        )

        self.output_dir = config.output_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.history: list[dict] = []
        self.best_val_f1: float = -1.0
        self.start_epoch: int = 0

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.float().to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=(self.device.type == "cuda" and self.config.train.amp),
            ):
                logits = self.model(inputs).squeeze(-1)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        logits_np = np.concatenate(all_logits)
        labels_np = np.concatenate(all_labels)
        preds = preds_from_logits(logits_np)
        probs = probs_from_logits(logits_np)
        metrics = compute_metrics(preds, labels_np, probs)
        metrics["train_loss"] = avg_loss
        return metrics

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        total_loss = 0.0
        n_batches = 0

        for inputs, labels in tqdm(self.val_loader, desc="       [val]  ", leave=False):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.float().to(self.device, non_blocking=True)

            logits = self.model(inputs).squeeze(-1)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        logits_np = np.concatenate(all_logits)
        labels_np = np.concatenate(all_labels)
        preds = preds_from_logits(logits_np)
        probs = probs_from_logits(logits_np)
        metrics = compute_metrics(preds, labels_np, probs)
        metrics["val_loss"] = avg_loss
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_f1": self.best_val_f1,
            "config": {
                "architecture": self.config.model.architecture,
                "temporal_strategy": self.config.data.temporal_strategy,
                "batch_size": self.config.train.batch_size,
                "learning_rate": self.config.train.learning_rate,
            },
        }
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pth"
        torch.save(state, path)

        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(state, best_path)
            logger.info("  Best model saved (F1=%.4f)", metrics["f1"])

        last_path = self.output_dir / "last_model.pth"
        torch.save(state, last_path)

    def _warmup(self, epoch: int) -> None:
        """Linear warmup for first N epochs."""
        if epoch < self.config.train.warmup_epochs:
            warmup_factor = (epoch + 1) / self.config.train.warmup_epochs
            base_lr = self.config.train.learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = base_lr * warmup_factor

    def fit(self) -> list[dict]:
        """Run the full training loop."""
        logger.info(
            "Training on %s | epochs=%d | bs=%d | lr=%s",
            self.device, self.config.train.epochs,
            self.config.train.batch_size, self.config.train.learning_rate,
        )

        t_start = time.time()

        for epoch in range(self.start_epoch + 1, self.config.train.epochs + 1):
            # Warmup or cosine schedule
            if epoch <= self.config.train.warmup_epochs:
                self._warmup(epoch)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Train
            train_metrics = self._train_one_epoch(epoch)

            # Validate
            val_metrics = self._validate()

            # Record history
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"],
                "train_acc": train_metrics["accuracy"],
                "val_acc": val_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "train_precision": train_metrics["precision"],
                "val_precision": val_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_recall": val_metrics["recall"],
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "learning_rate": current_lr,
            }
            self.history.append(record)

            # Checkpoint
            is_best = val_metrics["f1"] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics["f1"]
            if epoch % self.config.train.checkpoint_every == 0 or is_best:
                self._save_checkpoint(epoch, val_metrics, is_best)

            # Log
            elapsed = time.time() - t_start
            logger.info(
                "Epoch %3d | loss: %.4f/%.4f | F1: %.4f/%.4f | AUC: %.4f/%.4f | lr: %.2e | %.0fs",
                epoch,
                train_metrics["train_loss"], val_metrics["val_loss"],
                train_metrics["f1"], val_metrics["f1"],
                train_metrics["auc"], val_metrics["auc"],
                current_lr, elapsed,
            )
            tqdm.write(
                f"Epoch {epoch:3d} | "
                f"loss: {train_metrics['train_loss']:.4f}/{val_metrics['val_loss']:.4f} | "
                f"F1: {train_metrics['f1']:.4f}/{val_metrics['f1']:.4f} | "
                f"AUC: {train_metrics['auc']:.4f}/{val_metrics['auc']:.4f} | "
                f"lr: {current_lr:.2e}"
            )

            # Early stopping
            if self.early_stopping.step(val_metrics["f1"]):
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Save history CSV
        # Final last_model.pth if not already saved this epoch
        last_path = self.output_dir / "last_model.pth"
        if not last_path.exists() or self.history[-1]["epoch"] != self.config.train.epochs:
            state = {
                "epoch": self.history[-1]["epoch"],
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_f1": self.best_val_f1,
                "config": {
                    "architecture": self.config.model.architecture,
                    "temporal_strategy": self.config.data.temporal_strategy,
                    "batch_size": self.config.train.batch_size,
                    "learning_rate": self.config.train.learning_rate,
                },
            }
            torch.save(state, last_path)

        self._save_history()
        return self.history

    def _save_history(self) -> None:
        path = self.output_dir / "training_history.csv"
        if not self.history:
            return
        keys = self.history[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
        logger.info("History saved to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"]
        self.best_val_f1 = ckpt.get("best_val_f1", -1.0)
        logger.info("Resumed from epoch %d (best_f1=%.4f)", self.start_epoch, self.best_val_f1)
