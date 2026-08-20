"""Quick end-to-end pipeline test with 50 samples."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.config import Config
from training.dataset import DeforestationDataset
from training.models import build_model
from training.trainer import Trainer
from training.metrics import compute_metrics, probs_from_logits, preds_from_logits
from training.visualize import (
    plot_training_curves, plot_confusion_matrix,
    plot_roc_curve, plot_learning_rate,
)

TEMP = Path(r"C:\Users\LENOVO\AppData\Local\Temp\opencode")

cfg = Config()
cfg.data.train_csv = TEMP / "train_mini.csv"
cfg.data.val_csv = TEMP / "val_mini.csv"
cfg.train.epochs = 3
cfg.train.batch_size = 16
cfg.run_name = "test_run"
cfg.model.pretrained = False
cfg.train.amp = False

train_ds = DeforestationDataset(
    cfg.data.train_csv, cfg.data.labels_csv, cfg.data.patches_dir,
    temporal_strategy="temporal_stack",
)
val_ds = DeforestationDataset(
    cfg.data.val_csv, cfg.data.labels_csv, cfg.data.patches_dir,
    temporal_strategy="temporal_stack",
)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, In-channels: {train_ds.in_channels}")

model = build_model("resnet18", train_ds.in_channels, pretrained=False, dropout=0.3)

train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

neg, pos = train_ds.class_counts()
cw = torch.tensor([neg / pos]) if pos > 0 else None

trainer = Trainer(model, train_loader, val_loader, cfg, cw)
t0 = time.time()
history = trainer.fit()
elapsed = time.time() - t0
print(f"Training done: {len(history)} epochs in {elapsed:.0f}s")

model.eval()
all_logits, all_labels, all_probs = [], [], []
with torch.no_grad():
    for x, y in val_loader:
        logits = model(x).squeeze(-1).numpy()
        all_logits.append(logits)
        all_labels.append(y.numpy())

logits_np = np.concatenate(all_logits)
labels_np = np.concatenate(all_labels)
probs_np = probs_from_logits(logits_np)
preds_np = preds_from_logits(logits_np)
metrics = compute_metrics(preds_np, labels_np, probs_np)
print(f"Val F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")
print(f"Confusion matrix: {metrics['confusion_matrix']}")

output_dir = cfg.output_dir
plot_training_curves(history, output_dir)
plot_confusion_matrix(metrics["confusion_matrix"], output_dir)
plot_roc_curve(probs_np, labels_np, output_dir)
plot_learning_rate(history, output_dir)
print(f"Plots saved to {output_dir}")

files = [
    "training_curves.png", "confusion_matrix.png",
    "roc_curve.png", "learning_rate_curve.png",
    "training_history.csv", "metrics.json",
    "best_model.pth", "last_model.pth",
]
for f in files:
    p = output_dir / f
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    print(f"  {f}: {'OK' if exists else 'MISSING'} ({size} bytes)")

print("Pipeline test PASSED")
