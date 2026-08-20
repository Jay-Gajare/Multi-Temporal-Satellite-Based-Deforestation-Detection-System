"""
Optimized pipeline benchmark — measures data loading from .npy cache.
Outputs JSON for comparison with the original GeoTIFF benchmark.
"""
import sys
import time
import json
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from training.config import Config
from training.dataset import DeforestationDataset
from training.models import build_model
from training.trainer import Trainer

TEMP = Path(r"C:\Users\LENOVO\AppData\Local\Temp\opencode")

cfg = Config()
cfg.model.pretrained = False
cfg.model.architecture = "resnet18"
cfg.data.temporal_strategy = "temporal_stack"
cfg.train.amp = False
cfg.train.batch_size = 32
cfg.train.epochs = 1

results = {}

# ── 1. Dataset init ──────────────────────────────────────────────
print("=== 1. Dataset initialization (cache backend) ===")
t0 = time.perf_counter()
train_ds = DeforestationDataset(
    cfg.data.train_csv, cfg.data.labels_csv, cfg.data.patches_dir,
    temporal_strategy=cfg.data.temporal_strategy, use_cache=True,
)
t_init = time.perf_counter() - t0
results["dataset_init_seconds"] = round(t_init, 3)
results["dataset_size"] = len(train_ds)
results["in_channels"] = train_ds.in_channels
results["use_cache"] = train_ds.use_cache
print(f"  Loaded {len(train_ds)} samples in {t_init:.3f}s (in_channels={train_ds.in_channels}, cache={train_ds.use_cache})")

# ── 2. Single sample loading ─────────────────────────────────────
print("\n=== 2. Single sample loading (100 samples) ===")
sample_times = []
for i in range(100):
    t0 = time.perf_counter()
    x, y = train_ds[i]
    dt = time.perf_counter() - t0
    sample_times.append(dt)

avg_sample = statistics.mean(sample_times)
median_sample = statistics.median(sample_times)
results["sample_load_avg_ms"] = round(avg_sample * 1000, 2)
results["sample_load_median_ms"] = round(median_sample * 1000, 2)
results["sample_load_p95_ms"] = round(sorted(sample_times)[int(len(sample_times) * 0.95)] * 1000, 2)
results["sample_shape"] = list(x.shape)
results["sample_dtype"] = str(x.dtype)
print(f"  shape={list(x.shape)} dtype={x.dtype}")
print(f"  avg={avg_sample*1000:.1f}ms  median={median_sample*1000:.1f}ms  p95={results['sample_load_p95_ms']}ms")

# ── 3. Batch loading ────────────────────────────────────────────
print("\n=== 3. Batch loading (30 batches) ===")
loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
loader_iter = iter(loader)
batch_times = []
for i in range(30):
    t0 = time.perf_counter()
    bx, by = next(loader_iter)
    dt = time.perf_counter() - t0
    batch_times.append(dt)

avg_batch = statistics.mean(batch_times)
results["batch_load_avg_ms"] = round(avg_batch * 1000, 2)
results["batch_load_median_ms"] = round(statistics.median(batch_times) * 1000, 2)
results["batch_size"] = cfg.train.batch_size
results["batch_shape"] = list(bx.shape)
print(f"  batch_shape={list(bx.shape)}  avg={avg_batch*1000:.1f}ms  median={statistics.median(batch_times)*1000:.1f}ms")

# ── 4. System info ──────────────────────────────────────────────
print("\n=== 4. System info ===")
cpu_count = psutil.cpu_count()
cpu_freq = psutil.cpu_freq()
ram = psutil.virtual_memory()
results["cpu_count"] = cpu_count
results["cpu_freq_max_mhz"] = round(cpu_freq.max, 0) if cpu_freq else None
results["ram_total_gb"] = round(ram.total / (1024**3), 1)
results["ram_available_gb"] = round(ram.available / (1024**3), 1)
results["ram_used_pct"] = ram.percent
results["device"] = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  CPU: {cpu_count} cores, {cpu_freq.max:.0f} MHz")
print(f"  RAM: {ram.total/(1024**3):.1f} GB total, {ram.available/(1024**3):.1f} GB available")

# ── 5. Full epoch timing ────────────────────────────────────────
print("\n=== 5. Full epoch timing (1 epoch, train + val) ===")
neg, pos = train_ds.class_counts()
total = neg + pos
w_neg = total / (2.0 * neg) if neg > 0 else 1.0
w_pos = total / (2.0 * pos) if pos > 0 else 1.0
class_weights = torch.tensor([w_pos / w_neg], dtype=torch.float32)

val_ds = DeforestationDataset(
    cfg.data.val_csv, cfg.data.labels_csv, cfg.data.patches_dir,
    temporal_strategy=cfg.data.temporal_strategy, use_cache=True,
)
train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

model = build_model("resnet18", train_ds.in_channels, pretrained=False, dropout=0.3)
trainer = Trainer(model, train_loader, val_loader, cfg, class_weights)

t0 = time.perf_counter()
history = trainer.fit()
epoch_time = time.perf_counter() - t0
results["one_epoch_seconds"] = round(epoch_time, 1)
results["epochs_ran"] = len(history)
if history:
    results["last_epoch_val_f1"] = round(history[-1]["val_f1"], 4)
    results["last_epoch_val_auc"] = round(history[-1]["val_auc"], 4)
print(f"  1 epoch completed in {epoch_time:.1f}s")

# ── 6. Projections ──────────────────────────────────────────────
results["est_10_epochs_min"] = round(epoch_time * 10 / 60, 1)
results["est_50_epochs_min"] = round(epoch_time * 50 / 60, 1)
results["est_50_epochs_hours"] = round(epoch_time * 50 / 3600, 2)

# ── 7. Memory footprint ────────────────────────────────────────
process = psutil.Process()
results["process_rss_gb"] = round(process.memory_info().rss / (1024**3), 2)
cache_size_gb = Path(cfg.data.patches_dir).parent.joinpath("cache", "temporal_stack_data.npy").stat().st_size / (1024**3)
results["cache_file_gb"] = round(cache_size_gb, 2)
print(f"  Process RSS: {results['process_rss_gb']:.2f} GB | Cache file: {results['cache_file_gb']:.2f} GB")

# ── Save ─────────────────────────────────────────────────────────
out_path = TEMP / "benchmark_cached_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(json.dumps(results, indent=2))
