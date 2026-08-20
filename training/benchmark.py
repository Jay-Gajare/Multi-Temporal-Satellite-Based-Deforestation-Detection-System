"""
Training pipeline benchmark — measures every stage of the data/compute pipeline.
Outputs structured JSON for report generation.
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

results = {}

# ── 1. Dataset init ──────────────────────────────────────────────
print("=== 1. Dataset initialization ===")
t0 = time.perf_counter()
train_ds = DeforestationDataset(
    cfg.data.train_csv, cfg.data.labels_csv, cfg.data.patches_dir,
    temporal_strategy=cfg.data.temporal_strategy,
)
t_init = time.perf_counter() - t0
results["dataset_init_seconds"] = round(t_init, 3)
results["dataset_size"] = len(train_ds)
results["in_channels"] = train_ds.in_channels
print(f"  Loaded {len(train_ds)} samples in {t_init:.3f}s (in_channels={train_ds.in_channels})")

neg, pos = train_ds.class_counts()
results["class_neg"] = neg
results["class_pos"] = pos
results["class_ratio"] = round(pos / neg, 3) if neg > 0 else None
print(f"  Classes: neg={neg} pos={pos} ratio={pos/neg:.3f}")

# ── 2. Single sample loading ─────────────────────────────────────
print("\n=== 2. Single sample loading (20 samples) ===")
sample_times = []
for i in range(min(20, len(train_ds))):
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
print("\n=== 3. Batch loading (20 batches) ===")
loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
loader_iter = iter(loader)
batch_times = []
for i in range(20):
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
if torch.cuda.is_available():
    results["gpu_name"] = torch.cuda.get_device_name(0)
    results["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
print(f"  CPU: {cpu_count} cores, {cpu_freq.max:.0f} MHz")
print(f"  RAM: {ram.total/(1024**3):.1f} GB total, {ram.available/(1024**3):.1f} GB available")
print(f"  Device: {results['device']}")

# ── 5. Model info ───────────────────────────────────────────────
print("\n=== 5. Model info ===")
model = build_model("resnet18", train_ds.in_channels, pretrained=False, dropout=0.3)
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
results["model_params_m"] = round(n_params / 1e6, 2)
results["model_trainable_params_m"] = round(n_trainable / 1e6, 2)
print(f"  Params: {n_params/1e6:.2f}M (trainable: {n_trainable/1e6:.2f}M)")

# ── 6. Forward pass timing ──────────────────────────────────────
print("\n=== 6. Forward pass timing (50 passes) ===")
model.eval()
dummy_input = bx  # reuse batch from above
fwd_times = []
with torch.no_grad():
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model(dummy_input)
        dt = time.perf_counter() - t0
        fwd_times.append(dt)

avg_fwd = statistics.mean(fwd_times)
results["forward_pass_avg_ms"] = round(avg_fwd * 1000, 2)
results["forward_pass_median_ms"] = round(statistics.median(fwd_times) * 1000, 2)
print(f"  avg={avg_fwd*1000:.2f}ms  median={statistics.median(fwd_times)*1000:.2f}ms")

# ── 7. Forward + backward pass timing ───────────────────────────
print("\n=== 7. Forward + backward pass timing (20 passes) ===")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
fwd_bwd_times = []
for i in range(20):
    t0 = time.perf_counter()
    out = model(dummy_input).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, by.float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    dt = time.perf_counter() - t0
    fwd_bwd_times.append(dt)

avg_fwd_bwd = statistics.mean(fwd_bwd_times)
avg_bwd = avg_fwd_bwd - avg_fwd
results["fwd_bwd_pass_avg_ms"] = round(avg_fwd_bwd * 1000, 2)
results["backward_pass_avg_ms"] = round(avg_bwd * 1000, 2)
print(f"  fwd+bwd avg={avg_fwd_bwd*1000:.2f}ms  backward only={avg_bwd*1000:.2f}ms")

# ── 8. Full epoch timing ────────────────────────────────────────
print("\n=== 8. Full epoch timing (1 epoch, train + val) ===")
neg, pos = train_ds.class_counts()
total = neg + pos
w_neg = total / (2.0 * neg) if neg > 0 else 1.0
w_pos = total / (2.0 * pos) if pos > 0 else 1.0
class_weights = torch.tensor([w_pos / w_neg], dtype=torch.float32)

val_ds = DeforestationDataset(
    cfg.data.val_csv, cfg.data.labels_csv, cfg.data.patches_dir,
    temporal_strategy=cfg.data.temporal_strategy,
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
    results["last_epoch_train_loss"] = round(history[-1]["train_loss"], 4)
    results["last_epoch_val_loss"] = round(history[-1]["val_loss"], 4)
    results["last_epoch_val_f1"] = round(history[-1]["val_f1"], 4)
    results["last_epoch_val_auc"] = round(history[-1]["val_auc"], 4)
print(f"  1 epoch completed in {epoch_time:.1f}s")

# ── 9. Projections ──────────────────────────────────────────────
results["est_10_epochs_min"] = round(epoch_time * 10 / 60, 1)
results["est_50_epochs_min"] = round(epoch_time * 50 / 60, 1)
results["est_50_epochs_hours"] = round(epoch_time * 50 / 3600, 2)

# ── 10. Bottleneck analysis ─────────────────────────────────────
print("\n=== 9. Bottleneck analysis ===")
data_pct = (avg_sample * len(train_ds)) / epoch_time * 100
compute_pct = (avg_fwd_bwd * len(train_loader)) / epoch_time * 100
results["pct_data_loading"] = round(data_pct, 1)
results["pct_compute"] = round(compute_pct, 1)
print(f"  Data loading: ~{data_pct:.0f}% of epoch time")
print(f"  Compute (fwd+bwd): ~{compute_pct:.0f}% of epoch time")

# ── 11. Memory footprint ────────────────────────────────────────
process = psutil.Process()
results["process_rss_gb"] = round(process.memory_info().rss / (1024**3), 2)
print(f"  Process RSS: {results['process_rss_gb']:.2f} GB")

# ── Save ─────────────────────────────────────────────────────────
out_path = TEMP / "benchmark_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(json.dumps(results, indent=2))
