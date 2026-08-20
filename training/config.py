"""
Training configuration — all hyperparameters and paths in one place.

Override via CLI flags or by editing this file directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "exports"
PATCHES_DIR = EXPORT_DIR / "patches"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass
class DataConfig:
    labels_csv: Path = EXPORT_DIR / "patch_labels.csv"
    train_csv: Path = EXPORT_DIR / "splits" / "train.csv"
    val_csv: Path = EXPORT_DIR / "splits" / "val.csv"
    test_csv: Path = EXPORT_DIR / "splits" / "test.csv"
    patches_dir: Path = PATCHES_DIR

    n_bands: int = 9
    n_months: int = 12
    patch_size: int = 64

    temporal_strategy: str = "temporal_stack"  # single_month | average | temporal_stack
    best_month: int = 7  # used only for single_month strategy (1-12)

    num_workers: int = 0  # 0 = main process (Windows-safe)
    pin_memory: bool = True


@dataclass
class ModelConfig:
    architecture: str = "resnet18"  # resnet18 | resnet50 | efficientnet_b0
    pretrained: bool = True
    dropout: float = 0.3


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    use_class_weights: bool = True
    max_grad_norm: float = 1.0

    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda

    amp: bool = True  # mixed precision (no effect on CPU, harmless)
    checkpoint_every: int = 5

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    run_name: str = "run_01"

    @property
    def output_dir(self) -> Path:
        d = MODELS_DIR / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def checkpoint_dir(self) -> Path:
        d = self.output_dir / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d
