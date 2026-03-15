"""Shared helpers: seed, project root, and path utilities."""

import os
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_checkpoint_path(model_name: str, mode: str) -> Path:
    """Return checkpoint directory path for a given model and input mode."""
    return PROJECT_ROOT / "reports" / "checkpoints" / model_name / mode


def build_metrics_path(model_name: str, mode: str) -> Path:
    """Return metrics output directory path for a given model and input mode."""
    return PROJECT_ROOT / "src" / "outputs" / "metrics" / model_name / mode
