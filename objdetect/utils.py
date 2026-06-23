"""Small shared helpers: device selection and reproducible seeding."""

import random

import numpy as np
import torch

from objdetect import config


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Pick the best available device unless one is explicitly given.

    Preference order: explicit argument, then CUDA (if present), then Apple
    Silicon MPS (local Mac), then CPU.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = config.SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
