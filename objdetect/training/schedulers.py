"""Learning-rate schedulers: step decay and cosine annealing."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    StepLR,
)


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    epochs: int,
    step_size: int = 3,
    gamma: float = 0.1,
    eta_min: float = 1e-5,
) -> LRScheduler:
    """Construct a scheduler by name: 'step' or 'cosine'."""
    if name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    raise ValueError(f"unknown scheduler '{name}', expected 'step' or 'cosine'")


def sample_lr_curve(
    name: str,
    epochs: int,
    base_lr: float = 0.005,
    **kwargs,
) -> list[float]:
    """Return the LR at each epoch for ``name``, for plotting/inspection."""
    import torch

    dummy = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy], lr=base_lr)
    scheduler = build_scheduler(name, optimizer, epochs=epochs, **kwargs)

    lrs = []
    for _ in range(epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs
