"""Learning-rate schedulers required by the assignment.

Requirement 5 asks us to demonstrate changing the learning rate as epochs
increase, specifically:

- **step decay** (:class:`torch.optim.lr_scheduler.StepLR`): the LR is held
  constant, then multiplied by ``gamma`` every ``step_size`` epochs — a
  piecewise-constant staircase,
- **cosine annealing**
  (:class:`torch.optim.lr_scheduler.CosineAnnealingLR`): the LR follows a
  half cosine from the initial value down to ``eta_min`` over ``T_max``
  epochs — a smooth decay that spends longer at high and low extremes.

Both are thin wrappers so the training script and the report use identical
configuration, and :func:`sample_lr_curve` lets us plot the schedules without
running a real training.
"""

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
    """Return the LR at each epoch for ``name``, for plotting/inspection.

    Builds a throwaway optimizer over a single dummy parameter and steps the
    scheduler ``epochs`` times, recording the LR before each step. No model or
    data is needed, which makes the schedule curves trivially testable.
    """
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
