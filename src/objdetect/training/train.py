"""Fine-tuning loop for the torchvision Faster R-CNN detector.

Kept deliberately small and explicit so it is presentable: one epoch loop,
one optimizer, one scheduler, per-epoch LR and loss recorded into a history
that the report and the LR-schedule plots consume.
"""

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from objdetect.training.schedulers import build_scheduler
from objdetect.utils import resolve_device, seed_everything


@dataclass
class TrainHistory:
    """Per-epoch record of the quantities we plot and report."""

    learning_rate: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Run one training epoch and return the mean total loss.

    torchvision detection models return a dict of losses when called in train
    mode with targets; we sum them, as is standard for Faster R-CNN.
    ``max_batches`` caps the epoch for quick local smoke runs.
    """
    model.train()
    running, seen = 0.0, 0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += float(loss)
        seen += 1
    return running / max(seen, 1)


def fine_tune(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int,
    scheduler_name: str,
    base_lr: float = 0.005,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    seed: int = 42,
) -> TrainHistory:
    """Fine-tune ``model`` and return the training history.

    The optimizer is SGD with momentum (the torchvision detection default);
    ``scheduler_name`` selects 'cosine' or 'step', which is the knob the
    LR-schedule experiment varies while holding everything else fixed.
    """
    seed_everything(seed)
    device = resolve_device(device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = build_scheduler(scheduler_name, optimizer, epochs=epochs)

    history = TrainHistory()
    for epoch in range(epochs):
        history.learning_rate.append(optimizer.param_groups[0]["lr"])
        loss = train_one_epoch(model, loader, optimizer, device, max_batches)
        history.train_loss.append(loss)
        scheduler.step()
        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"lr={history.learning_rate[-1]:.5f}  loss={loss:.4f}"
        )
    return history
