"""Plot the two LR schedules required by the assignment (Requirement 5)."""

import argparse

import matplotlib.pyplot as plt

from objdetect import config
from objdetect.training.schedulers import sample_lr_curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--base-lr", type=float, default=0.005)
    args = parser.parse_args()

    epochs = list(range(args.epochs))
    cosine = sample_lr_curve("cosine", args.epochs, base_lr=args.base_lr)
    step = sample_lr_curve("step", args.epochs, base_lr=args.base_lr)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, cosine, marker="o", label="Cosine annealing")
    plt.plot(epochs, step, marker="s", label="Step decay")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning-rate schedules over epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "lr_schedules.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
