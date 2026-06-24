"""Run the LR-schedule experiments that fill the model report."""

import argparse
import json

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from objdetect import config
from objdetect.data.coco import CocoSubsetDataset, collate_detections
from objdetect.evaluation import evaluate_coco_map
from objdetect.models import build_detector
from objdetect.training.train import fine_tune


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=0.005)
    parser.add_argument("--max-batches", type=int, default=15)
    parser.add_argument("--max-eval-images", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_ds = CocoSubsetDataset(train=True)
    eval_ds = CocoSubsetDataset(train=False)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_detections,
    )

    results = {}
    histories = {}
    for schedule in ("cosine", "step"):
        print(f"\n=== Fine-tuning with {schedule} schedule ===")
        detector = build_detector("faster_rcnn", num_classes=len(train_ds.classes))
        history = fine_tune(
            detector.model, loader,
            epochs=args.epochs, scheduler_name=schedule,
            base_lr=args.base_lr, device=args.device,
            max_batches=args.max_batches,
        )
        histories[schedule] = history
        detector.set_class_names(
            [train_ds.label_to_name[i + 1] for i in range(len(train_ds.classes))]
        )
        metrics = evaluate_coco_map(
            detector.model, eval_ds, device=args.device,
            max_images=args.max_eval_images,
        )
        results[schedule] = {
            "final_loss": history.train_loss[-1],
            "metrics": metrics,
            "lr_curve": history.learning_rate,
            "loss_curve": history.train_loss,
        }
        print(f"{schedule}: mAP={metrics['mAP']:.3f}  final_loss={history.train_loss[-1]:.3f}")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_lr) = plt.subplots(1, 2, figsize=(13, 5))
    for schedule, hist in histories.items():
        epochs = range(1, len(hist.train_loss) + 1)
        ax_loss.plot(epochs, hist.train_loss, marker="o", label=schedule)
        ax_lr.plot(epochs, hist.learning_rate, marker="o", label=schedule)
    ax_loss.set(title="Training loss vs epoch", xlabel="Epoch", ylabel="Loss")
    ax_lr.set(title="Learning rate vs epoch", xlabel="Epoch", ylabel="LR")
    ax_loss.legend(); ax_lr.legend()
    ax_loss.grid(alpha=0.3); ax_lr.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "lr_experiment_comparison.png", dpi=120)

    out = config.REPORTS_DIR / "experiment_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out} and comparison figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
