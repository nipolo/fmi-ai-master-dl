"""Fine-tune Faster R-CNN on the traffic-cone dataset."""

import argparse
import json

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from objdetect import config
from objdetect.data.coco import collate_detections
from objdetect.data.cone import cone_dataset
from objdetect.evaluation import evaluate_coco_map
from objdetect.models import build_detector
from objdetect.training.train import fine_tune


def _plot_training(history, scheduler: str, out_path) -> None:
    epochs = range(1, len(history.train_loss) + 1)
    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(
        epochs, history.train_loss, "o-", color="indianred", label="train loss"
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss", color="indianred")
    ax_loss.tick_params(axis="y", labelcolor="indianred")

    ax_lr = ax_loss.twinx()
    ax_lr.plot(
        epochs, history.learning_rate, "s--", color="steelblue", label="learning rate"
    )
    ax_lr.set_ylabel("Learning rate", color="steelblue")
    ax_lr.tick_params(axis="y", labelcolor="steelblue")

    plt.title(f"Faster R-CNN cone fine-tune — {scheduler} schedule")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="keep small: Faster R-CNN training is memory-heavy on MPS",
    )
    parser.add_argument("--base-lr", type=float, default=0.005)
    parser.add_argument(
        "--min-size",
        type=int,
        default=512,
        help="resize shorter image side to this (lower = less memory)",
    )
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="cap batches/epoch for a quick smoke run",
    )
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument(
        "--device", default=None, help="cpu | mps | cuda (default: auto)"
    )
    args = parser.parse_args()

    train_ds = cone_dataset("train", train=True)
    val_ds = cone_dataset("val", train=False)
    print(
        f"cone train images: {len(train_ds)}  val images: {len(val_ds)}  "
        f"class: {config.CONE_CLASS_NAMES}"
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_detections,
    )

    detector = build_detector("faster_rcnn", num_classes=len(config.CONE_CLASS_NAMES))
    detector.model.transform.min_size = (args.min_size,)
    detector.model.transform.max_size = args.max_size

    history = fine_tune(
        detector.model,
        loader,
        epochs=args.epochs,
        scheduler_name=args.scheduler,
        base_lr=args.base_lr,
        device=args.device,
        max_batches=args.max_batches,
    )

    metrics = evaluate_coco_map(
        detector.model,
        val_ds,
        device=args.device,
        max_images=args.max_eval_images,
    )
    print("val metrics:", json.dumps(metrics, indent=2))

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = config.CHECKPOINTS_DIR / f"faster_rcnn_cone_{args.scheduler}.pth"
    torch.save(detector.model.state_dict(), weights_path)

    results = {
        "model": "faster_rcnn",
        "dataset": "traffic_cone (held-out val split)",
        "scheduler": args.scheduler,
        "epochs": args.epochs,
        "base_lr": args.base_lr,
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "metrics": metrics,
        "history": history.__dict__,
    }
    results_path = config.REPORTS_DIR / "cone_frcnn_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    figure_path = config.FIGURES_DIR / "cone_frcnn_training.png"
    _plot_training(history, args.scheduler, figure_path)

    print(f"saved weights  -> {weights_path}")
    print(f"saved results  -> {results_path}")
    print(f"saved figure   -> {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
