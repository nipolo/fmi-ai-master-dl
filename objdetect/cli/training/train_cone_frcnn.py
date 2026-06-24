"""Fine-tune Faster R-CNN on the traffic-cone dataset (Req. 3 deep dive).

This mirrors the YOLO cone experiment (§7 of the model report) for the
*two-stage* detector: it fine-tunes Faster R-CNN on the same seeded 80/20 cone
split, then evaluates COCO mAP on the held-out val images — giving a real,
converged two-stage-vs-one-stage comparison on a custom out-of-COCO class.

Unlike a COCO-subset fine-tune, this earns its compute: the cone class is a
genuine domain shift, so training actually helps.

Prerequisite (run once):
    uv run python -m objdetect.cli.data.prepare_cone_coco      # YOLO -> COCO labels

Run the fine-tune (this is the training step — sized for your machine):
    # Safe everywhere (CPU): slower but never exhausts GPU memory.
    uv run python -m objdetect.cli.training.train_cone_frcnn --device cpu --epochs 15

    # Faster on Apple Silicon (MPS). Faster R-CNN training is memory-heavy, so
    # this defaults to batch 1 and a 512 px input. If MPS still over-commits the
    # 16 GB unified memory and the machine swaps, cap it so it errors instead:
    #   PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 uv run python -m objdetect.cli.training.train_cone_frcnn --epochs 20
    # ...and if it still OOMs, fall back to --device cpu.

    # Quick smoke to confirm it runs at all (a few batches):
    uv run python -m objdetect.cli.training.train_cone_frcnn --device cpu --epochs 1 --max-batches 5

Writes weights to DATA/checkpoints/ and results.json + training.png to reports/.
To use the result in the app, publish the weights to the path it loads:
    cp DATA/checkpoints/faster_rcnn_cone_cosine.pth DATA/weights/faster_rcnn_cone.pth
"""

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
    """Save a loss-vs-epoch / LR-vs-epoch figure for the report."""
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
    # Shrink the model's internal resize to cut activation memory (the cone
    # images are <=1024 px and the class is large/distinct, so 512 is ample).
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

    # Persist weights (checkpoints), a results record + training curve (reports).
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
