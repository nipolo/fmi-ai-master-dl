"""Fine-tune Faster R-CNN on the COCO class subset.

Usage:
  uv run python scripts/train.py --scheduler cosine --epochs 6
  uv run python scripts/train.py --scheduler step   --epochs 6 --max-batches 20

``--max-batches`` caps batches per epoch for a quick local smoke run; omit it
for a full run over the whole subset.
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from objdetect import config
from objdetect.data.coco import CocoSubsetDataset, collate_detections
from objdetect.models import build_detector
from objdetect.training.train import fine_tune


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=0.005)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dataset = CocoSubsetDataset(train=True)
    print(f"training images: {len(dataset)}  classes: {dataset.classes}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_detections,
    )

    detector = build_detector("faster_rcnn", num_classes=len(dataset.classes))

    history = fine_tune(
        detector.model,
        loader,
        epochs=args.epochs,
        scheduler_name=args.scheduler,
        base_lr=args.base_lr,
        device=args.device,
        max_batches=args.max_batches,
    )

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = config.CHECKPOINTS_DIR / f"faster_rcnn_{args.scheduler}.pth"
    torch.save(detector.model.state_dict(), weights_path)

    history_path = config.CHECKPOINTS_DIR / f"history_{args.scheduler}.json"
    history_path.write_text(json.dumps(history.__dict__, indent=2))
    print(f"saved weights -> {weights_path}")
    print(f"saved history -> {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
