"""Evaluate a detector's COCO mAP on the class subset.

Usage:
  uv run python -m objdetect.cli.evaluation.evaluate --weights DATA/checkpoints/faster_rcnn_cosine.pth
  uv run python -m objdetect.cli.evaluation.evaluate            # pretrained baseline
"""

import argparse
import json

from objdetect.data.coco import CocoSubsetDataset
from objdetect.evaluation import evaluate_coco_map
from objdetect.models import build_detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None, help="fine-tuned checkpoint path")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dataset = CocoSubsetDataset(train=False)

    if args.weights:
        detector = build_detector(
            "faster_rcnn",
            num_classes=len(dataset.classes),
            weights_path=args.weights,
            device=args.device,
        )
    else:
        # Pretrained 80-class model evaluated on the subset categories.
        detector = build_detector("faster_rcnn", device=args.device)

    metrics = evaluate_coco_map(
        detector.model, dataset, device=args.device, max_images=args.max_images
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
