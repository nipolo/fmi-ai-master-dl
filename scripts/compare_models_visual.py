"""Render a side-by-side Faster R-CNN vs YOLO detection figure (presentation aid).

Picks a COCO val image, runs both detectors, and saves a two-panel image so the
accuracy/recall difference is visible at a glance.

Usage:
  uv run python scripts/compare_models_visual.py --image DATA/coco/val2017/000000000139.jpg
"""

import argparse

import matplotlib.pyplot as plt
from PIL import Image

from objdetect import config
from objdetect.models import build_detector
from objdetect.visualization import draw_detections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", default=str(config.COCO_IMAGES_DIR / "000000000139.jpg")
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    frcnn = build_detector("faster_rcnn")
    yolo = build_detector("yolo", weights="yolov8n.pt")

    frcnn_dets = frcnn.predict(image, score_threshold=args.threshold)
    yolo_dets = yolo.predict(image, score_threshold=args.threshold)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(draw_detections(image, frcnn_dets))
    axes[0].set_title(f"Faster R-CNN — {len(frcnn_dets)} detections")
    axes[1].imshow(draw_detections(image, yolo_dets))
    axes[1].set_title(f"YOLOv8n — {len(yolo_dets)} detections")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "model_comparison_visual.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    print(f"Faster R-CNN: {len(frcnn_dets)} dets, YOLOv8n: {len(yolo_dets)} dets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
