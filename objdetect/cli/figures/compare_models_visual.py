"""Render a side-by-side Faster R-CNN vs YOLO detection figure (presentation aid).

Runs both detectors on an image and saves a two-panel figure so the
accuracy/recall difference is visible at a glance. Defaults to the committed
demo photo so the presentation slide reproduces without the COCO dataset.

Usage:
  uv run python -m objdetect.cli.figures.compare_models_visual
  uv run python -m objdetect.cli.figures.compare_models_visual --image reports/figures/photo-to-compare-models.JPG
"""

import argparse

import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from objdetect import config
from objdetect.models import build_detector
from objdetect.visualization import draw_detections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="reports/figures/photo-to-compare-models.JPG",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")

    frcnn = build_detector("faster_rcnn")
    yolo = build_detector("yolo", weights=config.YOLO_BASE_WEIGHTS)

    frcnn_dets = frcnn.predict(image, score_threshold=args.threshold)
    yolo_dets = yolo.predict(image, score_threshold=args.threshold)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(draw_detections(image, frcnn_dets))
    axes[0].set_title(f"Faster R-CNN — {len(frcnn_dets)} detections")
    axes[1].imshow(draw_detections(image, yolo_dets))
    axes[1].set_title(f"YOLO26n — {len(yolo_dets)} detections")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "model_comparison_visual.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    print(f"Faster R-CNN: {len(frcnn_dets)} dets, YOLO26n: {len(yolo_dets)} dets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
