"""Benchmark the pretrained Faster R-CNN and YOLO baselines (Reqs. 3-4).

Measures, on the same COCO subset, for each model:
  - mAP@[.50:.95], mAP@.50 (via the shared detector evaluator),
  - inference speed (images/second),
  - parameter count.
Writes reports/baseline_results.json, which the model report quotes.

Usage:
  uv run python -m objdetect.cli.evaluation.benchmark_baselines --max-images 200
"""

import argparse
import json
import time

from PIL import Image

from objdetect import config
from objdetect.data.coco import CocoSubsetDataset
from objdetect.evaluation import evaluate_detector_map
from objdetect.models import build_detector


def _count_params(detector) -> int:
    if hasattr(detector, "model") and hasattr(detector.model, "parameters"):
        try:
            return sum(p.numel() for p in detector.model.parameters())
        except TypeError:
            pass
    return 0


def _measure_fps(detector, dataset, num_images: int) -> float:
    """Average images/second over the first ``num_images`` (after a warm-up)."""
    paths = [
        dataset.images_dir / dataset.coco.loadImgs([dataset.image_ids[i]])[0]["file_name"]
        for i in range(min(num_images, len(dataset)))
    ]
    images = [Image.open(p).convert("RGB") for p in paths]
    detector.predict(images[0], score_threshold=0.5)  # warm-up (lazy CUDA/MPS init)

    start = time.time()
    for image in images:
        detector.predict(image, score_threshold=0.5)
    elapsed = time.time() - start
    return len(images) / elapsed if elapsed > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--fps-images", type=int, default=30)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dataset = CocoSubsetDataset(train=False)
    models = {
        "Faster R-CNN": ("faster_rcnn", {"device": args.device}),
        "YOLO26n": ("yolo", {"weights": config.YOLO_BASE_WEIGHTS, "device": args.device}),
    }

    results = {}
    for display, (builder, kwargs) in models.items():
        print(f"\n=== {display} ===")
        detector = build_detector(builder, **kwargs)
        metrics = evaluate_detector_map(detector, dataset, max_images=args.max_images)
        fps = _measure_fps(detector, dataset, args.fps_images)
        params = _count_params(detector)
        results[display] = {
            "mAP": round(metrics["mAP"], 4),
            "mAP_50": round(metrics["mAP_50"], 4),
            "mAP_75": round(metrics["mAP_75"], 4),
            "fps": round(fps, 2),
            "params_millions": round(params / 1e6, 1),
            "evaluated_images": min(args.max_images, len(dataset)),
        }
        print(json.dumps(results[display], indent=2))

    out = config.REPORTS_DIR / "baseline_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
