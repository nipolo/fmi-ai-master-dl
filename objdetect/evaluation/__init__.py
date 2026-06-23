"""Evaluation subpackage: COCO-style mAP metrics."""

from objdetect.evaluation.metrics import (
    evaluate_coco_map,
    evaluate_detector_map,
)

__all__ = ["evaluate_coco_map", "evaluate_detector_map"]
