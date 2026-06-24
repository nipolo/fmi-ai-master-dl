"""Model subpackage: a common detector interface over Faster R-CNN and YOLO."""

from objdetect.models.base import Detection, Detector
from objdetect.models.ensemble import EnsembleDetector
from objdetect.models.faster_rcnn import FasterRCNNDetector
from objdetect.models.yolo import YOLODetector

__all__ = [
    "Detection",
    "Detector",
    "EnsembleDetector",
    "FasterRCNNDetector",
    "YOLODetector",
    "build_detector",
]


def build_detector(kind: str, **kwargs) -> Detector:
    """Factory: ``kind`` is 'faster_rcnn', 'yolo' or 'ensemble'."""
    if kind == "ensemble":
        members = [build_detector(k, **kw) for k, kw in kwargs.pop("members")]
        return EnsembleDetector(members, **kwargs)

    builders = {
        "faster_rcnn": FasterRCNNDetector,
        "yolo": YOLODetector,
    }
    if kind not in builders:
        raise ValueError(f"unknown model '{kind}', expected one of {list(builders)}")
    return builders[kind](**kwargs)
