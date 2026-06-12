"""Model subpackage: a common detector interface over Faster R-CNN and YOLO."""

from objdetect.models.base import Detection, Detector
from objdetect.models.faster_rcnn import FasterRCNNDetector
from objdetect.models.yolo import YOLODetector

__all__ = [
    "Detection",
    "Detector",
    "FasterRCNNDetector",
    "YOLODetector",
    "build_detector",
]


def build_detector(name: str, **kwargs) -> Detector:
    """Factory: ``name`` is 'faster_rcnn' or 'yolo'."""
    builders = {
        "faster_rcnn": FasterRCNNDetector,
        "yolo": YOLODetector,
    }
    if name not in builders:
        raise ValueError(f"unknown model '{name}', expected one of {list(builders)}")
    return builders[name](**kwargs)
