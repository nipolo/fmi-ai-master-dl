"""The detector abstraction shared by Faster R-CNN and YOLO."""

from dataclasses import dataclass
from typing import Protocol

from PIL.Image import Image


@dataclass(frozen=True)
class Detection:
    """A single detected object in image-pixel coordinates."""

    label: str
    score: float
    box: tuple[float, float, float, float]


class Detector(Protocol):
    """Anything that turns an image into a list of detections."""

    name: str

    def predict(self, image: Image, score_threshold: float = 0.5) -> list[Detection]:
        """Run inference on a single PIL image and return kept detections."""
        ...
