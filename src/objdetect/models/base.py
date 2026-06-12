"""The detector abstraction shared by Faster R-CNN and YOLO.

Both model families are wrapped behind one ``Detector`` protocol returning a
list of ``Detection`` objects, so the Streamlit app and the evaluation code
never branch on which model produced a result.
"""

from dataclasses import dataclass
from typing import Protocol

from PIL.Image import Image


@dataclass(frozen=True)
class Detection:
    """A single detected object in image-pixel coordinates.

    Attributes:
        label: human-readable class name (e.g. "dog").
        score: confidence in [0, 1].
        box: ``(x1, y1, x2, y2)`` in pixels, top-left origin.
    """

    label: str
    score: float
    box: tuple[float, float, float, float]


class Detector(Protocol):
    """Anything that turns an image into a list of detections."""

    name: str

    def predict(self, image: Image, score_threshold: float = 0.5) -> list[Detection]:
        """Run inference on a single PIL image and return kept detections."""
        ...
