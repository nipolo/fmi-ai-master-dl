"""Shared pytest fixtures and lightweight test doubles.

These let the whole suite run fast and offline: no model weights are
downloaded and no COCO images are needed. A ``FakeDetector`` stands in for the
real Faster R-CNN / YOLO wrappers wherever a test only cares about the
detection plumbing, not the neural network itself.
"""

import numpy as np
import pytest
from PIL import Image

from objdetect.models.base import Detection


class FakeDetector:
    """A detector returning canned detections, filtered by score threshold."""

    name = "Fake"

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections if detections is not None else [
            Detection(label="dog", score=0.95, box=(10.0, 10.0, 100.0, 120.0)),
            Detection(label="person", score=0.80, box=(50.0, 30.0, 90.0, 200.0)),
            Detection(label="cat", score=0.40, box=(5.0, 5.0, 40.0, 60.0)),
        ]

    def predict(self, image, score_threshold: float = 0.5) -> list[Detection]:
        return [d for d in self._detections if d.score >= score_threshold]


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def sample_image() -> Image.Image:
    """A small deterministic RGB image (no disk or network needed)."""
    rng = np.random.default_rng(42)
    array = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")
