"""Unit tests for the app-facing inference helpers (Requirement 6).

Course-exercise style: unittest.TestCase, Arrange / Act / Assert.
Uses the FakeDetector from conftest so no model weights are needed.

The ``detections_to_frame`` and ``summarize_counts`` behaviours are described
by Gherkin scenarios in ``tests/features/app_workflow.feature`` instead, so they
are intentionally not duplicated here. ``run_detection`` keeps a unit test
because it pins the full descending sort order, which the feature only checks
at the top position.
"""

import unittest

import numpy as np
from PIL import Image

from objdetect.app.inference import run_detection
from tests.conftest import FakeDetector


def _make_image() -> Image.Image:
    rng = np.random.default_rng(7)
    return Image.fromarray(
        rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8), mode="RGB"
    )


class TestRunDetection(unittest.TestCase):

    def test_when_threshold_low_then_returns_sorted_by_score(self):
        # Arrange
        detector = FakeDetector()
        image = _make_image()
        threshold = 0.0
        expected_count = 3
        expected_first_label = "dog"

        # Act
        detections = run_detection(detector, image, score_threshold=threshold)
        actual_count = len(detections)
        actual_first_label = detections[0].label
        scores = [d.score for d in detections]

        # Assert
        self.assertEqual(actual_count, expected_count)
        self.assertEqual(actual_first_label, expected_first_label)
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_when_threshold_high_then_filters_low_scores(self):
        # Arrange
        detector = FakeDetector()
        image = _make_image()
        threshold = 0.5
        expected_count = 2

        # Act
        detections = run_detection(detector, image, score_threshold=threshold)
        actual_count = len(detections)

        # Assert
        self.assertEqual(actual_count, expected_count)


if __name__ == "__main__":
    unittest.main()
