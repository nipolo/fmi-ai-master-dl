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

from objdetect.app.inference import run_detection, select_detections
from objdetect.models.base import Detection
from tests.conftest import FakeDetector


def _three_detections() -> list[Detection]:
    """Three distinct detections whose order the tests rely on."""
    return [
        Detection(label="tv", score=0.88, box=(1.0, 1.0, 2.0, 2.0)),
        Detection(label="chair", score=0.60, box=(3.0, 3.0, 4.0, 4.0)),
        Detection(label="tv", score=0.55, box=(5.0, 5.0, 6.0, 6.0)),
    ]


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


class TestSelectDetections(unittest.TestCase):

    def test_when_no_rows_selected_then_returns_all_detections(self):
        # Arrange
        detections = _three_detections()
        selected_rows: list[int] = []
        expected = detections

        # Act
        actual = select_detections(detections, selected_rows)

        # Assert
        self.assertEqual(actual, expected)

    def test_when_rows_selected_then_returns_only_those_in_original_order(self):
        # Arrange
        detections = _three_detections()
        selected_rows = [2, 0]  # out of order on purpose
        expected = [detections[0], detections[2]]

        # Act
        actual = select_detections(detections, selected_rows)

        # Assert
        self.assertEqual(actual, expected)

    def test_when_selection_index_out_of_range_then_it_is_ignored(self):
        # Arrange
        detections = _three_detections()
        selected_rows = [1, 99]  # 99 is a stale index from a previous image
        expected = [detections[1]]

        # Act
        actual = select_detections(detections, selected_rows)

        # Assert
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
