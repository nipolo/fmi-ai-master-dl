"""Unit tests for the app-facing inference helpers (Requirement 6).

Course-exercise style: unittest.TestCase, Arrange / Act / Assert.
Uses the FakeDetector from conftest so no model weights are needed.
"""

import unittest

import numpy as np
from PIL import Image

from objdetect.app.inference import (
    detections_to_frame,
    run_detection,
    summarize_counts,
)
from objdetect.models.base import Detection
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


class TestDetectionsToFrame(unittest.TestCase):

    def test_when_given_detections_then_frame_has_expected_columns_and_rows(self):
        # Arrange
        detections = [
            Detection(label="dog", score=0.9, box=(1.0, 2.0, 3.0, 4.0)),
            Detection(label="cat", score=0.7, box=(5.0, 6.0, 7.0, 8.0)),
        ]
        expected_rows = 2
        expected_columns = ["label", "score", "x1", "y1", "x2", "y2"]

        # Act
        frame = detections_to_frame(detections)
        actual_rows = len(frame)
        actual_columns = list(frame.columns)

        # Assert
        self.assertEqual(actual_rows, expected_rows)
        self.assertEqual(actual_columns, expected_columns)


class TestSummarizeCounts(unittest.TestCase):

    def test_when_repeated_labels_then_counts_per_class(self):
        # Arrange
        detections = [
            Detection(label="person", score=0.9, box=(0, 0, 1, 1)),
            Detection(label="person", score=0.8, box=(0, 0, 1, 1)),
            Detection(label="dog", score=0.7, box=(0, 0, 1, 1)),
        ]
        expected = {"person": 2, "dog": 1}

        # Act
        actual = summarize_counts(detections)

        # Assert
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
