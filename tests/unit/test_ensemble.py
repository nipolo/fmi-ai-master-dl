"""Unit tests for the ensemble detector."""

import unittest

import numpy as np
from PIL import Image

from objdetect.models.base import Detection
from objdetect.models.ensemble import EnsembleDetector
from tests.conftest import FakeDetector


def _blank_image() -> Image.Image:
    return Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")


class TestEnsembleDetectorPredict(unittest.TestCase):
    def test_when_predicting_then_members_detections_are_concatenated_in_order(self):
        # Arrange
        first = FakeDetector([Detection(label="dog", score=0.9, box=(0, 0, 1, 1))])
        second = FakeDetector(
            [
                Detection(label="car", score=0.8, box=(2, 2, 3, 3)),
                Detection(label="cat", score=0.7, box=(4, 4, 5, 5)),
            ]
        )
        ensemble = EnsembleDetector([first, second], name="Combo")
        image = _blank_image()
        expected_labels = ["dog", "car", "cat"]

        # Act
        detections = ensemble.predict(image, score_threshold=0.0)
        actual_labels = [d.label for d in detections]

        # Assert
        self.assertEqual(actual_labels, expected_labels)
        self.assertEqual(ensemble.name, "Combo")

    def test_when_threshold_high_then_it_is_passed_to_each_member(self):
        # Arrange
        first = FakeDetector(
            [
                Detection(label="dog", score=0.9, box=(0, 0, 1, 1)),
                Detection(label="cat", score=0.4, box=(4, 4, 5, 5)),
            ]
        )
        second = FakeDetector([Detection(label="car", score=0.8, box=(2, 2, 3, 3))])
        ensemble = EnsembleDetector([first, second])
        image = _blank_image()
        expected_labels = ["dog", "car"]

        # Act
        detections = ensemble.predict(image, score_threshold=0.5)
        actual_labels = [d.label for d in detections]

        # Assert
        self.assertEqual(actual_labels, expected_labels)


if __name__ == "__main__":
    unittest.main()
