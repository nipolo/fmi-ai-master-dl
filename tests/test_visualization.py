"""Unit tests for detection drawing"""

import unittest

import numpy as np
from PIL import Image

from objdetect.models.base import Detection
from objdetect.visualization import draw_detections


def _blank_image() -> Image.Image:
    return Image.fromarray(np.zeros((80, 80, 3), dtype=np.uint8), mode="RGB")


class TestDrawDetections(unittest.TestCase):
    def test_when_drawing_then_returns_same_size_rgb_image(self):
        # Arrange
        image = _blank_image()
        detections = [Detection(label="dog", score=0.9, box=(5, 5, 40, 40))]
        expected_size = image.size
        expected_mode = "RGB"

        # Act
        result = draw_detections(image, detections)

        # Assert
        self.assertEqual(result.size, expected_size)
        self.assertEqual(result.mode, expected_mode)

    def test_when_drawing_then_does_not_mutate_original(self):
        # Arrange
        image = _blank_image()
        before = np.asarray(image).copy()
        detections = [Detection(label="cat", score=0.8, box=(0, 0, 30, 30))]

        # Act
        draw_detections(image, detections)
        after = np.asarray(image)

        # Assert
        self.assertTrue(np.array_equal(before, after))

    def test_when_box_present_then_pixels_change(self):
        # Arrange
        image = _blank_image()
        detections = [Detection(label="cup", score=0.7, box=(10, 10, 50, 50))]

        # Act
        result = draw_detections(image, detections)
        difference = np.abs(
            np.asarray(result, dtype=int) - np.asarray(image, dtype=int)
        ).sum()

        # Assert
        self.assertGreater(difference, 0)


if __name__ == "__main__":
    unittest.main()
