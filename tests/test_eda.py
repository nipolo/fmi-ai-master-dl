"""Unit tests for the EDA utilities

A tiny fake COCO object provides three annotations so
the analysis functions can be checked without downloading the dataset.
"""

import unittest

from objdetect import eda


class _FakeCoco:
    """Minimal stand-in exposing the pycocotools methods eda.py calls."""

    def __init__(self):
        self._cats = {
            1: {"id": 1, "name": "person", "supercategory": "person"},
            2: {"id": 2, "name": "dog", "supercategory": "animal"},
        }
        self._imgs = {
            10: {"id": 10, "width": 100, "height": 100},
            11: {"id": 11, "width": 200, "height": 100},
        }
        self._anns = {
            100: {
                "id": 100,
                "image_id": 10,
                "category_id": 1,
                "bbox": [0, 0, 50, 80],
                "iscrowd": 0,
            },
            101: {
                "id": 101,
                "image_id": 10,
                "category_id": 2,
                "bbox": [10, 10, 2, 2],
                "iscrowd": 0,
            },
            102: {
                "id": 102,
                "image_id": 11,
                "category_id": 1,
                "bbox": [0, 0, 199, 5],
                "iscrowd": 1,
            },
        }

    def getCatIds(self, catNms=None):
        return list(self._cats)

    def loadCats(self, ids):
        return [self._cats[i] for i in ids]

    def getImgIds(self, catIds=None):
        return list(self._imgs)

    def loadImgs(self, ids):
        return [self._imgs[i] for i in ids]

    def getAnnIds(self, imgIds=None, catIds=None, iscrowd=None):
        return list(self._anns)

    def loadAnns(self, ids):
        return [self._anns[i] for i in ids]


class TestAnnotationsFrame(unittest.TestCase):
    def test_when_built_then_one_row_per_annotation_with_derived_fields(self):
        # Arrange
        coco = _FakeCoco()
        expected_rows = 3
        expected_area_first = 50 * 80

        # Act
        frame = eda.annotations_frame(coco)
        actual_rows = len(frame)
        actual_area_first = frame.iloc[0]["area"]

        # Assert
        self.assertEqual(actual_rows, expected_rows)
        self.assertEqual(actual_area_first, expected_area_first)
        self.assertIn("relative_area", frame.columns)
        self.assertIn("aspect_ratio", frame.columns)


class TestClassDistribution(unittest.TestCase):
    def test_when_computed_then_person_has_two_instances(self):
        # Arrange
        coco = _FakeCoco()
        frame = eda.annotations_frame(coco)
        expected_person_instances = 2

        # Act
        distribution = eda.class_distribution(frame)
        actual_person_instances = int(
            distribution.loc[distribution["category"] == "person", "instances"].iloc[0]
        )

        # Assert
        self.assertEqual(actual_person_instances, expected_person_instances)


class TestFindAnomalies(unittest.TestCase):
    def test_when_scanned_then_flags_tiny_crowd_and_extreme_boxes(self):
        # Arrange
        coco = _FakeCoco()
        frame = eda.annotations_frame(coco)

        # Act
        anomalies = eda.find_anomalies(frame)
        flagged_types = set(anomalies["anomaly"])

        # Assert
        # The 2x2 box is tiny; the 199x5 box is both extreme-aspect and crowd.
        self.assertIn("tiny", flagged_types)
        self.assertIn("crowd", flagged_types)
        self.assertIn("extreme_aspect", flagged_types)


if __name__ == "__main__":
    unittest.main()
