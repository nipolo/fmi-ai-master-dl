"""EDA subpackage: COCO exploratory analysis and the report generator."""

from objdetect.eda.analysis import (
    annotations_frame,
    boxes_per_image,
    class_cooccurrence,
    class_distribution,
    find_anomalies,
)

__all__ = [
    "annotations_frame",
    "boxes_per_image",
    "class_cooccurrence",
    "class_distribution",
    "find_anomalies",
]
