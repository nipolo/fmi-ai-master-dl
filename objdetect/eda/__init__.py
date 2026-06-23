"""EDA subpackage: COCO exploratory analysis and the report generator.

The analysis functions are re-exported here so callers can simply
``import eda`` and use ``eda.annotations_frame(...)`` etc.; the figure/summary
generator lives in :mod:`eda.report` (run via ``python -m objdetect.eda.report``).
"""

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
