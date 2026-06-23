"""Data subpackage: COCO download and dataset wrappers (EDA lives in the `eda` package)."""

from objdetect.data.coco import (
    CocoSubsetDataset,
    download_coco_val,
    load_coco_api,
)

__all__ = ["CocoSubsetDataset", "download_coco_val", "load_coco_api"]
