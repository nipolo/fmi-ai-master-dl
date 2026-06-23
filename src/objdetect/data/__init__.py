"""Data subpackage: COCO download, dataset wrappers, and EDA utilities."""

from objdetect.data.coco import (
    CocoSubsetDataset,
    download_coco_val,
    load_coco_api,
)

__all__ = ["CocoSubsetDataset", "download_coco_val", "load_coco_api"]
