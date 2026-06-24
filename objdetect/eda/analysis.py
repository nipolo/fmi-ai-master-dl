"""Exploratory data analysis over COCO instance annotations."""

import pandas as pd
from pycocotools.coco import COCO


def annotations_frame(coco: COCO) -> pd.DataFrame:
    """One row per annotation (= one labelled box) with derived geometry."""
    categories = {c["id"]: c for c in coco.loadCats(coco.getCatIds())}
    images = {i["id"]: i for i in coco.loadImgs(coco.getImgIds())}

    rows = []
    for ann in coco.loadAnns(coco.getAnnIds()):
        cat = categories[ann["category_id"]]
        img = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        rows.append(
            {
                "image_id": ann["image_id"],
                "category": cat["name"],
                "supercategory": cat["supercategory"],
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": w * h,
                "aspect_ratio": w / h if h > 0 else float("nan"),
                "iscrowd": bool(ann["iscrowd"]),
                "image_width": img["width"],
                "image_height": img["height"],
                "relative_area": (w * h) / (img["width"] * img["height"]),
            }
        )
    return pd.DataFrame(rows)


def class_distribution(annotations: pd.DataFrame) -> pd.DataFrame:
    """Instances and images per class, sorted by instance count (imbalance view)."""
    grouped = annotations.groupby("category").agg(
        instances=("image_id", "size"),
        images=("image_id", "nunique"),
    )
    return grouped.sort_values("instances", ascending=False).reset_index()


def boxes_per_image(annotations: pd.DataFrame) -> pd.Series:
    """Number of annotated boxes in each image (scene-density distribution)."""
    return annotations.groupby("image_id").size().rename("boxes")


def class_cooccurrence(annotations: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """How often pairs of the ``top_n`` most frequent classes share an image."""
    top_classes = class_distribution(annotations).head(top_n)["category"]
    present = (
        annotations[annotations["category"].isin(top_classes)]
        .groupby(["image_id", "category"])
        .size()
        .unstack(fill_value=0)
        .gt(0)
    )
    cooc = present.T.astype(int) @ present.astype(int)
    return cooc.loc[top_classes, top_classes]


def find_anomalies(
    annotations: pd.DataFrame,
    tiny_area_px: float = 16.0,
    extreme_aspect: float = 10.0,
) -> pd.DataFrame:
    """Flag suspicious annotations, with one ``anomaly`` label per row."""
    frames = []
    for mask, name in [
        (annotations["area"] < tiny_area_px, "tiny"),
        (
            (annotations["aspect_ratio"] > extreme_aspect)
            | (annotations["aspect_ratio"] < 1 / extreme_aspect),
            "extreme_aspect",
        ),
        (annotations["iscrowd"], "crowd"),
        (
            (annotations["x"] < 0)
            | (annotations["y"] < 0)
            | (annotations["x"] + annotations["width"] > annotations["image_width"])
            | (annotations["y"] + annotations["height"] > annotations["image_height"]),
            "out_of_bounds",
        ),
    ]:
        flagged = annotations[mask].copy()
        flagged["anomaly"] = name
        frames.append(flagged)
    return pd.concat(frames, ignore_index=True)
