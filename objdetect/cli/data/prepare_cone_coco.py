"""Convert the YOLO-format traffic-cone dataset to COCO-format annotations.

The cone dataset (laid out by ``cli.data.prepare_cone_dataset``) stores labels in
YOLO format — one ``*.txt`` per image, each line ``cls cx cy w h`` normalised to
the image size. Faster R-CNN in this project reads COCO-style annotations
(``CocoSubsetDataset`` over a ``pycocotools`` API), so to fine-tune Faster R-CNN
on the *same* images as the YOLO cone experiment we first emit COCO instance
JSON files:

    DATA/traffic_cone/annotations/instances_train.json
    DATA/traffic_cone/annotations/instances_val.json

This is data prep, not training — run it once before
``cli.training.train_cone_frcnn``:

    uv run python -m objdetect.cli.data.prepare_cone_coco
"""

import json

from PIL import Image

from objdetect import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _yolo_line_to_bbox(line: str, width: int, height: int) -> list[float] | None:
    """Convert one YOLO label line to a COCO ``[x, y, w, h]`` pixel box.

    Returns ``None`` for blank lines or degenerate (<=1 px) boxes.
    """
    parts = line.split()
    if len(parts) != 5:
        return None
    _cls, cx, cy, bw, bh = (float(p) for p in parts)
    w_px, h_px = bw * width, bh * height
    x_px, y_px = (cx - bw / 2) * width, (cy - bh / 2) * height
    # Clamp to the image and drop degenerate boxes (consistent with the COCO path).
    x_px, y_px = max(0.0, x_px), max(0.0, y_px)
    w_px = min(w_px, width - x_px)
    h_px = min(h_px, height - y_px)
    if w_px <= 1 or h_px <= 1:
        return None
    return [round(x_px, 2), round(y_px, 2), round(w_px, 2), round(h_px, 2)]


def _build_split(split: str) -> dict:
    """Build a COCO instances dict for one split (train/val)."""
    images_dir = config.TRAFFIC_CONE_DIR / "images" / split
    labels_dir = config.TRAFFIC_CONE_DIR / "labels" / split
    if not images_dir.is_dir():
        raise SystemExit(
            f"cone images not found at {images_dir}; run "
            "`uv run python -m objdetect.cli.data.prepare_cone_dataset` first"
        )

    # Single class; COCO category ids are 1-based (0 is background for the model).
    categories = [
        {"id": i + 1, "name": name, "supercategory": "none"}
        for i, name in enumerate(config.CONE_CLASS_NAMES)
    ]

    images, annotations = [], []
    ann_id = 1
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue  # skip the cached .npy files Ultralytics leaves behind
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        for line in label_path.read_text().splitlines():
            bbox = _yolo_line_to_bbox(line, width, height)
            if bbox is None:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,  # single class -> category id 1
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def main() -> int:
    out_dir = config.CONE_COCO_TRAIN.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, out_path in (
        ("train", config.CONE_COCO_TRAIN),
        ("val", config.CONE_COCO_VAL),
    ):
        coco = _build_split(split)
        out_path.write_text(json.dumps(coco))
        print(
            f"{split}: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} boxes -> {out_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
