"""COCO val2017 download and a PyTorch dataset over a class subset.

We use the *validation* split of COCO 2017 (5 000 images, ~780 MB) instead of
the full training split (118 000 images, ~19 GB): it is large enough for
fine-tuning experiments on a 10-class subset while staying inside the
project's time and cloud budget.
"""

import pathlib as pl
import zipfile

import requests
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import tv_tensors

from objdetect import config


def _download_file(url: str, destination: pl.Path, chunk_size: int = 1 << 20) -> None:
    """Stream a large file to disk with a simple progress printout."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with open(destination, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{destination.name}: {done / total:6.1%}", end="")
    print()


def _fetch_and_unpack(url: str, target: pl.Path, force: bool = False) -> None:
    """Download a COCO zip and extract it into ``COCO_DIR`` unless ``target`` exists."""
    if target.exists() and not force:
        print(f"{target} already present, skipping")
        return
    config.COCO_DIR.mkdir(parents=True, exist_ok=True)
    archive = config.COCO_DIR / url.rsplit("/", 1)[-1]
    if not archive.exists() or force:
        print(f"Downloading {url}")
        _download_file(url, archive)
    print(f"Unpacking {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(config.COCO_DIR)
    archive.unlink()


def download_coco_annotations(force: bool = False) -> None:
    """Download just the COCO val2017 instance annotations (no images).

    The EDA reads only box/image metadata, so this skips the ~780 MB image
    download. Safe to call repeatedly.
    """
    _fetch_and_unpack(
        config.COCO_ANNOTATIONS_URL, config.COCO_ANNOTATIONS_FILE.parent, force=force
    )


def download_coco_val(force: bool = False) -> None:
    """Download and unpack COCO val2017 images and instance annotations.

    Skips work that is already done, so it is safe to call repeatedly.
    """
    _fetch_and_unpack(config.COCO_VAL_IMAGES_URL, config.COCO_IMAGES_DIR, force=force)
    _fetch_and_unpack(
        config.COCO_ANNOTATIONS_URL, config.COCO_ANNOTATIONS_FILE.parent, force=force
    )


def load_coco_api(
    annotations_file: pl.Path = config.COCO_ANNOTATIONS_FILE,
) -> COCO:
    """Load the pycocotools API object for the instance annotations."""
    return COCO(str(annotations_file))


class CocoSubsetDataset(Dataset):
    """COCO restricted to ``config.SUBSET_CLASSES``, in torchvision format.

    Each item is ``(image_tensor, target)`` where ``target`` is the dict that
    torchvision detection models expect:

    - ``boxes``: float tensor of shape (n, 4) in ``[x1, y1, x2, y2]`` pixels,
    - ``labels``: int tensor of shape (n,) with values 1..len(classes)
      (label 0 is reserved for background in Faster R-CNN),
    - ``image_id``: the original COCO image id (needed for COCO evaluation).

    Images that contain no boxes from the subset are dropped, as are
    ``iscrowd`` annotations (group boxes that are not individual objects)
    and degenerate boxes with zero width or height.
    """

    def __init__(
        self,
        coco: COCO | None = None,
        classes: list[str] | None = None,
        images_dir: pl.Path = config.COCO_IMAGES_DIR,
        train: bool = False,
    ) -> None:
        self.coco = coco if coco is not None else load_coco_api()
        self.classes = classes if classes is not None else config.SUBSET_CLASSES
        self.images_dir = images_dir

        # Map COCO category ids (sparse, 1..90) to contiguous labels 1..N.
        self.cat_ids = self.coco.getCatIds(catNms=self.classes)
        self.cat_id_to_label = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}
        self.label_to_name = {
            i + 1: self.coco.loadCats([cid])[0]["name"]
            for i, cid in enumerate(self.cat_ids)
        }

        # Keep only images that have at least one usable annotation.
        image_ids: set[int] = set()
        for cid in self.cat_ids:
            image_ids.update(self.coco.getImgIds(catIds=[cid]))
        self.image_ids = sorted(image_ids)

        # Geometric augmentation only for training; both pipelines produce
        # float tensors in [0, 1], which torchvision detection models expect.
        if train:
            self.transforms = T.Compose(
                [
                    T.ToImage(),
                    T.RandomHorizontalFlip(0.5),
                    T.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.transforms = T.Compose(
                [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")

        ann_ids = self.coco.getAnnIds(
            imgIds=[image_id], catIds=self.cat_ids, iscrowd=False
        )
        boxes, labels = [], []
        for ann in self.coco.loadAnns(ann_ids):
            x, y, w, h = ann["bbox"]  # COCO stores [x, y, width, height]
            if w <= 1 or h <= 1:  # degenerate boxes break box regression loss
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])

        # BoundingBoxes (not a plain tensor) so geometric augmentations like
        # the horizontal flip transform the boxes together with the image.
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                format="XYXY",
                canvas_size=(info["height"], info["width"]),
            ),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": image_id,
        }
        image_tensor, target = self.transforms(image, target)
        return image_tensor, target


def collate_detections(batch: list) -> tuple[list, list]:
    """Collate function for detection: images vary in size, so no stacking."""
    return tuple(zip(*batch))
