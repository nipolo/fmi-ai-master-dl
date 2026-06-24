"""Traffic-cone dataset in torchvision-detection format, for Faster R-CNN.

Reuses :class:`~data.coco.CocoSubsetDataset` over the COCO-format cone
annotations emitted by ``cli.data.prepare_cone_coco``, so the two-stage detector
trains and is evaluated through exactly the same pipeline as the COCO subset
(same target format, same ``evaluate_coco_map``).
"""

from pycocotools.coco import COCO

from objdetect import config
from objdetect.data.coco import CocoSubsetDataset


def cone_dataset(split: str, train: bool) -> CocoSubsetDataset:
    """Build the cone train/val dataset for the given split.

    ``split`` is ``"train"`` or ``"val"``; ``train`` toggles augmentation.
    """
    annotations_file = config.CONE_COCO_TRAIN if split == "train" else config.CONE_COCO_VAL
    if not annotations_file.is_file():
        raise SystemExit(
            f"cone COCO annotations not found at {annotations_file}; run "
            "`uv run python -m objdetect.cli.data.prepare_cone_coco` first"
        )
    coco = COCO(str(annotations_file))
    return CocoSubsetDataset(
        coco=coco,
        classes=config.CONE_CLASS_NAMES,
        images_dir=config.TRAFFIC_CONE_DIR / "images" / split,
        train=train,
    )
