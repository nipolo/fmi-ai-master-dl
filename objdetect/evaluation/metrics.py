"""COCO-style mean Average Precision (mAP) evaluation.

mAP is *the* object-detection metric. A prediction counts as correct when its
box overlaps a ground-truth box of the same class with Intersection-over-Union
above a threshold; precision is averaged over recall levels, over classes, and
(for the primary COCO metric) over IoU thresholds from 0.50 to 0.95.

We delegate the actual computation to ``pycocotools.COCOeval`` — the reference
implementation everyone benchmarks against — and just marshal predictions into
the format it expects.
"""

import contextlib
import io

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from objdetect.data.coco import collate_detections
from objdetect.utils import resolve_device


def _summarize_coco_eval(
    coco_gt: COCO, results: list[dict], cat_ids: list[int]
) -> dict[str, float]:
    """Run COCOeval on ``results`` restricted to ``cat_ids`` and pull headline AP."""
    if not results:
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = sorted({r["image_id"] for r in results})
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return {
        "mAP": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
    }


def _predictions_to_coco_results(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    max_images: int | None,
) -> list[dict]:
    """Run the model over the dataset and collect detections as COCO dicts.

    ``dataset`` yields contiguous labels 1..N; we map them back to real COCO
    category ids via the dataset so results align with the ground-truth API.
    """
    label_to_cat_id = {v: k for k, v in dataset.cat_id_to_label.items()}
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=collate_detections
    )

    model.eval()
    results: list[dict] = []
    seen = 0
    with torch.no_grad():
        for images, targets in loader:
            if max_images is not None and seen >= max_images:
                break
            outputs = model([img.to(device) for img in images])
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"])
                for box, label, score in zip(
                    output["boxes"], output["labels"], output["scores"]
                ):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": label_to_cat_id.get(int(label), int(label)),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO wants xywh
                            "score": float(score),
                        }
                    )
                seen += 1
    return results


def evaluate_coco_map(
    model: torch.nn.Module,
    dataset,
    device: str | torch.device | None = None,
    max_images: int | None = None,
) -> dict[str, float]:
    """Evaluate ``model`` on ``dataset`` and return the headline COCO metrics.

    Returns a dict with ``mAP`` (averaged over IoU 0.50:0.95), ``mAP_50``, and
    ``mAP_75``. Returns zeros when the model predicts nothing, so callers do
    not have to special-case an empty run.
    """
    device = resolve_device(device)
    model.to(device)

    results = _predictions_to_coco_results(model, dataset, device, max_images)
    return _summarize_coco_eval(dataset.coco, results, dataset.cat_ids)


def evaluate_detector_map(
    detector,
    dataset,
    score_threshold: float = 0.05,
    max_images: int | None = None,
) -> dict[str, float]:
    """Evaluate any :class:`~models.base.Detector` via mAP.

    Unlike :func:`evaluate_coco_map` (which needs a torchvision model and its
    label space), this works through the common ``predict`` interface, so the
    *same* function fairly scores both the pretrained Faster R-CNN and YOLO
    baselines. Predicted class *names* are mapped back to COCO category ids, so
    it only requires that the detector emits COCO class names (true for both
    pretrained models). A low ``score_threshold`` keeps weak detections so the
    precision-recall curve — and thus AP — is complete.
    """
    coco_gt: COCO = dataset.coco
    name_to_cat_id = {
        coco_gt.loadCats([cid])[0]["name"]: cid for cid in dataset.cat_ids
    }

    results: list[dict] = []
    for index in range(len(dataset) if max_images is None else min(max_images, len(dataset))):
        image_id = dataset.image_ids[index]
        info = coco_gt.loadImgs([image_id])[0]
        image = Image.open(dataset.images_dir / info["file_name"]).convert("RGB")
        for det in detector.predict(image, score_threshold=score_threshold):
            if det.label not in name_to_cat_id:  # outside the evaluated subset
                continue
            x1, y1, x2, y2 = det.box
            results.append(
                {
                    "image_id": image_id,
                    "category_id": name_to_cat_id[det.label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": det.score,
                }
            )
    return _summarize_coco_eval(coco_gt, results, dataset.cat_ids)
