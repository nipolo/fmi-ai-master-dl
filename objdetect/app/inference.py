"""App-facing inference helpers, kept free of any Streamlit imports."""

import os

import pandas as pd
from PIL.Image import Image

from objdetect import config
from objdetect.models import build_detector
from objdetect.models.base import Detection, Detector

CONE_WEIGHTS = config.CONE_WEIGHTS
CONE_WEIGHTS_20EP = config.CONE_WEIGHTS_20EP
CONE_FRCNN_WEIGHTS = config.CONE_FRCNN_WEIGHTS

AVAILABLE_MODELS = {
    "Faster R-CNN (two-stage)": ("faster_rcnn", {}),
    "Faster R-CNN + Cones (81 classes)": (
        "ensemble",
        {
            "name": "Faster R-CNN + Cones",
            "members": [
                ("faster_rcnn", {}),
                (
                    "faster_rcnn",
                    {
                        "num_classes": 1,
                        "weights_path": CONE_FRCNN_WEIGHTS,
                        "class_names": ["traffic cone"],
                    },
                ),
            ],
        },
    ),
    "YOLO26n (one-stage)": ("yolo", {"weights": config.YOLO_BASE_WEIGHTS}),
    "YOLO26n + Cones — 100 ep (81 classes)": (
        "ensemble",
        {
            "name": "YOLO26n + Cones (100 ep)",
            "members": [
                ("yolo", {"weights": config.YOLO_BASE_WEIGHTS}),
                ("yolo", {"weights": CONE_WEIGHTS}),
            ],
        },
    ),
    "YOLO26n + Cones — 20 ep (81 classes)": (
        "ensemble",
        {
            "name": "YOLO26n + Cones (20 ep)",
            "members": [
                ("yolo", {"weights": config.YOLO_BASE_WEIGHTS}),
                ("yolo", {"weights": CONE_WEIGHTS_20EP}),
            ],
        },
    ),
}


def _missing_weights(builder_name: str, kwargs: dict) -> str | None:
    if builder_name == "ensemble":
        for member_name, member_kwargs in kwargs.get("members", []):
            missing = _missing_weights(member_name, member_kwargs)
            if missing:
                return missing
        return None
    path = kwargs.get("weights_path") or kwargs.get("weights")
    if path and not os.path.exists(path):
        return path
    return None


def load_model(display_name: str) -> Detector:
    """Instantiate the detector chosen in the UI dropdown."""
    if display_name not in AVAILABLE_MODELS:
        raise ValueError(f"unknown model '{display_name}'")
    builder_name, kwargs = AVAILABLE_MODELS[display_name]
    missing = _missing_weights(builder_name, kwargs)
    if missing:
        raise FileNotFoundError(
            f"Model weights not found: {missing}\n"
            "The cone Faster R-CNN weights (~165 MB) are not committed. Either\n"
            f"download them to {missing} from\n"
            "https://drive.google.com/file/d/1DKT2E__iErPYJHZxSdWXMG_TYYIIJO9c/view\n"
            "or train them: uv run python -m objdetect.cli.training.train_cone_frcnn --device cpu --epochs 20"
        )
    return build_detector(builder_name, **kwargs)


def run_detection(
    model: Detector, image: Image, score_threshold: float = 0.5
) -> list[Detection]:
    """Run the detector and return detections sorted by descending score."""
    detections = model.predict(image, score_threshold=score_threshold)
    return sorted(detections, key=lambda d: d.score, reverse=True)


def detections_to_frame(detections: list[Detection]) -> pd.DataFrame:
    """Tabulate detections for display, one row per detected object."""
    return pd.DataFrame(
        [
            {
                "label": d.label,
                "score": round(d.score, 3),
                "x1": d.box[0],
                "y1": d.box[1],
                "x2": d.box[2],
                "y2": d.box[3],
            }
            for d in detections
        ],
        columns=["label", "score", "x1", "y1", "x2", "y2"],
    )


def select_detections(
    detections: list[Detection], selected_rows: list[int]
) -> list[Detection]:
    """Pick the detections to draw given the table rows the user selected."""
    if not selected_rows:
        return detections
    chosen = set(selected_rows)
    return [det for i, det in enumerate(detections) if i in chosen]


def summarize_counts(detections: list[Detection]) -> dict[str, int]:
    """Count detections per class label (the app's summary line)."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return counts
