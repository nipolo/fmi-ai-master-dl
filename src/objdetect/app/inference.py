"""App-facing inference helpers, kept free of any Streamlit imports."""

import pandas as pd
from PIL.Image import Image

from objdetect import config
from objdetect.models import build_detector
from objdetect.models.base import Detection, Detector

# Fine-tuned single-class cone weights (see scripts/prepare_cone_dataset.py and
# the cone fine-tune); a path so YOLO loads the checkpoint rather than a hub name.
CONE_WEIGHTS = str(config.CHECKPOINTS_DIR / "cone_yolo26n.pt")

# Friendly label -> (builder name, kwargs) for the models the app offers.
# The cone model adds a class absent from COCO's 80; on its own it detects only
# that class, so it is exposed only via the combined entry, which runs it
# alongside stock YOLO26n to cover all 81 classes (80 COCO + traffic cone).
AVAILABLE_MODELS = {
    "Faster R-CNN (two-stage)": ("faster_rcnn", {}),
    "YOLO26n (one-stage)": ("yolo", {"weights": "yolo26n.pt"}),
    "YOLO26n + Cones (81 classes)": (
        "ensemble",
        {
            "name": "YOLO26n + Cones",
            "members": [
                ("yolo", {"weights": "yolo26n.pt"}),
                ("yolo", {"weights": CONE_WEIGHTS}),
            ],
        },
    ),
}


def load_model(display_name: str) -> Detector:
    """Instantiate the detector chosen in the UI dropdown."""
    if display_name not in AVAILABLE_MODELS:
        raise ValueError(f"unknown model '{display_name}'")
    builder_name, kwargs = AVAILABLE_MODELS[display_name]
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
    """Pick the detections to draw given the table rows the user selected.

    ``selected_rows`` holds positional indices into ``detections`` (the table
    shares its order). When nothing is selected the user has not filtered, so
    every detection is returned; otherwise only the chosen rows are kept, in
    the order ``detections`` already has. Out-of-range indices are ignored so a
    stale selection from a previous image can never raise.
    """
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
