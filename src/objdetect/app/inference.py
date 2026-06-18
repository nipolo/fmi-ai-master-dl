"""App-facing inference helpers, kept free of any Streamlit imports.

Separating this logic from ``main.py`` is what makes the user interface
testable: the BDD and unit tests drive these functions directly, exactly as
the Streamlit callbacks do, without spinning up a browser.
"""

import pandas as pd
from PIL.Image import Image

from objdetect.models import build_detector
from objdetect.models.base import Detection, Detector

# Friendly label -> (builder name, kwargs) for the models the app offers.
AVAILABLE_MODELS = {
    "Faster R-CNN (two-stage)": ("faster_rcnn", {}),
    "YOLO26n (one-stage)": ("yolo", {"weights": "yolo26n.pt"}),
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


def summarize_counts(detections: list[Detection]) -> dict[str, int]:
    """Count detections per class label (the app's summary line)."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return counts
