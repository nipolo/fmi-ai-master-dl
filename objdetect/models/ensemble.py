"""Ensemble detector: runs several detectors and merges their detections.

The app uses this for the combined "81-class" option, which pairs the stock
80-class YOLO26n with the single-class fine-tuned cone model so the UI can show
COCO objects and traffic cones together. Single-class fine-tuning makes the
cone model forget the 80 COCO classes, so neither model alone covers all 81 —
running both and concatenating their detections is what produces the union.
"""

from PIL.Image import Image

from objdetect.models.base import Detection, Detector


class EnsembleDetector:
    """Run several detectors on one image and concatenate their detections.

    Conforms to the project's ``Detector`` protocol (``name`` + ``predict``), so
    the Streamlit app, evaluation and visualization treat it like any other
    model and need no special-casing. The member classes are assumed disjoint
    (COCO's 80 vs. the cone class), so detections are simply concatenated; no
    cross-model NMS is applied.
    """

    def __init__(self, members: list[Detector], name: str = "Ensemble") -> None:
        self.members = members
        self.name = name

    def predict(self, image: Image, score_threshold: float = 0.5) -> list[Detection]:
        detections: list[Detection] = []
        for member in self.members:
            detections.extend(member.predict(image, score_threshold=score_threshold))
        return detections
