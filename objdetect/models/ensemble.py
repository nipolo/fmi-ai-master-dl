"""Ensemble detector: runs several detectors and merges their detections."""

from PIL.Image import Image

from objdetect.models.base import Detection, Detector


class EnsembleDetector:
    """Run several detectors on one image and concatenate their detections."""

    def __init__(self, members: list[Detector], name: str = "Ensemble") -> None:
        self.members = members
        self.name = name

    def predict(self, image: Image, score_threshold: float = 0.5) -> list[Detection]:
        detections: list[Detection] = []
        for member in self.members:
            detections.extend(member.predict(image, score_threshold=score_threshold))
        return detections
