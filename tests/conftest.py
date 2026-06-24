"""Shared lightweight test doubles."""

from objdetect.models.base import Detection


class FakeDetector:
    name = "Fake"

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections if detections is not None else [
            Detection(label="dog", score=0.95, box=(10.0, 10.0, 100.0, 120.0)),
            Detection(label="person", score=0.80, box=(50.0, 30.0, 90.0, 200.0)),
            Detection(label="cat", score=0.40, box=(5.0, 5.0, 40.0, 60.0)),
        ]

    def predict(self, image, score_threshold: float = 0.5) -> list[Detection]:
        return [d for d in self._detections if d.score >= score_threshold]
