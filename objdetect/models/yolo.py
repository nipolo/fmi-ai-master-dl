"""YOLO detector (one-stage) built on Ultralytics."""

from PIL.Image import Image

from objdetect.models.base import Detection
from objdetect.utils import resolve_device


class YOLODetector:
    """Ultralytics YOLO wrapped to the project's common detector interface."""

    name = "YOLO"

    def __init__(
        self,
        weights: str = "yolo26n.pt",
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.device = str(resolve_device(device))
        self.model = YOLO(weights)
        self.class_names = self.model.names

    def predict(self, image: Image, score_threshold: float = 0.5) -> list[Detection]:
        results = self.model.predict(
            image, conf=score_threshold, device=self.device, verbose=False
        )
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        label=self.class_names[int(box.cls)],
                        score=float(box.conf),
                        box=(x1, y1, x2, y2),
                    )
                )
        return detections
