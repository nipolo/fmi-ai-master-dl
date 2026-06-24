"""Faster R-CNN detector (two-stage) built on torchvision."""

import torch
from PIL.Image import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.v2 import functional as F

from objdetect.models.base import Detection
from objdetect.utils import resolve_device


class FasterRCNNDetector:
    """torchvision Faster R-CNN with a ResNet-50 FPN backbone."""

    name = "Faster R-CNN"

    def __init__(
        self,
        num_classes: int | None = None,
        device: str | torch.device | None = None,
        weights_path: str | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.device = resolve_device(device)

        if num_classes is None:
            self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
            self.class_names = list(self.weights.meta["categories"])
        else:
            self.weights = None
            self.model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes + 1
            )
            self.class_names = ["__background__"] + [
                str(i) for i in range(1, num_classes + 1)
            ]

        if weights_path is not None:
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )

        if class_names is not None:
            self.set_class_names(class_names)

        self.model.to(self.device)
        self.model.eval()

    def set_class_names(self, names: list[str]) -> None:
        """Provide readable names for a fine-tuned model (index = label id)."""
        self.class_names = ["__background__"] + list(names)

    @torch.no_grad()
    def predict(self, image: Image, score_threshold: float = 0.5) -> list[Detection]:
        tensor = F.to_dtype(F.to_image(image.convert("RGB")), torch.float32, scale=True)
        prediction = self.model([tensor.to(self.device)])[0]

        detections = []
        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            if score < score_threshold:
                continue
            detections.append(
                Detection(
                    label=self.class_names[int(label)],
                    score=float(score),
                    box=tuple(round(float(v), 1) for v in box),
                )
            )
        return detections
