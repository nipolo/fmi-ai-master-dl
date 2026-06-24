# Literature Review — Object Detection

## 1. The task

**Object detection** answers two questions about an image at once: *what* objects are present (classification) and *where* each one is (localization, as an axis-aligned bounding box). It is harder than image classification because the number of objects is unknown in advance and the same scene can contain many objects at many scales. The canonical benchmark is **COCO** (Common Objects in Context), 80 everyday classes annotated with bounding boxes; performance is reported as **mean Average Precision (mAP)**.

## 2. Core concepts (vocabulary used throughout the project)

- **Bounding box**: four numbers `(x1, y1, x2, y2)` locating an object.
- **Intersection over Union (IoU)**: area of overlap divided by area of union between two boxes. A prediction is a *true positive* when its IoU with a ground-truth box of the same class exceeds a threshold (0.5 is the classic
  PASCAL VOC threshold).
- **Non-Maximum Suppression (NMS)**: detectors emit many overlapping boxes for the same object; NMS keeps the highest-scoring box and discards others that overlap it above an IoU cutoff. Used by Faster R-CNN; YOLO26's default one-to-one head is end-to-end (NMS-free).
- **Anchors**: pre-defined reference boxes of several scales and aspect ratios, tiled across the image, that the network refines rather than predicting boxes from scratch. Faster R-CNN's RPN is anchor-based; modern YOLO (incl. YOLO26) is anchor-free.
- **mAP**: Average Precision (area under the precision–recall curve) averaged over classes. COCO's primary metric, `mAP@[.50:.95]`, additionally averages over ten IoU thresholds from 0.50 to 0.95, rewarding tight localization.

## 3. Two families of detectors

### 3.1 Two-stage detectors (the R-CNN line)

The idea: first *propose* regions that might contain objects, then *classify and refine* each proposal.

- **R-CNN** (Girshick et al., 2014) ran a CNN on ~2000 externally-proposed region crops — accurate but extremely slow.
- **Fast R-CNN** (Girshick, 2015) ran the CNN once over the whole image and pooled features per region (RoI pooling), sharing computation.
- **Faster R-CNN** (Ren et al., 2015) replaced the external proposal step with a learned **Region Proposal Network (RPN)** sharing the backbone's features, making the whole detector a single end-to-end network. This is the model the project uses for the two-stage experiments.
- **Feature Pyramid Network (FPN)** (Lin et al., 2017) adds a top-down pathway so the detector sees features at multiple resolutions, which sharply improves small-object detection. The torchvision model used here is `fasterrcnn_resnet50_fpn`: a ResNet-50 backbone with FPN.

Two-stage detectors are generally **more accurate**, especially on small objects, because the second stage examines each proposal closely.

### 3.2 One-stage detectors (the YOLO / SSD line)

The idea: skip the proposal stage and predict boxes and classes directly from a dense grid in a single forward pass.

- **YOLO** ("You Only Look Once", Redmon et al., 2016) divides the image into a grid and each cell predicts boxes and class probabilities at once — far faster, enabling real-time detection.
- **SSD** (Liu et al., 2016) predicts from multiple feature-map scales for better scale handling.
- **Focal Loss / RetinaNet** (Lin et al., 2017) addressed one-stage detectors' weakness — the overwhelming number of easy background anchors — letting a one-stage model match two-stage accuracy.
- **Modern YOLO** (v3 through v8/v11 up to YOLO26; Ultralytics) brought anchor-free heads, better backbones/necks (CSP, PANet), and strong training tricks (mosaic augmentation). The project uses **YOLO26** (nano) via Ultralytics — the latest release in the family.

One-stage detectors are generally **faster and lighter**, at some accuracy cost that modern versions have largely closed.

The assignment requires experiments with **both** Faster R-CNN and **YOLO** — the textbook representatives of these two families — so comparing them tells the central story of the field: **the accuracy/speed trade-off**. The model report quantifies it on the same COCO subset using mAP, inference speed (FPS), and model size.

## 4. Training techniques used

- **Transfer learning / fine-tuning**: both models start from COCO-pretrained weights. YOLO26 is evaluated directly as a pretrained baseline; Faster R-CNN is additionally fine-tuned — its classifier head is replaced with a fresh one sized for the 10-class subset, then the whole network (backbone included) is trained, rather than training from scratch. This is standard practice and essential given the compute budget.
- **Learning-rate scheduling**: the learning rate is decayed as training proceeds — large early steps to make progress, small later steps to settle into a minimum. The project demonstrates two schedules:
  - **Step decay**: multiply the LR by a factor every *k* epochs (a staircase).
  - **Cosine annealing** (Loshchilov & Hutter, 2017): follow a half-cosine from the initial LR down to a small floor — a smooth decay shown to improve convergence in many settings.

## 5. Dataset

**COCO 2017** (Lin et al., 2014): 80 classes of common objects in natural context. The project uses the validation split (5 000 images) restricted to a 10-class everyday-context subset for affordable fine-tuning. The exploratory analysis (see `EDA_REPORT.md`) characterizes class imbalance, object sizes, scene density, class co-occurrence, and annotation anomalies.

## 6. Key references

- Girshick et al., *Rich feature hierarchies for accurate object detection* (R-CNN), CVPR 2014 — https://arxiv.org/abs/1311.2524
- Girshick, *Fast R-CNN*, ICCV 2015 — https://arxiv.org/abs/1504.08083
- Ren, He, Girshick, Sun, *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS 2015 — https://arxiv.org/abs/1506.01497
- Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017 — https://arxiv.org/abs/1612.03144
- Redmon, Divvala, Girshick, Farhadi, *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016 — https://arxiv.org/abs/1506.02640
- Ultralytics, *YOLO26* (model used in this project) — https://docs.ultralytics.com/models/yolo26/
- Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014 — https://arxiv.org/abs/1405.0312
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine annealing), ICLR 2017 — https://arxiv.org/abs/1608.03983
- Wikipedia, *Object detection* — https://en.wikipedia.org/wiki/Object_detection

## 7. Course materials

- *Лекция 10 (Object detection)* — https://learn.fmi.uni-sofia.bg/mod/resource/view.php?id=379128
- Simeon Hristov, exercise notes — *Object detection* — https://github.com/SimeonHristov99/DL_25-26/blob/main/notes.md#week-05---object-detection
