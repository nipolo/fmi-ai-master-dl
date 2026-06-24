"""Central configuration: paths, the class subset, and reproducibility seeds."""

import os
import pathlib as pl

ROOT_DIR = pl.Path(__file__).resolve().parents[1]

DATA_DIR = pl.Path(os.environ.get("OBJDETECT_DATA_DIR", ROOT_DIR / "DATA"))
COCO_DIR = DATA_DIR / "coco"
COCO_IMAGES_DIR = COCO_DIR / "val2017"
COCO_ANNOTATIONS_FILE = COCO_DIR / "annotations" / "instances_val2017.json"

CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RESEARCH_DIR = ROOT_DIR / "research"
EDA_FIGURES_DIR = RESEARCH_DIR / "figures"
EDA_SUMMARY_FILE = RESEARCH_DIR / "eda_summary.json"

WEIGHTS_DIR = DATA_DIR / "weights"
YOLO_BASE_WEIGHTS = str(WEIGHTS_DIR / "yolo26n.pt")
CONE_WEIGHTS = str(WEIGHTS_DIR / "cone_yolo26n.pt")
CONE_WEIGHTS_20EP = str(WEIGHTS_DIR / "cone_yolo26n_20ep.pt")
CONE_FRCNN_WEIGHTS = str(WEIGHTS_DIR / "faster_rcnn_cone.pth")

TRAFFIC_CONE_DIR = DATA_DIR / "traffic_cone"
CONE_CLASS_NAMES = ["traffic cone"]
CONE_COCO_TRAIN = TRAFFIC_CONE_DIR / "annotations" / "instances_train.json"
CONE_COCO_VAL = TRAFFIC_CONE_DIR / "annotations" / "instances_val.json"
CONE_YAML = TRAFFIC_CONE_DIR / "traffic_cone.yaml"

COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

SUBSET_CLASSES = [
    "person",
    "bicycle",
    "car",
    "dog",
    "cat",
    "chair",
    "bottle",
    "cup",
    "laptop",
    "cell phone",
]

SEED = 42

DEFAULT_SCORE_THRESHOLD = 0.5
