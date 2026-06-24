"""Central configuration: paths, the class subset, and reproducibility seeds.

Everything that an experiment depends on lives here so that no notebook or
script carries magic constants.
"""

import os
import pathlib as pl

# --- Paths -----------------------------------------------------------------

# Repository root (this file lives in objdetect/config.py).
ROOT_DIR = pl.Path(__file__).resolve().parents[1]

# Heavy artifacts (datasets, weights) are git-ignored and live under DATA/.
DATA_DIR = pl.Path(os.environ.get("OBJDETECT_DATA_DIR", ROOT_DIR / "DATA"))
COCO_DIR = DATA_DIR / "coco"
COCO_IMAGES_DIR = COCO_DIR / "val2017"
COCO_ANNOTATIONS_FILE = COCO_DIR / "annotations" / "instances_val2017.json"

CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# Model / experiment artifacts (Requirements 3-5) live under reports/.
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Literature review + data exploration deliverables (Requirements 1-2) live
# together under research/; the EDA report and notebook quote these outputs.
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

# --- COCO download sources ---------------------------------------------------

COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

# --- Experiment subset -------------------------------------------------------

# Fine-tuning on all 80 COCO classes is out of budget for this project, so all
# training experiments use a fixed subset of everyday-context classes.  The
# Streamlit app still uses the full pretrained 80-class models.
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

# --- Reproducibility ---------------------------------------------------------

SEED = 42

# --- Inference defaults ------------------------------------------------------

DEFAULT_SCORE_THRESHOLD = 0.5
