"""Object detection on everyday-context images (COCO).

University Deep Learning course project — Topic 4.

The package is organised by responsibility:

- ``data``       — COCO download, dataset wrappers, exploratory data analysis.
- ``models``     — Faster R-CNN (two-stage) and YOLO (one-stage) wrappers
                   behind one common ``predict`` interface.
- ``training``   — fine-tuning loop and learning-rate schedulers
                   (cosine annealing, step decay).
- ``evaluation`` — COCO-style mAP metrics.
- ``app``        — Streamlit web application.
"""

__version__ = "0.1.0"


def main() -> int:
    """Console entry point: print a short usage guide."""
    print(
        "objdetect — Deep Learning course project (object detection)\n\n"
        "Usage:\n"
        "  uv run python scripts/download_data.py   # fetch COCO val2017\n"
        "  uv run python scripts/train.py --help    # fine-tune a model\n"
        "  uv run python scripts/evaluate.py --help # measure mAP\n"
        "  uv run streamlit run src/objdetect/app/main.py  # launch the app\n"
        "  uv run pytest                            # run the test suite"
    )
    return 0
