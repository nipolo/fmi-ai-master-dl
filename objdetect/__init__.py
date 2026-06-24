"""Object detection on everyday-context images (COCO).

University Deep Learning course project — Topic 4.

The package is organised by responsibility:

- ``data``       — COCO download and dataset wrappers.
- ``eda``        — exploratory data analysis and the EDA report generator.
- ``models``     — Faster R-CNN (two-stage) and YOLO (one-stage) wrappers
                   behind one common ``predict`` interface.
- ``training``   — fine-tuning loop and learning-rate schedulers
                   (cosine annealing, step decay).
- ``evaluation`` — COCO-style mAP metrics.
- ``app``        — Streamlit web application.
- ``cli``        — command-line entry points (``python -m objdetect.cli.<group>.<name>``).
"""

__version__ = "0.1.0"


def main() -> int:
    """Console entry point: print a short usage guide."""
    print(
        "objdetect — Deep Learning course project (object detection)\n\n"
        "Usage:\n"
        "  uv run python -m objdetect.cli.data.download_data   # fetch COCO val2017\n"
        "  uv run python -m objdetect.cli.training.train --help    # fine-tune a model\n"
        "  uv run python -m objdetect.cli.evaluation.evaluate --help # measure mAP\n"
        "  uv run python -m objdetect.eda.report          # EDA figures + summary\n"
        "  uv run streamlit run objdetect/app/main.py     # launch the app\n"
        "  uv run pytest                                  # run the test suite"
    )
    return 0
