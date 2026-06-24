"""Fine-tune YOLO26n on the traffic-cone dataset (Req. 4 deep dive).

A thin wrapper over the Ultralytics trainer so the YOLO cone experiment is
reproducible the same way as the Faster R-CNN one (cli.train_cone_frcnn): same
CLI shape, and it copies its training curve + metrics into reports/ so the model
report stays in sync instead of relying on a hand-copy from DATA/runs/.

Prerequisite (run once): clone + lay out the dataset
    git clone --depth 1 https://github.com/krisstern/traffic-cone-image-dataset.git DATA/_cone_src
    uv run python -m objdetect.cli.prepare_cone_dataset

Run the fine-tune:
    uv run python -m objdetect.cli.train_cone_yolo --epochs 100 --device mps
    uv run python -m objdetect.cli.train_cone_yolo --epochs 5 --device cpu   # quick smoke

Writes reports/figures/cone_yolo_training.png + reports/cone_yolo_results.json.
Publish weights for the app:
    cp DATA/runs/cone_yolo26n/weights/best.pt DATA/weights/cone_yolo26n.pt
"""

import argparse
import json
import shutil

from ultralytics import YOLO

from objdetect import config


def _write_report_artifacts(run_dir, metrics) -> None:
    """Copy the training curve and write a metrics JSON into reports/."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "results.png", config.FIGURES_DIR / "cone_yolo_training.png")

    results = {
        "model": "yolo26n",
        "dataset": "traffic_cone (held-out val split)",
        "metrics": {
            "mAP": float(metrics.box.map),
            "mAP_50": float(metrics.box.map50),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
    }
    (config.REPORTS_DIR / "cone_yolo_results.json").write_text(json.dumps(results, indent=2))
    return results["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps", help="mps | cpu | cuda")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--name", default="cone_yolo26n")
    args = parser.parse_args()

    runs_dir = config.DATA_DIR / "runs"
    model = YOLO(config.YOLO_BASE_WEIGHTS)
    model.train(
        data=str(config.CONE_YAML),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, patience=args.patience, cache="disk",
        project=str(runs_dir), name=args.name, exist_ok=True,
    )

    metrics = model.val(data=str(config.CONE_YAML), split="val", device=args.device)
    summary = _write_report_artifacts(runs_dir / args.name, metrics)

    print("val metrics:", json.dumps(summary, indent=2))
    print(f"weights -> {runs_dir / args.name / 'weights' / 'best.pt'}")
    print(f"publish: cp {runs_dir / args.name / 'weights' / 'best.pt'} {config.CONE_WEIGHTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
