"""Fine-tune YOLO26n on the traffic-cone dataset."""

import argparse
import json
import shutil

from ultralytics import YOLO

from objdetect import config


def _write_report_artifacts(run_dir, metrics, epochs) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{epochs}ep"
    shutil.copy2(run_dir / "results.png", config.FIGURES_DIR / f"cone_yolo_training{suffix}.png")

    results = {
        "model": "yolo26n",
        "epochs": epochs,
        "dataset": "traffic_cone (held-out val split)",
        "metrics": {
            "mAP": float(metrics.box.map),
            "mAP_50": float(metrics.box.map50),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
    }
    (config.REPORTS_DIR / f"cone_yolo_results{suffix}.json").write_text(json.dumps(results, indent=2))
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
    summary = _write_report_artifacts(runs_dir / args.name, metrics, args.epochs)

    print("val metrics:", json.dumps(summary, indent=2))
    print(f"weights -> {runs_dir / args.name / 'weights' / 'best.pt'}")
    print(f"publish: cp {runs_dir / args.name / 'weights' / 'best.pt'} {config.CONE_WEIGHTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
