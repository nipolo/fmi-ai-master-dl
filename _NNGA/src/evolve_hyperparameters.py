"""Evolve a YOLO detector's hyperparameters with a genetic algorithm.

Drives Ultralytics' built-in evolutionary tuner (`YOLO.tune`): each generation
breeds a hyperparameter genome (selection + BLX-alpha crossover + mutation),
trains YOLO with it, and scores it by validation mAP (fitness = mAP@0.5:0.95).
The best genome and the tuner's result table/plots are copied into
`_NNGA/reports/`.

Usage:
    uv run python _NNGA/src/evolve_hyperparameters.py \
        --data DATA/traffic_cone/traffic_cone.yaml --epochs 10 --iterations 20 --device mps
"""

import argparse
import shutil
from pathlib import Path

# Reuse the DL project's single source of truth for the random seed so the two
# projects stay reproducible in the same way.
try:
    from objdetect import config

    DEFAULT_SEED = config.SEED
except Exception:  # pragma: no cover - objdetect optional when run standalone
    DEFAULT_SEED = 42

NNGA_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = NNGA_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RUNS_DIR = NNGA_DIR / "runs"

# Artifacts the Ultralytics GA tuner writes that we want to keep with the report.
# (Recent Ultralytics writes the results table as newline-delimited JSON.)
_TUNE_ARTIFACTS = {
    "tune_results.ndjson": REPORTS_DIR / "tune_results.ndjson",
    "best_hyperparameters.yaml": REPORTS_DIR / "best_hyperparameters.yaml",
    "tune_fitness.png": FIGURES_DIR / "tune_fitness.png",
    "tune_scatter_plots.png": FIGURES_DIR / "tune_scatter_plots.png",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="yolo26n.pt", help="base YOLO weights")
    parser.add_argument("--data", default="coco8.yaml", help="Ultralytics dataset YAML")
    parser.add_argument(
        "--epochs", type=int, default=3, help="training epochs per GA individual"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="number of GA generations (hyperparameter genomes evaluated)",
    )
    parser.add_argument(
        "--device", default=None, help="'mps', 'cpu', or CUDA index like '0'"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # Imported lazily so `--help` is instant and the heavy import only happens on
    # a real run.
    from ultralytics import YOLO

    device = args.device
    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(
        f"GA hyperparameter evolution\n"
        f"  base network : {args.model}\n"
        f"  dataset      : {args.data}\n"
        f"  generations  : {args.iterations}  (epochs/individual: {args.epochs})\n"
        f"  device       : {device}   seed: {args.seed}\n"
    )

    model = YOLO(args.model)
    # `tune` IS the genetic algorithm: mutate -> train -> score -> select.
    model.tune(
        data=args.data,
        epochs=args.epochs,
        iterations=args.iterations,
        optimizer="AdamW",
        seed=args.seed,
        device=device,
        plots=True,
        save=False,
        val=True,
        project=str(RUNS_DIR),
        name="ga_evolution",
        exist_ok=True,
    )

    candidates = sorted(RUNS_DIR.glob("ga_evolution*"), key=lambda p: p.stat().st_mtime)
    tune_dir = candidates[-1] if candidates else RUNS_DIR / "ga_evolution"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for src_name, dest in _TUNE_ARTIFACTS.items():
        src = tune_dir / src_name
        if src.exists():
            shutil.copy2(src, dest)
            copied.append(dest.relative_to(NNGA_DIR))
    print("\nSaved GA artifacts:")
    for path in copied:
        print(f"  _NNGA/{path}")
    if not copied:
        print(f"  (none found under {tune_dir} — check the run above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
