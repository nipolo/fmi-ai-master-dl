# NNGA — Evolving an Object Detector with a Genetic Algorithm

Course work for **Neural Networks & Genetic Algorithms**, variant with synergism. It is a **derived project**: it reuses the object-detection neural network, COCO data, and EDA from the Deep Learning course project and adds the synergism.

> a **genetic algorithm** searches the detector's hyperparameters, using the **network's validation mAP as the fitness function**.

## Layout

```
_NNGA/
  README.md                      # this file
  PRESENTATION.md                
  src/
    evolve_hyperparameters.py    # the GA × NN synergism (Ultralytics genetic tuner)
    plot_evolution.py            # fitness-vs-generation figure from the GA results
  reports/
    GA_SYNERGISM_REPORT.md       # the full written report
    tune_results.ndjson          # GA population & fitness per generation (generated)
    best_hyperparameters.yaml    # best genome found (generated)
    figures/                     # fitness curve, scatter plots, EDA chart
  runs/                          # raw Ultralytics output (git-ignored, regenerable)
```

## Reproduce

> **Derived project / standalone note.** This folder ships as part of the parent Deep Learning repo and the commands below are meant to be run from its root (where `uv` and the `objdetect` package are set up). The two scripts themselves are standalone — they need only `ultralytics`, `torch` and `matplotlib`, and `evolve_hyperparameters.py` falls back to seed 42 if `objdetect` is absent — but faithfully reproducing the **full** run (especially the traffic-cone dataset, laid out by the parent repo's `scripts/prepare_cone_dataset.py`) expects that parent environment. The committed results, figures and `best_hyperparameters.yaml` let the report and presentation stand on their own without rerunning anything.

From the repository root (the `objdetect` env is already set up — `uv sync` if not):

```bash
# 1) Run the genetic algorithm (smoke demo — minutes on Apple Silicon / CPU):
uv run python _NNGA/src/evolve_hyperparameters.py --data coco8.yaml --epochs 3 --iterations 8

# 2) Make the presentation fitness curve from the GA's results table:
uv run python _NNGA/src/plot_evolution.py

# Real, on-device run on the DL project's traffic-cone dataset (Apple M3, no GPU).
# Small single-class set -> the whole evolution runs on the laptop and earns
# meaningful (non-toy) fitness. ~1-1.5 h for 20 generations; scale down freely:
uv run python _NNGA/src/evolve_hyperparameters.py \
    --data DATA/traffic_cone/traffic_cone.yaml --epochs 10 --iterations 20 --device mps

# Even-larger run (a CUDA GPU, identical command otherwise):
uv run python _NNGA/src/evolve_hyperparameters.py --data <coco-subset>.yaml \
    --epochs 30 --iterations 100 --device 0
```

> The traffic-cone dataset is laid out by the DL project's `scripts/prepare_cone_dataset.py` into `DATA/traffic_cone/` (see DL `reports/MODEL_REPORT.md` §7). Run that once if the folder is absent.

The GA writes `tune_results.ndjson`, `best_hyperparameters.yaml`, and fitness plots, which the script copies into `reports/`.

## Relationship to the DL project

This is intentionally a thin, focused layer on top of the DL project. The neural network (YOLO26n), the COCO data, the EDA, and now the DL project's **traffic-cone dataset** (its §7 single-class extension, reused here as a small on-device GA target) are shared; the only new idea is the **evolutionary hyperparameter search**. Submitting a derived project across both courses is permitted for this course.
