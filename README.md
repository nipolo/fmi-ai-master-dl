# Object Detection in Everyday Images

Deep Learning university course project — **Topic 4: Object Detection**.

A web application that detects objects in everyday photos, backed by experiments
comparing **Faster R-CNN** (two-stage) and **YOLO** (one-stage) on the **COCO**
dataset, with learning-rate-schedule demonstrations, behaviour-driven tests, and
a model report.

## What's here

| Path | What it is |
|------|-----------|
| `src/objdetect/` | the package: `data`, `models`, `training`, `evaluation`, `app` |
| `scripts/` | one-command entry points (download, EDA, train, evaluate, benchmark) |
| `notebooks/01_eda.ipynb` | interactive data exploration |
| `tests/` | exercise-style `unittest` + `pytest-bdd` BDD tests |
| `reports/` | literature review, EDA report, **model report**, presentation, figures |
| `Documentation/` | requirements PDF, project plan, **learning plan** |

## Requirements coverage

1. Literature review → `reports/LITERATURE_REVIEW.md`
2. COCO data exploration → `reports/EDA_REPORT.md`, `notebooks/01_eda.ipynb`
3. Faster R-CNN experiments → `src/objdetect/models/faster_rcnn.py`, benchmarks
4. YOLO experiments → `src/objdetect/models/yolo.py`, benchmarks
5. LR schedules (cosine annealing + step decay) → `src/objdetect/training/schedulers.py`
6. Streamlit UI + tests → `src/objdetect/app/`, `tests/`
7. Presentation → `reports/PRESENTATION.md`
- BDD → `tests/features/`, `tests/steps/`
- Model report → `reports/MODEL_REPORT.md`

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (uv installs it).

```bash
uv sync            # create the environment and install everything
uv run pytest      # run the test suite (should be all green)
```

## Common commands

```bash
# Data
uv run python scripts/download_data.py        # fetch COCO val2017 (~780 MB)
uv run python scripts/run_eda.py              # EDA figures + summary

# Models / experiments
uv run python scripts/benchmark_baselines.py --max-images 200   # mAP/FPS/size
uv run python scripts/train.py --scheduler cosine --epochs 6    # fine-tune
uv run python scripts/run_experiments.py                        # cosine vs step
uv run python scripts/plot_lr_schedules.py                      # LR curves figure
uv run python scripts/compare_models_visual.py                  # side-by-side image

# App
uv run streamlit run src/objdetect/app/main.py
```

## Where to start understanding it

Follow `Documentation/LEARNING_PLAN.md` — eight modules synced to the project
phases, each with self-check questions that double as defense questions.
