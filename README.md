# Object Detection in Everyday Images

Deep Learning university course project — **Topic 4: Object Detection**.

A web application that detects objects in everyday photos, backed by experiments comparing **Faster R-CNN** (two-stage) and **YOLO** (one-stage) on the **COCO** dataset, with learning-rate-schedule demonstrations, behaviour-driven tests, and a model report.

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

## What's here

| Path                     | What it is                                                             |
| ------------------------ | ---------------------------------------------------------------------- |
| `src/objdetect/`         | the package: `data`, `models`, `training`, `evaluation`, `app`         |
| `scripts/`               | one-command entry points (download, EDA, train, evaluate, benchmark)   |
| `notebooks/01_eda.ipynb` | interactive data exploration                                           |
| `tests/`                 | exercise-style `unittest` + `pytest-bdd` BDD tests                     |
| `reports/`               | literature review, EDA report, **model report**, presentation, figures |
| `Documentation/`         | requirements PDF, project plan, **learning plan**                      |

## Setup

### 1. Install uv

This project is managed with [uv](https://docs.astral.sh/uv/). Install it once
(see the [official guide](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
```

uv reads `pyproject.toml` (declared dependencies) and `uv.lock` (exact pinned versions) and also fetches the right Python — 3.13.7, pinned in `.python-version`.

### 2. Create the environment (.venv)

```bash
uv sync            # create .venv/ and install everything (runtime + dev deps)
uv run pytest      # run the test suite (should be all green)
```

If you prefer an activated shell, `source .venv/bin/activate` (and `deactivate` to exit) works too.

### 3. Notebooks

Select the `.venv` interpreter as the Jupyter kernel (VS Code: kernel picker → Python Environments → `.venv`); the system Python won't have the project installed, so its cells will fail on the first import.

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

## VS Code

Pre-configured tasks and debug profiles live in `.vscode/`.

**Tasks** (`Cmd+Shift+P` → _Tasks: Run Task_):

| Task      | What it does                                                |
| --------- | ----------------------------------------------------------- |
| setup     | build `.venv` and install all dependencies (`uv sync`)      |
| run:app   | launch the Streamlit web app                                |
| run:eda   | generate EDA figures + summary from COCO data               |
| run:tests | run the BDD + unit suite (also the default _Run Test Task_) |

**Debugging** (Run and Debug panel, or `F5`) — uses the `.venv` interpreter:

| Configuration                   | What it debugs                        |
| ------------------------------- | ------------------------------------- |
| Debug current Python file       | the file open in the editor           |
| Debug script: train.py (cosine) | `scripts/train.py` with example args  |
| Debug Streamlit app             | the Streamlit app, breakpoints active |

Make sure the `.venv` interpreter is selected (`Cmd+Shift+P` → _Python: Select Interpreter_ → `.venv`) so tasks and the debugger use the project environment.
