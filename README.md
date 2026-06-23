# Object Detection in Everyday Images

Deep Learning university course project — **Topic 4: Object Detection**.

A web application that detects objects in everyday photos, backed by experiments comparing **Faster R-CNN** (two-stage) and **YOLO** (one-stage) on the **COCO** dataset, with learning-rate-schedule demonstrations, behaviour-driven tests, and a model report.

## Requirements coverage

1. Literature review → `research/LITERATURE_REVIEW.md`
2. COCO data exploration → `research/EDA_REPORT.md`, `research/01_eda.ipynb`
3. Faster R-CNN experiments → `objdetect/models/faster_rcnn.py`, benchmarks
4. YOLO experiments → `objdetect/models/yolo.py`, benchmarks
5. LR schedules (cosine annealing + step decay) → `objdetect/training/schedulers.py`
6. Streamlit UI + tests → `objdetect/app/`, `tests/`
7. Presentation → `reports/PRESENTATION.md`

- BDD → `tests/features/`, `tests/steps/`
- Model report → `reports/MODEL_REPORT.md`

## What's here

| Path                     | What it is                                                             |
| ------------------------ | ---------------------------------------------------------------------- |
| `objdetect/`                   | the package: `data`, `eda`, `models`, `training`, `evaluation`, `app`, `cli` (+ `config.py`) |
| `objdetect/cli/`               | one-command entry points, run via `python -m objdetect.cli.<name>` (download, train, evaluate, benchmark) |
| `research/`              | Req. 1–2: literature review, EDA report + `01_eda.ipynb`, figures      |
| `tests/`                 | `unittest` + `pytest-bdd` BDD tests                     |
| `reports/`               | **model report**, presentation, experiment results + figures          |
| `Documentation/`         | requirements PDF                      |

## Setup

### 1. Install uv

This project is managed with [uv](https://docs.astral.sh/uv/). Install it once (see the [official guide](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
```

uv reads `pyproject.toml` (declared dependencies) and `uv.lock` (exact pinned versions) and also fetches the right Python — 3.13.7, pinned in `.python-version`.

### 2. Create the environment (.venv)

```bash
uv sync            # create .venv/ and install everything (runtime + dev deps)
uv run pytest      # run the test suite (should be all green)
```

### 3. Notebooks

Select the `.venv` interpreter as the Jupyter kernel.

## Common commands

```bash
# Data
uv run python -m objdetect.cli.download_data        # fetch COCO val2017 (~780 MB)
uv run python -m objdetect.eda.report    # EDA figures + summary

# Models / experiments
uv run python -m objdetect.cli.benchmark_baselines --max-images 200   # mAP/FPS/size
uv run python -m objdetect.cli.train --scheduler cosine --epochs 6    # fine-tune
uv run python -m objdetect.cli.run_experiments                        # cosine vs step
uv run python -m objdetect.cli.plot_lr_schedules                      # LR curves figure
uv run python -m objdetect.cli.compare_models_visual                  # side-by-side image

# App
uv run streamlit run objdetect/app/main.py
```

On first run each model downloads its pretrained weights automatically (YOLO via ultralytics, Faster R-CNN via torchvision) — needs internet once; cached afterwards. Only the data and EDA/training/benchmark commands need the COCO dataset (`download_data.py`); the app runs on uploaded photos without it.

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
| Debug script: objdetect.cli.train (cosine) | `objdetect.cli.train` with example args  |
| Debug Streamlit app             | the Streamlit app, breakpoints active |
