# DL-Project — Object Detection (University Deep Learning Course)

University Deep Learning course project: **Topic 4 — Object Detection**.
Goal: a web application that detects objects in everyday-context images, backed by
experiments with Faster R-CNN and YOLO-class models on the COCO dataset.
Requirements source: `Documentation/Project Requirements.pdf` (in Bulgarian).

> **Default scope = this Deep Learning course.** A second, *derived* project for
> the **NN & Genetic Algorithms** course also lives in this repo (under `_NNGA/`).
> Its instructions are **not** loaded by default — they live in `.claude/NNGA.md`.
> Apply them **only when the user explicitly mentions the NN & Genetic Algorithms
> course (or NNGA)**; otherwise assume everything refers to this DL project.

## Mandatory requirements (from the PDF)

1. Literature review of object-detection papers/techniques.
2. Data exploration of COCO — counts, feature types, statistical dependencies,
   anomalies — reported with visualizations and commentary.
3. Experiments with **Faster R-CNN**.
4. Experiments with **YOLO-class models**.
5. Demonstrate LR scheduling across epochs: **cosine annealing** and **step decay**.
6. **Streamlit** user interface + tests (written in the course-exercise style).
7. Presentation describing the approach.

Additional success criteria (user's own):
- **Behavior-driven development** (Gherkin feature files + step definitions).
- A **model report file** (`reports/MODEL_REPORT.md`).

## Tech stack

- Python 3.13.7 (latest version fully supported by the DL ecosystem), managed with `uv`.
- PyTorch + torchvision (Faster R-CNN), `ultralytics` (YOLO), `pycocotools`.
- Streamlit for the UI.
- `pytest` + `pytest-bdd` for BDD tests.
- matplotlib / seaborn for EDA visualizations.
- Training: local Apple Silicon (MPS) for smoke runs; AWS EC2 GPU (or similar
  cloud) for real fine-tuning runs on COCO subsets.

## Project layout

```
src/objdetect/        # importable package
  data/               # COCO download, dataset wrappers, EDA utilities
  models/             # faster_rcnn.py, yolo.py wrappers
  training/           # train loop, LR schedulers (cosine annealing, step decay)
  evaluation/         # mAP / COCO metrics
  app/                # Streamlit application
notebooks/            # EDA and experiment notebooks
tests/
  features/           # Gherkin .feature files (BDD)
  steps/              # step definitions
reports/              # MODEL_REPORT.md, EDA report, training curves
scripts/              # train / evaluate / download entry points
Documentation/        # requirements PDF, project & learning plans
```

## Conventions

- Code, comments, docs, report, and presentation: **English only** (confirmed).
- **Unit test style** (must match course exercises, see `Documentation/tests/`):
  `unittest.TestCase` classes, one class per unit named `Test<FunctionName>`,
  methods named `test_when_<condition>_then_<expected>`, with explicit
  `# Arrange / # Act / # Assert / # Clean` comment sections and
  `expected_*` / `actual_*` variable naming. BDD tests (Gherkin + `pytest-bdd`)
  live alongside in `tests/features/` + `tests/steps/`.
- Cloud budget approved: up to ~$20 on AWS if local MPS training is too slow.
- Deadline: presentation ~2026-06-20; code must be complete by ~2026-06-15 so
  the user has 5 days to study it.
- Every model experiment gets logged into `reports/MODEL_REPORT.md` (config,
  dataset slice, metrics: mAP@0.5, mAP@0.5:0.95, training curves).
- Keep training runs reproducible: seeds fixed, configs in files, no magic
  constants in notebooks.
- Heavy artifacts (datasets, weights, runs) are git-ignored; only small result
  images/CSVs go into `reports/`.
- Tests must pass with `uv run pytest` before any milestone is considered done.
- The project will be presented and defended by the user — prefer clear,
  well-explained code over clever code; every non-obvious DL decision should be
  explainable from the learning plan in `Documentation/LEARNING_PLAN.md`.

## Commands (once scaffolded)

- `uv sync` — install environment.
- `uv run pytest` — run all tests (BDD + unit).
- `uv run streamlit run src/objdetect/app/main.py` — launch the app.
- `uv run python scripts/train.py --model faster_rcnn|yolo --scheduler cosine|step` — training entry point.
