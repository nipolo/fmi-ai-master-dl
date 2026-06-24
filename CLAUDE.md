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
- Training: local Apple Silicon (MPS); COCO subsets are sized to fine-tune
  on-device, no cloud GPU required.

## Project layout

Flat layout: the importable `objdetect` package lives at the repo root (no
`src/` wrapper). Run command-line modules with `python -m objdetect.cli.<group>.<name>`.

```
objdetect/            # the importable package (lives at repo root, not under src/)
  config.py           # central paths, class subset, seeds
  utils.py            # device / seed helpers
  visualization.py    # draw detections on images
  data/               # COCO download + dataset wrappers (coco.py)
  eda/                # COCO exploratory analysis (analysis.py) + report.py
  models/             # faster_rcnn.py, yolo.py, ensemble.py behind one interface
  training/           # train loop, LR schedulers (cosine annealing, step decay)
  evaluation/         # mAP / COCO metrics
  app/                # Streamlit application
  cli/                # entry points, grouped by purpose, run via
                      # `python -m objdetect.cli.<group>.<name>`:
                      #   data/       — download_data, prepare_cone_dataset, prepare_cone_coco
                      #   training/   — train, train_cone_frcnn, train_cone_yolo, run_experiments
                      #   evaluation/ — evaluate, benchmark_baselines
                      #   figures/    — plot_lr_schedules, compare_models_visual
research/             # Req. 1-2: LITERATURE_REVIEW.md, EDA_REPORT.md, 01_eda.ipynb, figures/
tests/
  features/           # Gherkin .feature files (BDD)
  steps/              # step definitions
reports/              # MODEL_REPORT.md, presentation, experiment results & training curves
Documentation/        # requirements PDF, project & learning plans
```

## Conventions

- Code, comments, docs, report, and presentation: **English only** (confirmed).
- **Markdown line width:** do not hard-wrap prose — write each paragraph, bullet,
  and blockquote on a single line and let the viewer wrap it. **Exceptions** (keep
  whatever wrapping suits them): presentation/slide files (e.g.
  `reports/PRESENTATION.md`), `CLAUDE.md` and other `.claude/` instruction files,
  and any file whose format depends on line breaks. This applies to prose only —
  never reflow tables, code blocks, or YAML/Gherkin.
- **Unit test style** (must match course exercises, see `Documentation/tests/`):
  `unittest.TestCase` classes, one class per unit named `Test<FunctionName>`,
  methods named `test_when_<condition>_then_<expected>`, with explicit
  `# Arrange / # Act / # Assert / # Clean` comment sections and
  `expected_*` / `actual_*` variable naming. BDD tests (Gherkin + `pytest-bdd`)
  live alongside in `tests/features/` + `tests/steps/`.
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
- `uv run streamlit run objdetect/app/main.py` — launch the app.
- `uv run python -m objdetect.cli.training.train --model faster_rcnn|yolo --scheduler cosine|step` — training entry point.
