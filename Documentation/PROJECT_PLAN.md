# Project Plan — Object Detection Web App (Topic 4)

Maps every mandatory requirement from `Project Requirements.pdf` to a concrete
deliverable. Phases are ordered so each one produces something runnable and
presentable.

## Phase 0 — Scaffolding
- `uv`-managed Python 3.13.7 project, `pyproject.toml`, package skeleton
  `src/objdetect/`, `tests/`, `reports/`, `.gitignore` for datasets/weights.
- CI-style local check: `uv run pytest` green from day one.
- **Deliverable:** runnable empty project, first git commits.

## Phase 1 — Literature review (Req. 1)
- Short written survey in `reports/LITERATURE_REVIEW.md`: two-stage detectors
  (R-CNN → Fast R-CNN → Faster R-CNN), one-stage detectors (YOLO v1 → v8/v11,
  SSD), key concepts (IoU, anchors, NMS, mAP), and the COCO benchmark.
- **Deliverable:** 3–5 page survey with citations, reused later in the presentation.

## Phase 2 — Data exploration of COCO (Req. 2)
- Download COCO 2017 `val` split (~1 GB, 5k images) + annotations; define a
  fixed training subset (e.g. 10–20 everyday-context classes) so experiments
  stay affordable.
- EDA notebook + `reports/EDA_REPORT.md`: image counts, class distribution and
  imbalance, boxes-per-image, box size/aspect-ratio distributions, co-occurrence
  of classes (statistical dependencies), anomalies (crowd boxes, tiny/degenerate
  boxes, mislabeled samples), all with plots and written commentary.
- **Deliverable:** EDA report with visualizations — directly satisfies Req. 2.

## Phase 3 — Faster R-CNN experiments (Req. 3)
- torchvision `fasterrcnn_resnet50_fpn` — first as pretrained baseline
  (evaluate mAP on our subset), then fine-tuned on the subset.
- Training script with config files, fixed seeds, checkpointing.
- Cloud: smoke-test locally on MPS, full fine-tune on an AWS EC2 GPU instance
  (g5.xlarge spot, ~$0.50/h, a few hours total) — only if local proves too slow.
- **Deliverable:** baseline + fine-tuned metrics in `reports/MODEL_REPORT.md`.

## Phase 4 — YOLO experiments (Req. 4)
- Ultralytics YOLO (v8n/s and one larger variant) — pretrained baseline +
  fine-tune on the same subset for a fair comparison with Faster R-CNN.
- Comparison table: mAP@0.5, mAP@0.5:0.95, inference speed (FPS), model size.
- **Deliverable:** comparison section in `MODEL_REPORT.md` (speed/accuracy trade-off
  story for the presentation).

## Phase 5 — LR scheduling demos (Req. 5)
- Same model + subset trained twice: **cosine annealing** vs **step decay**
  (PyTorch `CosineAnnealingLR` / `StepLR`), identical seeds and epochs.
- Plots: LR-vs-epoch curves and loss/mAP-vs-epoch for both schedules, with
  written interpretation.
- **Deliverable:** dedicated section in `MODEL_REPORT.md` with the plots.

## Phase 6 — Streamlit app + BDD tests (Req. 6 + user criteria)
- Streamlit app: upload an image (or pick a sample), choose model
  (Faster R-CNN / YOLO), confidence-threshold slider, rendered boxes + labels,
  detection table, model-comparison view.
- **BDD:** Gherkin `.feature` files (e.g. "Detecting objects in an uploaded
  photo") with `pytest-bdd` step definitions; plus unit tests for data and
  inference utilities. Style must match course exercises — needs a sample from
  the user (see open questions).
- **Deliverable:** working app, green test suite.

## Phase 7 — Model report + presentation (Req. 7)
- Finalize `reports/MODEL_REPORT.md`: dataset, methodology, all experiments,
  metrics tables, curves, conclusions.
- Presentation deck (~15 slides) following the phases above, plus a live demo
  script: launch the app, run detection, show the comparison.
- **Deliverable:** deck + rehearsed demo flow.

## Open questions (answer before Phase 0)
1. **Exercise test style** — Req. 6 says tests "written the way shown in
   exercises". Share one exercise test file so the suite matches; BDD will be
   layered on top.
2. **Cloud budget** — is AWS OK (est. $5–20 total), or prefer free tiers
   (Kaggle/Colab GPUs)? Local Mac MPS may suffice for the small subset.
3. **Language** — report and presentation in Bulgarian, English, or both?
4. **Deadline** — to pace the phases.
