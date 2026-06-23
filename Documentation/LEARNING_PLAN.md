# Learning Plan — understand, run, and defend the project

Goal: by the end you can explain every design decision, run every command, and
answer defense questions. Each module is tied to the project phase it explains.
Estimated total: ~25–35 hours, spread across the project timeline.

## Module 1 — Object detection fundamentals (~4 h) → before Phase 1
Concepts: classification vs detection, bounding boxes, IoU, Non-Maximum
Suppression (NMS), anchors, mAP and how mAP@0.5 / mAP@0.5:0.95 are computed.
- Watch: a "mAP explained" + "NMS explained" video (short, visual ones).
- Read: Wikipedia Object Detection page (it's in the requirements — the
  examiner may reference it).
- Self-check: explain on paper how a detector's prediction is matched to a
  ground-truth box and when it counts as a true positive.

## Module 2 — The two model families (~5 h) → during Phases 3–4
- **Faster R-CNN** (two-stage): read the paper's intro + figures (Ren et al.
  2015), focus on: Region Proposal Network, RoI pooling, why two stages.
- **YOLO** (one-stage): YOLOv1 paper intro + a YOLOv8 architecture blog post;
  focus on: grid prediction, why it's faster, accuracy trade-off.
- Self-check: answer "why is YOLO faster but Faster R-CNN often more accurate
  on small objects?" — this is a classic defense question.

## Module 3 — COCO dataset & EDA (~3 h) → during Phase 2
- Browse cocodataset.org: the 80 classes, annotation JSON format, what
  "crowd" annotations are.
- Walk through the project's EDA notebook cell by cell; for every plot, write
  one sentence of interpretation in your own words.
- Self-check: name two anomalies found in the data and how they affect training.

## Module 4 — Training mechanics & LR schedules (~4 h) → during Phase 5
- Concepts: epochs, batches, loss curves, overfitting, fine-tuning vs training
  from scratch, why pretrained backbones.
- **LR schedules**: read PyTorch docs for `StepLR` and `CosineAnnealingLR`;
  understand the shape of each curve and why decaying LR helps convergence.
- Self-check: sketch both LR curves from memory and explain which performed
  better in our experiments and (hypothesize) why.

## Module 5 — PyTorch & the codebase (~6 h) → after Phases 3–5 are coded
- PyTorch basics: tensors, `Dataset`/`DataLoader`, a training loop's anatomy
  (forward, loss, backward, optimizer step).
- Code walkthrough: read `objdetect/` top-down — data → models → training →
  evaluation. For each file, write a one-line summary of what it does.
- Run everything yourself: download data, run a short training, run evaluation.
- Self-check: modify one hyperparameter (e.g. batch size) and rerun a smoke
  training without help.

## Module 6 — Streamlit & BDD testing (~4 h) → during Phase 6
- Streamlit: official "get started" tutorial (~1 h), then read our app code.
- BDD: what Gherkin Given/When/Then is, how `pytest-bdd` binds feature files to
  step functions; read our `.feature` files and step definitions side by side.
- Self-check: add one trivial scenario (e.g. "app shows an error for a corrupt
  image") yourself, watch it fail, make it pass.

## Module 7 — Presentation & defense prep (~3 h) → Phase 7
- Rehearse the demo: launch app, detect on 2–3 prepared images, show the
  model comparison and LR-schedule plots.
- Prepare answers to likely questions:
  - Why these two models? What's the trade-off between them?
  - What does mAP mean and what values did you get?
  - What did the LR-schedule experiment show?
  - What would you do with more time/compute?
  - Walk me through what happens when I click "Detect".
- Dry-run the full presentation once, timed.

## How to use this plan
Work module-by-module as the matching project phase lands; never let code get
more than one phase ahead of your understanding. Every self-check should be
done without looking at notes — those are exactly the defense questions.
