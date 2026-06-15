---
marp: true
title: Evolving an Object Detector with a Genetic Algorithm
paginate: true
---

# Evolving an Object Detector with a Genetic Algorithm

**Neural Networks & Genetic Algorithms — course work (variant *with synergism*)**

A genetic algorithm tunes a YOLO object-detection network on COCO

Borislav Valkov

> Render with: `marp _NNGA/PRESENTATION.md` (or the Marp VS Code extension).
> Figures live in `_NNGA/reports/figures/`.

---

## The task (каква е целта)

- **Object detection:** given an everyday image, find *what* objects are present
  **and** *where* (bounding boxes).
- The neural network is **YOLOv8n** — the one-stage detector from my Deep
  Learning project.
- **The new goal here:** don't hand-pick the network's hyperparameters —
  let a **genetic algorithm evolve them**, scored by the network's own accuracy.
- That coupling is the **synergism** this course asks for.

---

## The data — what it looks like (как изглеждат данните)

- **COCO** — Common Objects in Context. 80 classes, everyday scenes.
- 🔗 **Link to the data:** https://cocodataset.org
- From my EDA of COCO val2017 (4 952 images, 36 781 boxes):
  - **Severe imbalance** — person 11 004 vs toaster 9 instances (~1 200:1).
  - **Cluttered** — 7.4 objects per image on average (up to 63).
  - **Small-object-heavy** — 46.7 % of objects cover < 1 % of the image.

![w:380](reports/figures/eda_class_distribution.png)

---

## The data — transformations & conclusions

**Transformations / preliminary analysis (трансформации и анализи):**

- Dropped **crowd** boxes and degenerate (≤1 px) boxes.
- Images scaled to [0, 1]; standard YOLO augmentation (HSV, flip, mosaic…).

**Conclusions (изводи):**

- Because the data is **imbalanced and small-object-heavy**, hyperparameters
  like **learning rate** and **augmentation strength** strongly affect accuracy.
- → They are worth **searching automatically**, not guessing — motivating the GA.

---

## How I solve it — the network (НМ, архитектура, хиперпараметри)

| | |
|--|--|
| Network | **YOLOv8n** — one-stage, anchor-free |
| Backbone / neck | CSPDarknet + PAN |
| Parameters | ~3.2 M |
| Optimizer | AdamW, COCO-pretrained init |
| Evolved hyperparameters | `lr0`, `lrf`, momentum, weight decay, warmup, box/cls/dfl loss weights, HSV/translate/scale/flip/mosaic/mixup |

The ~20 evolved hyperparameters form one **genome**.

---

## The synergism — a GA tunes the network (какъв синергизъм)

**The loop (one iteration = one generation):**

1. **Selection** — pick the best previous genome(s) by fitness.
2. **Mutation** — Gaussian perturbation of ~80 % of genes, then clip to range.
3. **Evaluation** — *train the YOLO network* with that genome, *measure mAP*.
4. **Record & repeat** — keep the best-so-far genome.

**Fitness = 0.1 · mAP@0.5 + 0.9 · mAP@0.5:0.95**

> The GA can't score a genome without running the network; the network's
> hyperparameters come from the GA. **Neither works alone — that's the synergism.**

---

## The synergism — settings & conditions (настройки, условия)

| Setting | Smoke demo | Full run (GPU) |
|--|--|--|
| Generations | 8 | 100 |
| Epochs / individual | 3 | 30 |
| Dataset | coco8 (8 imgs) | COCO 10-class subset |
| Mutation prob. | 0.8 | 0.8 |
| Selection | weighted top-k parents | weighted top-k parents |
| Device | Apple MPS | CUDA GPU |
| Seed | 42 | 42 |

Tool: Ultralytics' built-in evolutionary tuner (`YOLO.tune`) — a mutation-driven
genetic algorithm.

---

## Results — the GA converging (какви резултати)

![w:620](reports/figures/ga_fitness_evolution.png)

- The GA evaluated a **population** of genomes and tracked the **best so far**.
- The best genome is saved to `reports/best_hyperparameters.yaml` — the
  hyperparameters to use for a final training run.

- Smoke run: **8 generations**, best fitness **0.000 → 0.103** (best-so-far rises
  monotonically; the winning genome raised box-loss weight to 10.8 and tuned
  augmentation).

---

## Results — what the search learned (какво означават)

![w:760](reports/figures/tune_scatter_plots.png)

- Fitness vs each gene shows **which hyperparameters matter**.
- **Learning rate** and **augmentation strengths** are the high-leverage genes —
  exactly what the EDA (imbalanced, small objects) predicted.
- The GA replaces manual trial-and-error with an **accuracy-driven search**.

---

## Honesty note (for the defense)

- The shown run is a **smoke demo**: 8 generations × 3 epochs on **8 images**.
- It proves the **mechanism** — a working GA → network → fitness → selection
  loop — **not** a converged, production search.
- The **full run** (100 generations on the COCO subset) is the *identical*
  command on a GPU. Same honesty discipline as the DL project's smoke runs.

---

## Technological description (технологично описание)

- **Environment:** Python 3.13, managed with `uv`; VS Code.
- **Library stack:** Ultralytics (YOLOv8 + genetic tuner), PyTorch /
  torchvision, pycocotools, matplotlib. Reuses my `objdetect` package for
  config, seed, and the class subset.
- **Hardware:** Apple Silicon (MPS) for smoke runs; CUDA GPU (AWS / Colab,
  budget ~\$20) for the full evolution.

---

## Summary

- Coupled a **genetic algorithm** with an **object-detection network** into a
  real synergism: **GA proposes hyperparameters → network scores them by mAP →
  GA selects & mutates toward better ones.**
- End-to-end and reproducible from two `uv run` commands.
- Built on top of my Deep Learning project — data, EDA, and YOLO network reused;
  the new, graded contribution is the **evolutionary hyperparameter search**.

**Reproduce →** `uv run python _NNGA/src/evolve_hyperparameters.py --data coco8.yaml`
