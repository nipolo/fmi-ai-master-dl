# GA × NN Synergism Report — Evolving a YOLO Detector's Hyperparameters

**Course:** Neural Networks & Genetic Algorithms — variant *"със синергизъм"* (with synergism). **Author:** Borislav Valkov.

This project reuses the object-detection neural network from the Deep Learning course (Topic 4) and adds the synergism this course requires: a **genetic algorithm optimizes the neural network's hyperparameters**, driven by the network's own validation accuracy as fitness.

> Reproduce everything in this report:
> ```bash
> # GA evolution (smoke demo — proves the mechanism, runs on a laptop / Apple MPS):
> uv run python _NNGA/src/evolve_hyperparameters.py --data coco8.yaml --epochs 3 --iterations 8
> # Real, on-device run on the DL project's traffic-cone dataset (Apple M3, no GPU):
> uv run python _NNGA/src/evolve_hyperparameters.py \
>     --data DATA/traffic_cone/traffic_cone.yaml --epochs 10 --iterations 20 --device mps
> # Presentation-friendly fitness curve from the GA's results table:
> uv run python _NNGA/src/plot_evolution.py
> ```

## 1. The task

Object detection: given an everyday image, predict *what* objects are present and *where* (bounding boxes). The neural network is **YOLO26n** (Ultralytics), the one-stage detector already studied and benchmarked in the DL project (`reports/MODEL_REPORT.md`: mAP@.50:.95 ≈ 0.47 on the COCO subset, ~57 FPS, 2.4 M params). Here the network is the *substrate*; the new contribution is how its hyperparameters are chosen.

## 2. The data

- **Source:** COCO (Common Objects in Context). Link: <https://cocodataset.org>. The DL project's EDA (`reports/EDA_REPORT.md`) characterises it: 80 classes, severe class imbalance (person 11 004 vs toaster 9 instances), 7.4 objects per image on average, and **46.7 % of objects smaller than 1 % of image area** — a hard, cluttered, small-object benchmark.
- **Smoke demo:** `coco8.yaml` — an 8-image COCO sample that ships with Ultralytics, in the same 80-class format. It makes each GA individual train in seconds so the full evolution loop runs end-to-end on a laptop — but on 8 images fitness stays tiny and noisy. It proves the *mechanism*, not a search.
- **Real, on-device run (recommended):** the DL project's **traffic-cone dataset** (added in `reports/MODEL_REPORT.md` §7). Link to the data: <https://github.com/krisstern/traffic-cone-image-dataset> — a single-class set, laid out into a seeded 80/20 split with a ready `DATA/traffic_cone/traffic_cone.yaml`. It is small enough that the **entire GA evolution runs on the laptop (Apple M3, no GPU)** — the DL project fully fine-tuned the *same* YOLO26n on it in ~32 min on-device — yet real enough that genomes earn meaningful, non-trivial fitness. This is the defensible "real" target; see the reproduce block above.
- **Even-larger run:** the DL project's 10-class everyday COCO subset (person, bicycle, car, dog, cat, chair, bottle, cup, laptop, cell phone) converted to YOLO format — same command, larger data, on a CUDA GPU.
- **Transformations / preliminary analysis:** unchanged from the DL project — crowd and degenerate (≤1 px) boxes dropped, images scaled to [0,1]. The conclusion from the EDA that *small objects and imbalance dominate* is exactly what makes hyperparameters like augmentation strength and learning rate worth evolving rather than guessing.

## 3. How the task is solved — the neural network

| | Value |
|--|--|
| Architecture | YOLO26n — one-stage, anchor-free; latest Ultralytics YOLO release |
| Parameters | ~2.4 M |
| Optimizer | AdamW |
| Init | COCO-pretrained weights, then fine-tuned per GA individual |
| Fixed | seed = 42 (shared with DL project `config.SEED`), epochs per individual |

## 4. The synergism — a genetic algorithm tunes the network

We use Ultralytics' built-in **evolutionary hyperparameter tuner** (`YOLO.tune`), which is a genetic algorithm operating on hyperparameter *genomes*.

**What a genome is.** A vector of training hyperparameters — learning rate (`lr0`), final-LR factor (`lrf`), momentum, weight decay, warmup, box/cls/dfl loss weights, and augmentation strengths (HSV jitter, translate, scale, flip, mosaic, mixup). ~20 genes in total.

**Fitness function.** Each genome is decoded by *training the YOLO network* with those hyperparameters and then *evaluating it* on the validation set. Fitness is Ultralytics' default detection score — in this version (8.4.66) the weights are hardcoded to `[0, 0, 0, 1]` over `[P, R, mAP@0.5, mAP@0.5:0.95]`, i.e.:

```
fitness = mAP@0.5:0.95
```

(Older YOLOv5-era docs describe a `0.1·mAP@0.5 + 0.9·mAP@0.5:0.95` blend; this release uses pure mAP@0.5:0.95 — see `ultralytics/utils/metrics.py:fitness`.)

This is the synergism: the **GA cannot score a genome without running the neural network**, and the **network's hyperparameters come from the GA** — neither part works alone.

**The GA loop** (one *iteration* = one generation). This is a **steady-state genetic algorithm**: every evaluated genome is kept in an ever-growing archive (`tune_results.ndjson`), and each iteration breeds and evaluates **one** new child from the best of that archive (no fixed generational population):

1. **Selection** — fitness-proportional ("roulette-wheel") pick of parents from the **top-9** genomes in the archive (`Tuner._crossover`).
2. **Crossover** — **BLX-α crossover**: blend the selected parents' genes within their min–max range (± a margin α).
3. **Mutation** — log-normal Gaussian mutation: each gene is perturbed with ~50 % probability (default `mutation=0.5`) by a sampled gain, then clipped to its allowed range. `sigma` decays 0.2 → 0.1 over the first 300 iterations.
4. **Evaluation** — train the network with the child genome, compute fitness.
5. **Record & repeat** — append to `tune_results.ndjson`; the best genome so far is saved to `best_hyperparameters.yaml`.

So it is a genuine **three-operator GA** (selection + crossover + mutation), not a crossover-free evolution strategy — verified directly in `ultralytics/engine/tuner.py` for this release (8.4.66).

**Settings / conditions used** (the slide asks for this explicitly):

| Setting | Smoke demo | Real run (on-device) | Even-larger run (GPU) |
|--|--|--|--|
| Generations (`--iterations`) | 8 | 20 | 100 |
| Epochs per individual (`--epochs`) | 3 | 10 | 30 |
| Dataset | coco8 (8 imgs) | traffic-cone (1 class) | COCO 10-class subset |
| Mutation probability | 0.5 (Ultralytics default) | 0.5 | 0.5 |
| Selection | weighted top-9 parents | weighted top-9 parents | weighted top-9 parents |
| Device | Apple Silicon MPS | Apple M3 (MPS) | CUDA GPU |
| Seed | 42 | 42 | 42 |

## 5. Results

The GA was run twice: a **coco8 smoke demo** that proves the loop works, and the **real on-device run** on the traffic-cone dataset that produces a defensible, non-trivial result. All the figures in this section come from the **cone run** (regenerated by `plot_evolution.py`).

### Smoke demo (coco8) — proves the mechanism

> Deliberately near-zero: 8 generations × 3 epochs on **8 images**. It shows a working GA→NN→fitness loop that records a population and selects a best genome — not a converged search.

| Generation | Genome fitness | mAP@.5 | mAP@.5:.95 | Best so far |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.0006 | 0.0059 | 0.0006 | 0.0006 |
| 4 | 0.0006 | 0.0025 | 0.0006 | 0.0006 |
| 5 | 0.000 | 0.000 | 0.000 | 0.0006 |
| 6 | 0.0074 | 0.0184 | 0.0074 | 0.0074 |
| 7 | **0.0308** | **0.0942** | **0.0308** | **0.0308** |
| 8 | 0.0047 | 0.0158 | 0.0047 | 0.0308 |

Best-so-far rises monotonically (0 → 0.0006 → 0.0074 → **0.0308**, peak at gen 7); individual generations bounce (gen 8 → 0.0047) — mutation *exploring*, selection keeping the best. The score is tiny by design (3 epochs, 8 images): the point is the *mechanism*, not the number.

### Real on-device run (traffic-cone) — the defensible result

Same GA, on the DL project's single-class traffic-cone dataset, **entirely on the Apple M3 (no GPU)** — 20 generations × 10 epochs, ~77 min total. Because cones are a small, visually distinct class, each genome trains to real accuracy, so fitness is meaningful and the selection pressure is clearly visible.

**Fitness over generations** (`reports/figures/ga_fitness_evolution.png`):

![GA fitness evolution](figures/ga_fitness_evolution.png)

Ultralytics' own diagnostics alongside: `reports/figures/tune_fitness.png` (fitness per iteration) and `reports/figures/tune_scatter_plots.png` (fitness vs each gene — *which* hyperparameters the search found to matter).

| Generation | Genome fitness | mAP@.5 | mAP@.5:.95 | Best so far |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 0.392 | 0.678 | 0.392 | 0.392 |
| 2 | 0.289 | 0.532 | 0.289 | 0.392 |
| 3 | 0.382 | 0.685 | 0.382 | 0.392 |
| 4 | 0.423 | 0.733 | 0.423 | 0.423 |
| 5 | 0.450 | 0.768 | 0.450 | 0.450 |
| 6 | 0.461 | 0.745 | 0.461 | 0.461 |
| 7 | 0.468 | 0.746 | 0.468 | 0.468 |
| 8 | 0.353 | 0.648 | 0.353 | 0.468 |
| 9 | 0.391 | 0.683 | 0.391 | 0.468 |
| 10 | 0.167 | 0.367 | 0.167 | 0.468 |
| 11 | 0.350 | 0.615 | 0.350 | 0.468 |
| 12 | 0.446 | 0.728 | 0.446 | 0.468 |
| 13 | 0.333 | 0.594 | 0.333 | 0.468 |
| 14 | 0.480 | 0.749 | 0.480 | 0.480 |
| 15 | 0.497 | 0.776 | 0.497 | 0.497 |
| 16 | **0.505** | **0.784** | **0.505** | **0.505** |
| 17 | 0.453 | 0.756 | 0.453 | 0.505 |
| 18 | 0.476 | 0.740 | 0.476 | 0.505 |
| 19 | 0.434 | 0.715 | 0.434 | 0.505 |
| 20 | 0.464 | 0.765 | 0.464 | 0.505 |

The **best-so-far** curve climbs 0.392 → 0.423 → 0.450 → 0.461 → 0.468 → 0.480 → 0.497 → **0.505** (peak at generation 16), then holds — a genuinely good single-class detector (**mAP@.5 ≈ 0.78, mAP@.5:.95 ≈ 0.505**) found by evolution, not by hand. Individual generations still bounce (gen 10 collapses to 0.167) — that is mutation *exploring*; selection keeps the best regardless. The winning genome (`reports/best_hyperparameters.yaml`, gen 16) **rebalanced the detection loss**: box weight **down to 4.64** (from the 7.5 default), `cls` **up to 0.70** and `dfl` **up to 1.74**, at a lower `lr0 ≈ 0.0054` with gentler mosaic (0.74) — evolution chose to weight *classification / box-quality* over raw box loss for this easy, single-class target. An automatically discovered configuration, not a guessed one.

> **Note on the fitness number.** Consistent with §4, the logged scalar `fitness` equals **mAP@.5:.95** exactly (the two columns are identical) — this release weights fitness `[0, 0, 0, 1]`, so the blend some docs mention does not apply. Either way it *is* the network's own validation accuracy — the synergism holds: no genome can be scored without training and validating the network.

> **Note on the epoch budget.** Each genome is scored after a **fixed, short training budget** (10 epochs per individual here). The winning genome is therefore optimal *for that budget*, **not guaranteed to be best for a longer final run** — short training favours fast-converging settings (higher LR, lighter regularisation) that can overfit or plateau over 30+ epochs, and the `lr0`/`lrf` schedule is normalised to the epoch count. Treat `best_hyperparameters.yaml` as a strong starting recipe to verify with one longer run, not a proven global optimum; evolving at a budget closer to the final epoch count (the reason the GPU tier uses 30 epochs/individual) narrows this gap at higher compute cost.

**What the results mean.**

- The GA evaluated a *population* of hyperparameter genomes and kept the best by validation fitness — i.e. it performed an automated, accuracy-driven hyperparameter search instead of manual trial-and-error.
- The best genome is saved in `reports/best_hyperparameters.yaml`; those values are what you would then use for a final, longer training run of the detector.
- The scatter plots reveal the *sensitivity* of fitness to each hyperparameter — the loss-weight genes (`box`, `cls`, `dfl`) and learning rate are the high-leverage ones here, consistent with the EDA finding that augmentation and LR matter on small-object, imbalanced data.

## 6. How this compares to the DL project's approach (why a GA at all)

The DL project set the learning-rate **schedule** by hand (cosine vs step) and fixed the other hyperparameters. That answers *"how should the LR change within a run?"* The GA answers a different, complementary question: *"which hyperparameters should we pick in the first place?"* — searched automatically by evolution rather than chosen by intuition. The two are layers of the same problem; this project adds the outer, evolutionary layer.

## 7. Technological description

- **Development environment:** Python 3.13.7, managed with `uv`; VS Code.
- **Library stack:** Ultralytics (YOLO26 + the genetic tuner), PyTorch + torchvision, pycocotools, matplotlib; the existing `objdetect` package is reused for config/seed and the data subset definition.
- **Hardware:** a **MacBook Air M3 (10-core GPU, 16 GB)** runs both the smoke demo *and* the recommended real run (the traffic-cone evolution) entirely on-device via MPS — no cloud needed, the same machine that fine-tuned the cone detector in ~32 min (DL `reports/MODEL_REPORT.md` §7). A CUDA GPU (AWS EC2 / Colab, budget ~\$20) is only needed for the even-larger COCO-subset run.

## 8. Conclusions

- A genetic algorithm and an object-detection neural network were coupled into a working synergism: **the GA proposes hyperparameter genomes; the network scores them by validation mAP; the GA selects and mutates toward better ones.**
- The loop runs end-to-end and produces a population, a fitness trajectory, and a best genome — all reproducible from two `uv run` commands.
- On a **real, on-device run** (traffic-cone, 20 generations on the Apple M3, no GPU) the search lifted best-so-far fitness to **mAP@.5:.95 ≈ 0.505** (mAP@.5 ≈ 0.78) at generation 16, rebalancing the detection-loss weights — a defensible result, not the near-zero coco8 smoke score.
- It is a genuine, defensible NN+GA pairing (not a cosmetic add-on): the fitness signal *is* the network's accuracy, so the two components are inseparable.
- Reusing the DL project's data, EDA, and YOLO network kept the new work focused on the synergism the course is actually grading.
