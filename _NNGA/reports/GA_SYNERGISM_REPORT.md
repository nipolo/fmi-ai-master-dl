# GA × NN Synergism Report — Evolving a YOLO Detector's Hyperparameters

**Course:** Neural Networks & Genetic Algorithms — variant *"със синергизъм"*
(with synergism). **Author:** Borislav Valkov.

This project reuses the object-detection neural network from the Deep Learning
course (Topic 4) and adds the synergism this course requires: a **genetic
algorithm optimizes the neural network's hyperparameters**, driven by the
network's own validation accuracy as fitness.

> Reproduce everything in this report:
> ```bash
> # GA evolution (smoke demo, runs on a laptop / Apple Silicon MPS):
> uv run python _NNGA/src/evolve_hyperparameters.py --data coco8.yaml --epochs 3 --iterations 8
> # Presentation-friendly fitness curve from the GA's results table:
> uv run python _NNGA/src/plot_evolution.py
> ```

## 1. The task

Object detection: given an everyday image, predict *what* objects are present and
*where* (bounding boxes). The neural network is **YOLOv8n** (Ultralytics), the
one-stage detector already studied and benchmarked in the DL project
(`reports/MODEL_REPORT.md`: mAP@.50:.95 ≈ 0.47 on the COCO subset, ~27 FPS,
3.2 M params). Here the network is the *substrate*; the new contribution is how
its hyperparameters are chosen.

## 2. The data

- **Source:** COCO (Common Objects in Context). Link: <https://cocodataset.org>.
  The DL project's EDA (`reports/EDA_REPORT.md`) characterises it: 80 classes,
  severe class imbalance (person 11 004 vs toaster 9 instances), 7.4 objects per
  image on average, and **46.7 % of objects smaller than 1 % of image area** —
  a hard, cluttered, small-object benchmark.
- **For this GA demo:** `coco8.yaml` — an 8-image COCO sample that ships with
  Ultralytics, in the same 80-class format. It makes each GA individual train in
  seconds so the full evolution loop runs end-to-end on a laptop. The **full
  run** uses the DL project's 10-class everyday subset (person, bicycle, car,
  dog, cat, chair, bottle, cup, laptop, cell phone) converted to YOLO format —
  same command, larger data, on a GPU.
- **Transformations / preliminary analysis:** unchanged from the DL project —
  crowd and degenerate (≤1 px) boxes dropped, images scaled to [0,1]. The
  conclusion from the EDA that *small objects and imbalance dominate* is exactly
  what makes hyperparameters like augmentation strength and learning rate worth
  evolving rather than guessing.

## 3. How the task is solved — the neural network

| | Value |
|--|--|
| Architecture | YOLOv8n — one-stage, anchor-free, CSPDarknet backbone + PAN neck |
| Parameters | ~3.2 M |
| Optimizer | AdamW |
| Init | COCO-pretrained weights, then fine-tuned per GA individual |
| Fixed | seed = 42 (shared with DL project `config.SEED`), epochs per individual |

## 4. The synergism — a genetic algorithm tunes the network

We use Ultralytics' built-in **evolutionary hyperparameter tuner** (`YOLO.tune`),
which is a genetic algorithm operating on hyperparameter *genomes*.

**What a genome is.** A vector of training hyperparameters — learning rate
(`lr0`), final-LR factor (`lrf`), momentum, weight decay, warmup, box/cls/dfl
loss weights, and augmentation strengths (HSV jitter, translate, scale, flip,
mosaic, mixup). ~20 genes in total.

**Fitness function.** Each genome is decoded by *training the YOLO network* with
those hyperparameters and then *evaluating it* on the validation set. Fitness is
Ultralytics' default detection score:

```
fitness = 0.1 · mAP@0.5 + 0.9 · mAP@0.5:0.95
```

This is the synergism: the **GA cannot score a genome without running the neural
network**, and the **network's hyperparameters come from the GA** — neither part
works alone.

**The GA loop** (one *iteration* = one generation):

1. **Selection** — pick the best previous genome(s) by fitness (weighted choice
   over the top results so far).
2. **Mutation** — Gaussian mutation: each gene is perturbed with ~80 %
   probability by a sampled gain, then clipped to its allowed range. (This tuner
   is mutation-driven — an evolutionary strategy, no crossover.)
3. **Evaluation** — train the network with the child genome, compute fitness.
4. **Record & repeat** — append to `tune_results.ndjson`; the best genome so far is
   saved to `best_hyperparameters.yaml`.

**Settings / conditions used** (the slide asks for this explicitly):

| Setting | Smoke demo | Full run (GPU) |
|--|--|--|
| Generations (`--iterations`) | 8 | 100 |
| Epochs per individual (`--epochs`) | 3 | 30 |
| Dataset | coco8 (8 imgs) | COCO 10-class subset |
| Mutation probability | 0.8 (Ultralytics default) | 0.8 |
| Selection | weighted top-k parents | weighted top-k parents |
| Device | Apple Silicon MPS | CUDA GPU |
| Seed | 42 | 42 |

## 5. Results

> The numbers below come from the **smoke demo** (8 generations on 8 images).
> They demonstrate the *mechanism* — a working GA→NN→fitness loop that records a
> population and selects a best genome — not a converged search. Like the DL
> project's smoke runs, the **full run** (100 generations on the subset) belongs
> on a GPU and uses the identical command.

**Fitness over generations** (`reports/figures/ga_fitness_evolution.png`):

![GA fitness evolution](figures/ga_fitness_evolution.png)

Ultralytics' own diagnostics are kept alongside:
`reports/figures/tune_fitness.png` (fitness per iteration) and
`reports/figures/tune_scatter_plots.png` (fitness vs each hyperparameter, which
shows *which* genes the search found to matter).

**The actual smoke run** (8 generations, 3 epochs each, coco8, Apple MPS, ~147 s):

| Generation | Genome fitness | mAP@.5 | mAP@.5:.95 | Best so far |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.0369 | 0.073 | 0.0369 | 0.0369 |
| 4 | 0.0206 | 0.053 | 0.0206 | 0.0369 |
| 5 | 0.0995 | 0.166 | 0.0995 | 0.0995 |
| 6 | 0.0240 | 0.055 | 0.0240 | 0.0995 |
| 7 | 0.0167 | 0.050 | 0.0167 | 0.0995 |
| 8 | **0.1027** | **0.218** | **0.1027** | **0.1027** |

The **best-so-far** curve rises monotonically (0 → 0.037 → 0.0995 → **0.1027**):
the GA found progressively better hyperparameter genomes, ending **~2.8×** above
its first non-zero genome. Individual generations bounce around (e.g. gen 7 =
0.017) — that is mutation *exploring*; selection keeps the best regardless. The
winning genome (`reports/best_hyperparameters.yaml`) shifted the box-loss weight
to **10.79** (from a 7.5 default), nudged augmentation (`hsv`, `translate`,
`scale`) and warmup — an automatically discovered configuration, not a guessed
one.


**What the results mean.**

- The GA evaluated a *population* of hyperparameter genomes and kept the best by
  validation fitness — i.e. it performed an automated, accuracy-driven
  hyperparameter search instead of manual trial-and-error.
- The best genome is saved in `reports/best_hyperparameters.yaml`; those values
  are what you would then use for a final, longer training run of the detector.
- The scatter plots reveal the *sensitivity* of fitness to each hyperparameter —
  learning rate and augmentation strengths are typically the high-leverage genes,
  consistent with the EDA finding that the data is small-object-heavy and
  imbalanced (so augmentation and LR matter).

## 6. How this compares to the DL project's approach (why a GA at all)

The DL project set the learning-rate **schedule** by hand (cosine vs step) and
fixed the other hyperparameters. That answers *"how should the LR change within a
run?"* The GA answers a different, complementary question: *"which
hyperparameters should we pick in the first place?"* — searched automatically by
evolution rather than chosen by intuition. The two are layers of the same
problem; this project adds the outer, evolutionary layer.

## 7. Technological description

- **Development environment:** Python 3.13.7, managed with `uv`; VS Code.
- **Library stack:** Ultralytics (YOLOv8 + the genetic tuner), PyTorch +
  torchvision, pycocotools, matplotlib; the existing `objdetect` package is
  reused for config/seed and the data subset definition.
- **Hardware:** smoke runs on Apple Silicon (MPS); the full evolution is intended
  for a CUDA GPU (AWS EC2 / Colab), budget ~\$20 as in the DL project.

## 8. Conclusions

- A genetic algorithm and an object-detection neural network were coupled into a
  working synergism: **the GA proposes hyperparameter genomes; the network
  scores them by validation mAP; the GA selects and mutates toward better ones.**
- The loop runs end-to-end and produces a population, a fitness trajectory, and a
  best genome — all reproducible from two `uv run` commands.
- It is a genuine, defensible NN+GA pairing (not a cosmetic add-on): the fitness
  signal *is* the network's accuracy, so the two components are inseparable.
- Reusing the DL project's data, EDA, and YOLO network kept the new work focused
  on the synergism the course is actually grading.
