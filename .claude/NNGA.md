# NNGA — NN & Genetic Algorithms course (derived project)

> Apply this file **only when the user explicitly mentions the NN & Genetic
> Algorithms course (or "NNGA")**. By default the repo is the DL project
> (`.claude/CLAUDE.md`). This is a *derived* project — reusing the DL work across
> both courses is explicitly allowed. Deliverables live under `_NNGA/`.

## What the project is

Course variant: **"със синергизъм" (with synergism)** — a neural network must be
paired with a genetic algorithm. We reuse the DL project's **YOLO26n** detector
and COCO data, and add the synergism:

> a **genetic algorithm evolves the detector's hyperparameters**, using the
> network's own **validation mAP as the fitness function**.

## Key points (the whole project in six lines)

- **Network:** YOLO26n (one-stage detector), COCO-pretrained, from the DL project.
- **GA tool:** Ultralytics' built-in evolutionary tuner (`YOLO.tune`) — chosen
  deliberately over a hand-written GA.
- **Genome:** ~20 training hyperparameters (`lr0`, `lrf`, momentum, weight decay,
  warmup, box/cls/dfl loss weights, HSV/translate/scale/flip/mosaic/mixup).
- **Fitness:** `0.1·mAP@0.5 + 0.9·mAP@0.5:0.95` — i.e. training+evaluating the
  network *is* how a genome is scored. GA and NN are inseparable = the synergism.
- **Operators:** mutation (Gaussian, ~80% of genes, then clipped) + weighted
  selection of the best prior genome. **No crossover** — it's an evolution
  strategy, not a textbook three-operator GA. State this precisely.
- **Result (smoke run, 8 generations, coco8, MPS):** best-so-far fitness rose
  0 → 0.0006 → 0.0074 → **0.0308** (peak at gen 7). Tiny/noisy by design (3 epochs
  on 8 images) — demonstrates the mechanism.
- **Real on-device run (done):** the same GA on the DL project's single-class
  **traffic-cone** dataset (`DATA/traffic_cone/traffic_cone.yaml`, DL
  `reports/MODEL_REPORT.md` §7), 20 generations × 10 epochs on the **Apple M3 (no
  GPU)**, ~77 min. Best-so-far fitness rose **0.392 → 0.505** (peak at gen 16:
  mAP@.5 ≈ 0.78, mAP@.5:.95 ≈ 0.505); winning genome rebalanced the loss (box ↓
  4.64, cls ↑ 0.70, dfl ↑ 1.74, lower lr0). A real, defensible result — the
  bridge to the DL project's §7 cone work. **Gotcha:** Ultralytics logs `fitness`
  == mAP@.5:.95 exactly here (not the 0.1/0.9 blend). All three figures
  (`ga_fitness_evolution.png`, `tune_fitness.png`, `tune_scatter_plots.png`) now
  show the cone run. An even-larger COCO-subset run is the identical command on a GPU.

## Deliverables (under `_NNGA/`)

- `PRESENTATION.md` — Marp deck, structured to the course slide's required bullets.
- `reports/GA_SYNERGISM_REPORT.md` — the written report.
- `src/evolve_hyperparameters.py` — the GA × NN synergism.
- `src/plot_evolution.py` — fitness-vs-generation figure.
- `reports/figures/`, `reports/tune_results.ndjson`, `reports/best_hyperparameters.yaml`.

## Conventions & gotchas

- English only; reuse the DL project's stack, `objdetect` package, and `SEED=42`.
- Heavy `_NNGA/runs/` and `*.pt` are git-ignored (regenerable).
- This Ultralytics version (8.4.66) writes results as **`tune_results.ndjson`**
  (not `tune_results.csv` as older docs say) — both scripts parse ndjson.
- Deadline: **2026-06-20**.

## Reproduce

```bash
# Smoke (mechanism only):
uv run python _NNGA/src/evolve_hyperparameters.py --data coco8.yaml --epochs 3 --iterations 8
# Real, on-device (recommended) — traffic-cone, Apple M3, no GPU:
uv run python _NNGA/src/evolve_hyperparameters.py \
    --data DATA/traffic_cone/traffic_cone.yaml --epochs 10 --iterations 20 --device mps
uv run python _NNGA/src/plot_evolution.py
```
