# NNGA — NN & Genetic Algorithms course (derived project)

> Apply this file **only when the user explicitly mentions the NN & Genetic
> Algorithms course (or "NNGA")**. By default the repo is the DL project
> (`.claude/CLAUDE.md`). This is a *derived* project — reusing the DL work across
> both courses is explicitly allowed. Deliverables live under `_NNGA/`.

## What the project is

Course variant: **"със синергизъм" (with synergism)** — a neural network must be
paired with a genetic algorithm. We reuse the DL project's **YOLOv8n** detector
and COCO data, and add the synergism:

> a **genetic algorithm evolves the detector's hyperparameters**, using the
> network's own **validation mAP as the fitness function**.

## Key points (the whole project in six lines)

- **Network:** YOLOv8n (one-stage detector), COCO-pretrained, from the DL project.
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
  0 → 0.037 → 0.0995 → **0.103**. Demonstrates the mechanism; the full run is the
  identical command with more generations on the COCO subset on a GPU.

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
uv run python _NNGA/src/evolve_hyperparameters.py --data coco8.yaml --epochs 3 --iterations 8
uv run python _NNGA/src/plot_evolution.py
```
