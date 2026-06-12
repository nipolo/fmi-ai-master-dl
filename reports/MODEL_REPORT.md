# Model Report — Object Detection on COCO

This is the project's **model report file** (a stated success criterion). It
records every experiment: the dataset slice, the configuration, the metrics,
and the conclusions, so the results are reproducible and presentable.

> How to regenerate everything in this report:
> ```bash
> uv run python scripts/download_data.py            # COCO val2017
> uv run python scripts/run_eda.py                  # EDA figures + summary
> uv run python scripts/plot_lr_schedules.py        # LR schedule figure
> uv run python scripts/evaluate.py                 # Faster R-CNN baseline mAP
> uv run python scripts/run_experiments.py          # fine-tune + LR comparison
> ```

## 1. Dataset

- **Source:** COCO 2017 validation split (5 000 images), the everyday-context
  benchmark named in the assignment.
- **Subset for fine-tuning** (`config.SUBSET_CLASSES`): person, bicycle, car,
  dog, cat, chair, bottle, cup, laptop, cell phone — ten common classes kept
  small so experiments fit the time and cloud budget.
- **Preprocessing:** crowd boxes and degenerate (≤1 px) boxes dropped; images
  scaled to float tensors in [0, 1]; horizontal-flip augmentation for training.
- See `EDA_REPORT.md` for the full data analysis (Requirement 2).

## 2. Models compared

| Model | Family | Backbone | Source |
|-------|--------|----------|--------|
| Faster R-CNN | Two-stage | ResNet-50 + FPN | torchvision, COCO-pretrained |
| YOLOv8n | One-stage | CSP/anchor-free | Ultralytics, COCO-pretrained |

Both start from COCO-pretrained weights. The pretrained models are evaluated as
baselines; Faster R-CNN is additionally fine-tuned on the subset to demonstrate
the training pipeline and the LR-schedule experiment.

## 3. Metrics

- **mAP@[.50:.95]** — COCO primary metric (averaged over 10 IoU thresholds).
- **mAP@.50** — the classic looser metric.
- **Inference speed** — images/second on the evaluation device.
- **Model size** — parameter count / weight file size.

<!-- Results tables are filled by scripts/run_experiments.py; see section 4-6. -->

## 4. Baseline results (pretrained, on the subset)

Measured by `scripts/benchmark_baselines.py` on 200 subset images (CPU/MPS),
through the shared `evaluate_detector_map` so both models go through one fair
evaluation path. FPS measured on the same machine; treat the absolute speeds as
relative, not hardware-spec figures.

| Model | mAP@[.50:.95] | mAP@.50 | mAP@.75 | FPS | Params |
|-------|:-------------:|:-------:|:-------:|:---:|:------:|
| Faster R-CNN (ResNet-50 FPN) | **0.467** | **0.699** | 0.513 | 2.8 | 41.8 M |
| YOLOv8n | 0.469 | 0.610 | **0.515** | **27.4** | 3.2 M |

**Reading the table — the project's central result:**

- **Speed/size:** YOLOv8n runs **~10× faster** and is **~13× smaller** than
  Faster R-CNN. This is the one-stage vs two-stage trade-off in raw form.
- **Accuracy:** Faster R-CNN clearly wins at the *loose* IoU threshold
  (mAP@.50: 0.70 vs 0.61) — it localizes and recalls objects more reliably,
  consistent with its two-stage "propose then refine" design and FPN.
- **The surprise worth discussing:** overall mAP@[.50:.95] is essentially
  *tied* (0.467 vs 0.469), and YOLOv8n even edges ahead at the *tight* IoU
  threshold (mAP@.75). Modern one-stage detectors have largely closed the
  accuracy gap; YOLOv8's boxes are very precise when it does fire, while Faster
  R-CNN's edge is in catching more objects at moderate overlap.

This single table is the heart of the presentation: *which detector you pick
depends on whether you are bounded by accuracy/recall or by latency/size.*

## 5. Fine-tuning results

The fine-tuning pipeline (`scripts/train.py`, `scripts/run_experiments.py`) was
validated end-to-end on the subset: the training loss decreases and checkpoints
and history are saved. Example smoke run (Faster R-CNN, cosine schedule, CPU):

```
epoch 1/3  lr=0.00500  loss=1.4337
epoch 2/3  lr=0.00375  loss=0.9316
epoch 3/3  lr=0.00126  loss=0.9344
```

The smoke configuration (few epochs, capped batches) exists to prove the
pipeline on a laptop. A full fine-tune (all subset images, ~10-15 epochs) is
intended for a CUDA GPU instance (AWS g5.xlarge) and is launched with the same
script minus `--max-batches`; see the LR-schedule comparison below, which uses
this pipeline.

## 6. Learning-rate schedule experiment (Requirement 5)

Two Faster R-CNN fine-tuning runs, **identical except for the LR schedule**
(same seed, epochs, base LR, data), produced by `scripts/run_experiments.py`.

**The schedules themselves** (`scripts/plot_lr_schedules.py`):

![LR schedules](figures/lr_schedules.png)

- **Step decay** holds the LR flat, then multiplies it by 0.1 every few epochs —
  a staircase.
- **Cosine annealing** glides the LR down a half-cosine from the base value to a
  small floor — smooth, spending longer near the extremes.

**The actual runs** (5 epochs, capped batches — a demonstration, not a converged
model):

![LR experiment](figures/lr_experiment_comparison.png)

| Schedule | LR per epoch | Final train loss |
|----------|--------------|:----------------:|
| Cosine annealing | 0.0050 → 0.0045 → 0.0033 → 0.0017 → 0.0005 | **0.742** |
| Step decay | 0.0050 → 0.0050 → 0.0050 → 0.0005 → 0.0005 | 0.786 |

**What the experiment shows:**

- The **LR-vs-epoch curves clearly differ**: the smooth cosine glide vs the step
  staircase — which is exactly what Requirement 5 asks to demonstrate.
- Both runs converge quickly from the pretrained initialization (loss drops from
  ~1.5 to ~0.75 within one epoch).
- Cosine reached a slightly lower final loss here (0.742 vs 0.786): its gentle
  late-epoch decay let training keep settling, whereas step decay holds a
  comparatively high LR until its scheduled drop.

> **Honesty note for the defense:** these are deliberately *small* runs (≈100
> training images per run, evaluated on 50), so the absolute mAP (~0.09) is far
> below the pretrained baseline in §4 — they exist to demonstrate the schedule
> mechanism on a laptop, not to beat the baseline. The full run (all subset
> images, 10-15 epochs) belongs on a cloud GPU and uses the identical command
> without `--max-batches`.

## 7. Conclusions

- **The accuracy/speed trade-off is real and measurable.** On the same COCO
  subset, YOLOv8n is ~10× faster and ~13× smaller than Faster R-CNN, while
  Faster R-CNN is more reliable at loose IoU (mAP@.50 0.70 vs 0.61). Overall
  mAP@[.50:.95] is essentially tied — modern one-stage detectors have closed the
  historical accuracy gap.
- **Pick the detector by your constraint:** latency/size → YOLO; maximum recall
  and small-object reliability → Faster R-CNN.
- **The training pipeline works end-to-end** and the LR-schedule experiment
  demonstrates both required schedules, with cosine annealing edging out step
  decay on final loss in these runs.
- **Everything is reproducible** from the commands at the top of this report and
  is covered by an automated test suite (exercise-style unit tests + BDD).
