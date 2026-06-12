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

_To be filled by running `scripts/evaluate.py` and the YOLO evaluation._

| Model | mAP@[.50:.95] | mAP@.50 | FPS | Params |
|-------|---------------|---------|-----|--------|
| Faster R-CNN | _tbd_ | _tbd_ | _tbd_ | ~41 M |
| YOLOv8n | _tbd_ | _tbd_ | _tbd_ | ~3 M |

## 5. Fine-tuning results

_To be filled after the fine-tune run (cloud GPU for the full run)._

## 6. Learning-rate schedule experiment (Requirement 5)

Two fine-tuning runs, identical except for the schedule, demonstrate cosine
annealing vs step decay.

![LR schedules](figures/lr_schedules.png)

_Loss/mAP-vs-epoch comparison and discussion to be filled after the runs._

## 7. Conclusions

_The accuracy/speed trade-off story, filled once the numbers are in:_
Faster R-CNN is expected to lead on mAP (especially small objects) while YOLOv8n
is expected to be several times faster and far smaller — the classic two-stage
vs one-stage trade-off.
