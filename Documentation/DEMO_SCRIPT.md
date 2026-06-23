# Demo & Defense Script — presentation day

A step-by-step runsheet so the live demo cannot surprise you. Rehearse it once
end-to-end (Learning Plan, Module 8).

## Before you walk in

```bash
cd DL-Project
uv sync                      # ensure env is ready
uv run pytest                # confirm all green in front of nobody first
```

Pre-download model weights once (so the demo isn't waiting on a download):
```bash
uv run python scripts/compare_models_visual.py   # pulls both model weights
```

Have ready: 2-3 everyday photos on disk (a street scene, a desk, a pet). The
side-by-side slide figure is built from the committed demo photo
(`reports/figures/photo-to-compare-models.JPG`).

## The talk (≈12 min, follows reports/PRESENTATION.md)

1. Problem & the two detector families (slides 1-4).
2. Data: show the class-imbalance and object-size figures, name the anomalies
   and how you filter them (slides 5-6).
3. Results table — say the headline out loud: *"YOLO is ~18× faster and ~17×
   smaller; Faster R-CNN is more reliable at loose IoU; overall mAP is tied."*
   (slide 7).
4. Show the side-by-side detection image (slide 8).
5. LR schedules: explain cosine vs step from the curve (slide 9).
6. The app + testing approach, incl. BDD (slide 10).

## The live demo (≈3 min)

```bash
uv run streamlit run src/objdetect/app/main.py
```
1. Upload the street scene → model = Faster R-CNN → point out boxes + table.
2. Switch model to YOLO26n on the same image → note it's faster, fewer/cleaner
   boxes.
3. Drag the confidence slider up → fewer detections; explain the precision/recall
   trade-off.

Fallback if the app misbehaves: open `reports/figures/model_comparison_visual.png`
and walk through it instead.

## Likely defense questions (have answers ready)

- *What is mAP?* Average Precision (area under precision-recall) averaged over
  classes; COCO's mAP also averages IoU 0.50-0.95.
- *Why is YOLO faster?* One forward pass over a grid vs propose-then-classify.
- *Why might Faster R-CNN be better on small objects?* FPN multi-scale features
  + a dedicated second stage examining each proposal.
- *What did the LR experiment show?* Both decay the LR with epochs; cosine is
  smooth, step is a staircase — show the curves, state which gave lower loss.
- *Walk me through a click of "Detect".* `main.py` → `inference.run_detection`
  → `model.predict` → `visualization.draw_detections` → table + counts.
- *What would you improve with more compute?* More epochs on a larger subset, a
  bigger YOLO variant, per-class mAP.
```
