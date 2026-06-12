# Cloud Training Guide (optional, for the full fine-tune)

The local Mac (M3, MPS) is enough for the EDA, baselines, the app, and the
small LR-schedule demonstration runs. You only need the cloud if you want a
**fully converged** fine-tune on the whole subset over many epochs. This guide
is the cheapest sensible path; budget ~$5-15.

## When you need it

You do **not** need the cloud to satisfy the requirements — the baselines and
the LR-schedule demonstration already run locally. Use it only to produce a
fine-tuned model that beats the pretrained baseline, as a stretch result.

## AWS EC2 (GPU) — recommended instance

- **g5.xlarge** (1× NVIDIA A10G, 24 GB) — spot price ≈ $0.40-0.60/h.
- AMI: "Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)" — CUDA preinstalled.

## Steps

```bash
# 1. Launch the instance (console or CLI), then SSH in.
ssh -i your-key.pem ubuntu@<instance-ip>

# 2. Get the code + uv on the box.
git clone <your-repo-url> DL-Project && cd DL-Project
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv sync

# 3. Data: download COCO on the instance (fast network there).
uv run python scripts/download_data.py

# 4. Run inside tmux so it survives disconnects.
tmux new -s train
# Full run: no --max-batches, more epochs, CUDA auto-detected.
uv run python scripts/run_experiments.py --epochs 12 --max-eval-images 500
# detach with Ctrl-b d ; reattach with: tmux attach -t train

# 5. Pull results back to your laptop.
#    (run this FROM your laptop)
scp -i your-key.pem -r ubuntu@<instance-ip>:~/DL-Project/reports/ ./reports-cloud/
scp -i your-key.pem -r ubuntu@<instance-ip>:~/DL-Project/DATA/checkpoints/ ./DATA/

# 6. IMPORTANT — stop the instance so it stops charging.
#    Stop (keep disk) or Terminate (delete) from the console/CLI.
```

## Cost control checklist

- Use a **spot** instance (much cheaper; fine for a few-hour job).
- **Stop or terminate** the instance the moment training finishes — this is the
  one thing that actually costs money.
- A 12-epoch run on the subset is a few hours at most → a few dollars.

## Free alternatives

- **Google Colab / Kaggle**: free T4 GPUs. Upload the repo, `uv sync`, and run
  the same `scripts/run_experiments.py`. Slower and session-limited, but $0.
  Good enough for this project's subset.
