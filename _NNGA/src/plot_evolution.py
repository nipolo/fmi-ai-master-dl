"""Turn the GA tuner's `tune_results.ndjson` into a clean fitness-vs-generation plot."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

NNGA_DIR = Path(__file__).resolve().parents[1]
RESULTS_FILE = NNGA_DIR / "reports" / "tune_results.ndjson"
OUT_FIG = NNGA_DIR / "reports" / "figures" / "ga_fitness_evolution.png"


def _load_fitness(path: Path) -> list[float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r.get("iteration", 0))
    return [float(r["fitness"]) for r in rows]


def main() -> int:
    if not RESULTS_FILE.exists():
        raise SystemExit(
            f"{RESULTS_FILE} not found — run evolve_hyperparameters.py first."
        )

    fitness = _load_fitness(RESULTS_FILE)
    generations = range(1, len(fitness) + 1)
    best_so_far = []
    running = float("-inf")
    for value in fitness:
        running = max(running, value)
        best_so_far.append(running)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(generations, fitness, color="#888", label="genome fitness", zorder=3)
    ax.plot(
        generations,
        best_so_far,
        color="#1f77b4",
        marker="o",
        label="best so far",
        zorder=2,
    )
    ax.set(
        title="Genetic-algorithm hyperparameter evolution",
        xlabel="Generation (hyperparameter genome)",
        ylabel="Fitness = mAP@.5:.95",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=120)
    print(f"Saved {OUT_FIG.relative_to(NNGA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
