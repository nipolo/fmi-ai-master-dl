"""Generate the COCO EDA figures and a data summary (Requirement 2).

Runs the analysis functions in objdetect.eda over the downloaded COCO
val2017 annotations, saves plots into research/figures/, and writes a JSON
summary the EDA report quotes. Keeping this in the package (not only a
notebook) means the analysis is reproducible with one command and is covered
by tests.

Usage:  uv run python -m objdetect.eda.report
"""

import json

import matplotlib.pyplot as plt
import seaborn as sns

from objdetect import config
from objdetect.data.coco import load_coco_api
from objdetect import eda


def main() -> int:
    sns.set_theme(style="whitegrid")
    config.EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    coco = load_coco_api()
    annotations = eda.annotations_frame(coco)
    distribution = eda.class_distribution(annotations)
    per_image = eda.boxes_per_image(annotations)
    anomalies = eda.find_anomalies(annotations)

    # --- Figure 1: class distribution (imbalance) ---------------------------
    plt.figure(figsize=(12, 6))
    top = distribution.head(25)
    sns.barplot(data=top, x="instances", y="category", hue="category", legend=False,
                palette="viridis")
    plt.title("COCO val2017 — instances per class (top 25)")
    plt.xlabel("Number of annotated instances")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(config.EDA_FIGURES_DIR / "eda_class_distribution.png", dpi=120)
    plt.close()

    # --- Figure 2: boxes per image (scene density) --------------------------
    plt.figure(figsize=(9, 5))
    sns.histplot(per_image, bins=range(0, per_image.max() + 2), color="steelblue")
    plt.title("Objects per image")
    plt.xlabel("Boxes in image")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(config.EDA_FIGURES_DIR / "eda_boxes_per_image.png", dpi=120)
    plt.close()

    # --- Figure 3: relative box area (object scale) -------------------------
    plt.figure(figsize=(9, 5))
    sns.histplot(annotations["relative_area"].clip(upper=0.5), bins=50,
                 color="indianred")
    plt.title("Object size relative to image area")
    plt.xlabel("Box area / image area")
    plt.ylabel("Number of objects")
    plt.tight_layout()
    plt.savefig(config.EDA_FIGURES_DIR / "eda_relative_area.png", dpi=120)
    plt.close()

    # --- Figure 4: class co-occurrence (statistical dependencies) -----------
    cooc = eda.class_cooccurrence(annotations, top_n=15)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooc, annot=False, cmap="magma", square=True)
    plt.title("Class co-occurrence in the same image (top 15 classes)")
    plt.tight_layout()
    plt.savefig(config.EDA_FIGURES_DIR / "eda_cooccurrence.png", dpi=120)
    plt.close()

    # --- Numeric summary the report quotes ----------------------------------
    summary = {
        "num_images": int(annotations["image_id"].nunique()),
        "num_annotations": int(len(annotations)),
        "num_classes": int(annotations["category"].nunique()),
        "mean_boxes_per_image": round(float(per_image.mean()), 2),
        "median_boxes_per_image": int(per_image.median()),
        "max_boxes_per_image": int(per_image.max()),
        "most_common_class": distribution.iloc[0]["category"],
        "most_common_class_instances": int(distribution.iloc[0]["instances"]),
        "rarest_class": distribution.iloc[-1]["category"],
        "rarest_class_instances": int(distribution.iloc[-1]["instances"]),
        "imbalance_ratio": round(
            distribution.iloc[0]["instances"] / max(distribution.iloc[-1]["instances"], 1),
            1,
        ),
        "small_objects_frac": round(
            float((annotations["relative_area"] < 0.01).mean()), 3
        ),
        "anomaly_counts": anomalies["anomaly"].value_counts().to_dict(),
    }
    out = config.EDA_SUMMARY_FILE
    out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nFigures saved to {config.EDA_FIGURES_DIR}")
    print(f"Summary saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
