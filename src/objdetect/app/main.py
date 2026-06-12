"""Streamlit web application for object detection (Requirement 6).

Run with:  uv run streamlit run src/objdetect/app/main.py

The UI lets the user upload an everyday photo, pick a detector, set a
confidence threshold, and see the detected objects drawn on the image with a
table and per-class counts. All real work lives in ``inference.py`` so this
file stays a thin presentation layer.
"""

import streamlit as st
from PIL import Image

from objdetect.app.inference import (
    AVAILABLE_MODELS,
    detections_to_frame,
    load_model,
    run_detection,
    summarize_counts,
)
from objdetect.visualization import draw_detections


@st.cache_resource(show_spinner="Loading model…")
def _get_model(display_name: str):
    """Cache one instance per model so weights load only once per session."""
    return load_model(display_name)


def main() -> None:
    st.set_page_config(page_title="Object Detection", page_icon="🔍", layout="wide")
    st.title("🔍 Object Detection in Everyday Images")
    st.caption(
        "Deep Learning course project — compare a two-stage detector "
        "(Faster R-CNN) with a one-stage detector (YOLO) on your own photos."
    )

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Model", list(AVAILABLE_MODELS))
        score_threshold = st.slider(
            "Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )
        st.markdown(
            "**Two-stage** (Faster R-CNN): more accurate, slower.\n\n"
            "**One-stage** (YOLO): faster, lighter."
        )

    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is None:
        st.info("Upload a photo to run detection.")
        return

    image = Image.open(uploaded).convert("RGB")
    model = _get_model(model_name)

    with st.spinner("Running detection…"):
        detections = run_detection(model, image, score_threshold=score_threshold)

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with right:
        st.subheader(f"Detections — {model.name}")
        st.image(draw_detections(image, detections), use_container_width=True)

    if not detections:
        st.warning("No objects detected above the current confidence threshold.")
        return

    counts = summarize_counts(detections)
    st.success(
        f"Found {len(detections)} object(s): "
        + ", ".join(f"{n}× {label}" for label, n in counts.items())
    )
    st.dataframe(detections_to_frame(detections), use_container_width=True)


if __name__ == "__main__":
    main()
