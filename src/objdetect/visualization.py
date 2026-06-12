"""Draw detections onto images for the app and the report."""

from PIL import Image, ImageDraw, ImageFont

from objdetect.models.base import Detection

# A fixed palette so each run colours classes consistently.
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
]


def _colour_for(label: str) -> str:
    return _PALETTE[hash(label) % len(_PALETTE)]


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    show_score: bool = True,
) -> Image.Image:
    """Return a copy of ``image`` with each detection's box and label drawn."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # older Pillow: load_default takes no size argument
        font = ImageFont.load_default()

    for det in detections:
        colour = _colour_for(det.label)
        x1, y1, x2, y2 = det.box
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=3)

        caption = det.label
        if show_score:
            caption += f" {det.score:.2f}"
        text_box = draw.textbbox((x1, y1), caption, font=font)
        draw.rectangle(text_box, fill=colour)
        draw.text((x1, y1), caption, fill="white", font=font)

    return canvas
