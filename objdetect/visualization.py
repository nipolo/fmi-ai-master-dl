"""Draw detections onto images for the app and the report."""

from PIL import Image, ImageDraw, ImageFont

from objdetect.models.base import Detection

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

    short_side = min(canvas.size)
    line_width = max(2, round(short_side / 250))
    font_size = max(12, round(short_side / 35))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()
    pad = max(2, line_width)

    for det in detections:
        colour = _colour_for(det.label)
        x1, y1, x2, y2 = det.box
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=line_width)

        caption = det.label
        if show_score:
            caption += f" {det.score * 100:.0f}%"

        left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = right - left, bottom - top
        tag_w, tag_h = text_w + 2 * pad, text_h + 2 * pad

        tag_x = min(x1, canvas.width - tag_w)
        tag_x = max(0, tag_x)
        tag_y = y1 - tag_h
        if tag_y < 0:
            tag_y = y1

        draw.rectangle((tag_x, tag_y, tag_x + tag_w, tag_y + tag_h), fill=colour)
        draw.text((tag_x + pad - left, tag_y + pad - top), caption, fill="white", font=font)

    return canvas
