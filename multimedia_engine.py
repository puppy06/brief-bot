"""
Multimedia: lower-thirds (Pillow) and legacy placeholder helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from news_processor import NewsBriefPayload


# --- Full-frame slides (dashboard / broadcast cards) ---


def _gradient_background(
    width: int,
    height: int,
    top: tuple[int, int, int] = (99, 102, 241),
    bottom: tuple[int, int, int] = (168, 85, 247),
) -> Image.Image:
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    if pixels is None:
        return img
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        for x in range(width):
            pixels[x, y] = (r, g, b)
    return img


def _wrap_lines(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.replace("\n", " ").split()
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        test = " ".join(current + [w])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def render_slide_card(
    output_path: str | Path,
    *,
    title: str,
    body: str,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """
    Full-frame slide with a soft indigo–violet gradient and a frosted card panel,
    suitable for news segments or a TurboLearn-style visual.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    img = _gradient_background(width, height)
    draw = ImageDraw.Draw(img, "RGBA")

    card_margin_x = int(width * 0.09)
    card_margin_y = int(height * 0.12)
    card_w = width - 2 * card_margin_x
    card_h = height - 2 * card_margin_y
    radius = 28
    # Frosted panel
    panel = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 38))
    img.paste(panel, (card_margin_x, card_margin_y), panel)
    # Rounded rect outline (simplified: inner stroke)
    draw.rounded_rectangle(
        (card_margin_x, card_margin_y, card_margin_x + card_w, card_margin_y + card_h),
        radius=radius,
        outline=(255, 255, 255, 90),
        width=2,
    )

    try:
        font_title = ImageFont.truetype("arial.ttf", 52)
        font_body = ImageFont.truetype("arial.ttf", 32)
        font_meta = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()
        font_meta = ImageFont.load_default()

    pad = int(width * 0.06)
    x0 = card_margin_x + pad
    y = card_margin_y + int(card_h * 0.08)
    max_w = card_w - 2 * pad

    draw.text((x0, y), title.upper(), fill=(255, 255, 255, 255), font=font_title)
    bbox = draw.textbbox((x0, y), title.upper(), font=font_title)
    y = bbox[3] + 28

    draw.text((x0, y), "News briefing", fill=(226, 232, 240, 255), font=font_meta)
    y += 40

    lines = _wrap_lines(body, draw, font_body, max_w)
    line_h = int(draw.textbbox((0, 0), "Ag", font=font_body)[3] + 8)
    for line in lines[:18]:
        draw.text((x0, y), line, fill=(248, 250, 252, 255), font=font_body)
        y += line_h

    img.save(out, format="PNG")
    return out


def render_slides_for_brief(
    brief: NewsBriefPayload,
    out_dir: str | Path,
    *,
    prefix: str = "slide",
) -> list[Path]:
    """One PNG slide per script segment (matches ``brief['segments']`` order)."""
    d = Path(out_dir)
    paths: list[Path] = []
    for i, seg in enumerate(brief["segments"]):
        title = f"Segment {i + 1}"
        if i == 0:
            title = "Opening"
        elif i == len(brief["segments"]) - 1:
            title = "Closing"
        else:
            title = f"Story {i}"
        paths.append(
            render_slide_card(
                d / f"{prefix}_{i:02d}.png",
                title=title,
                body=seg["script_text"],
            )
        )
    return paths


# --- Lower thirds (Pillow) ---

def render_lower_third(
    output_path: str | Path,
    *,
    headline: str,
    subline: str | None = None,
    width: int = 1920,
    height: int = 1080,
    bar_height_ratio: float = 0.14,
    accent_color: tuple[int, int, int] = (200, 32, 32),
    bg_alpha: int = 220,
) -> Path:
    """
    Generate a news-style lower third PNG (with alpha) suitable for FFmpeg overlay.

    Tradeoff: raster PNGs are simple and portable; for animated lower thirds you
    would swap in ASS subtitles or a motion template — not needed for the initial scope.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bar_h = int(height * bar_height_ratio)
    y0 = height - bar_h - int(height * 0.06)
    # Semi-transparent bar
    draw.rectangle((0, y0, width, height), fill=(10, 14, 22, bg_alpha))
    draw.rectangle((0, y0, int(width * 0.012), height), fill=accent_color + (255,))

    title_size = max(28, int(bar_h * 0.28))
    sub_size = max(20, int(bar_h * 0.20))
    try:
        font_title = ImageFont.truetype("arial.ttf", title_size)
        font_sub = ImageFont.truetype("arial.ttf", sub_size)
    except OSError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    pad_x = int(width * 0.03)
    text_y = y0 + int(bar_h * 0.18)
    draw.text((pad_x, text_y), headline, fill=(255, 255, 255, 255), font=font_title)
    if subline:
        draw.text(
            (pad_x, text_y + title_size + 6),
            subline,
            fill=(220, 224, 230, 255),
            font=font_sub,
        )

    img.save(out, format="PNG")
    return out


def render_key_point_lower_thirds(
    key_developments: list[str],
    out_dir: str | Path,
    *,
    prefix: str = "lower_third_pt",
) -> list[Path]:
    """One lower third per key development (headline = point title, subline optional)."""
    d = Path(out_dir)
    paths: list[Path] = []
    for i, text in enumerate(key_developments, start=1):
        # First line: label; second: truncated body for readability on screen
        headline = f"Key development {i}"
        sub = text if len(text) < 120 else text[:117] + "..."
        paths.append(render_lower_third(d / f"{prefix}_{i}.png", headline=headline, subline=sub))
    return paths


# --- Placeholder: LTX-2.3 ---

def generate_broll_ltx_2_3(
    visual_description: str,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """
    Placeholder for LTX-2.3 video generation.

    Wire this to the vendor CLI/SDK when credentials and a local or remote
    runtime are available. Return a video file path compatible with FFmpeg.
    """
    raise NotImplementedError(
        "LTX-2.3 integration not implemented; connect API and write MP4 to output_path. "
        f"Prompt seed: {visual_description!r} kwargs={kwargs!r}"
    )
