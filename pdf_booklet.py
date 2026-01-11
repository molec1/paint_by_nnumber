from __future__ import annotations

import csv
import io
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import webcolors
from PIL import Image, ImageOps
from reportlab.lib.pagesizes import A1, A2, A3, A4, A5, landscape, portrait
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def closest_css3_name(rgb_tuple: Tuple[int, int, int]) -> str:
    """
    Find the nearest CSS3 color name in RGB distance and prettify the name.
    """
    r, g, b = rgb_tuple
    min_dist = None
    best = None
    for name, hex_code in webcolors.CSS3_NAMES_TO_HEX.items():
        cr, cg, cb = webcolors.hex_to_rgb(hex_code)
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if min_dist is None or dist < min_dist:
            min_dist = dist
            best = name
    return best.replace("-", " ").title()


@lru_cache(maxsize=256)
def closest_css3_name_cached(r: int, g: int, b: int) -> str:
    """
    Cached variant of closest_css3_name for repeated palette lookups.
    """
    return closest_css3_name((r, g, b))


def read_palette_with_names(palette_csv_path: str):
    """
    Read palette CSV:

        index, R, G, B, hex

    Returns
    -------
    list of (index, (R, G, B), hex, human_name), sorted by index.
    """
    colors = []
    with open(palette_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            idx, R, G, B, hex_code = row
            idx = int(idx)
            rgb = (int(R), int(G), int(B))
            name = closest_css3_name_cached(*rgb)
            colors.append((idx, rgb, hex_code, name))

    colors.sort(key=lambda t: t[0])
    return colors


def load_image_reader(path: str) -> ImageReader:
    """
    Create an ImageReader with EXIF orientation applied if needed.

    Fast path:
        - If EXIF orientation is normal (1) or missing, let ReportLab read
          JPEG/PNG directly from disk (keeps original compression).
    Slow path:
        - Apply EXIF orientation via PIL and re-encode to JPEG in-memory,
          so ReportLab embeds a compressed JPEG stream instead of raw pixels.
    """
    path = str(Path(path).resolve())
    try:
        with Image.open(path) as im_probe:
            exif = im_probe.getexif()
            # 274 = Orientation tag
            orientation = int(exif.get(274, 1)) if exif is not None else 1
    except Exception:
        orientation = 1

    if orientation == 1:
        # Keep original compression and let ReportLab read directly
        return ImageReader(path)

    # For rotated images: transpose and keep JPEG compression via BytesIO
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92, optimize=True)
        buf.seek(0)

    return ImageReader(buf)


def load_image_reader_scaled(path: str, max_long_px: int = 1800) -> ImageReader:
    """
    Load an image, apply EXIF orientation, downscale so that the long side
    is at most max_long_px, and return an ImageReader backed by JPEG bytes.

    This is used for the second page of the booklet, where full-resolution
    images are not required and would consume extra memory.
    """
    path = str(Path(path).resolve())
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")

        w, h = im.size
        long_side = max(w, h)
        if long_side > max_long_px:
            scale = max_long_px / float(long_side)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        # 88 quality is enough for preview while keeping size and memory lower
        im.save(buf, format="JPEG", quality=88, optimize=True)
        buf.seek(0)

    return ImageReader(buf)


def draw_centered_image(
    c: canvas.Canvas,
    img: ImageReader,
    box_w: float,
    box_h: float,
    x: float,
    y: float,
) -> None:
    """
    Draw an image inside (x, y, box_w, box_h) keeping aspect ratio.
    """
    iw, ih = img.getSize()

    scale = min(box_w / iw, box_h / ih)
    sw = iw * scale
    sh = ih * scale

    px = x + (box_w - sw) / 2
    py = y + (box_h - sh) / 2

    c.drawImage(img, px, py, sw, sh)


def draw_big_palette(
    c: canvas.Canvas,
    colors,
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
) -> None:
    """
    Draw a palette inside (box_x, box_y, box_w, box_h).

    Layout:
        - left-to-right, top-to-bottom, indices 1..N
        - color swatch
        - first line: "N: #HEX"
        - second line: human-readable name (shortened if needed)
    """
    if not colors:
        return

    swatch_h = 14 * mm
    swatch_w = 32 * mm
    line_h = 4.5 * mm
    gap_between_text = 0.7 * mm
    inner_pad = 1.0 * mm
    tile_h = swatch_h + 2 * line_h + gap_between_text + inner_pad
    gap_x = 6 * mm
    gap_y = 3 * mm

    cols = max(1, int((box_w + gap_x) // (swatch_w + gap_x)))
    rows = (len(colors) + cols - 1) // cols

    total_height = rows * tile_h + (rows - 1) * gap_y

    if total_height > box_h:
        scale = box_h / total_height
        swatch_h *= scale
        swatch_w *= scale
        line_h *= scale
        gap_between_text *= scale
        inner_pad *= scale
        tile_h = swatch_h + 2 * line_h + gap_between_text + inner_pad
        gap_y *= scale

    font_size = max(6, int(7 * (swatch_h / (14 * mm))))
    c.setFont("Helvetica", font_size)

    total_height = rows * tile_h + (rows - 1) * gap_y
    y_top = box_y + box_h - tile_h

    for i, (idx, rgb, hex_code, name) in enumerate(colors):
        row = i // cols
        col = i % cols

        y_base = y_top - row * (tile_h + gap_y)
        x = box_x + col * (swatch_w + gap_x)

        r, g, b = rgb
        c.setStrokeColorRGB(0, 0, 0)
        c.setFillColorRGB(r / 255.0, g / 255.0, b / 255.0)

        # Color swatch
        y_color = y_base + 2 * line_h + gap_between_text + inner_pad
        c.rect(x, y_color, swatch_w, swatch_h, stroke=1, fill=1)

        c.setFillColorRGB(0, 0, 0)

        # First line: "index: #HEX"
        text1 = f"{idx}: {hex_code}"
        c.drawString(
            x + inner_pad,
            y_base + line_h + gap_between_text,
            text1,
        )

        # Second line: human-readable name, shortened to avoid overflow
        name_short = name[:36]
        c.drawString(
            x + inner_pad,
            y_base + inner_pad,
            name_short,
        )


def build_pbn_pdf_booklet(
    root: str,
    original_path: str,
    outline_path: str,
    palette_csv_path: str,
    pdf_name: str | None = None,
    paper_size: str = "A3",
    num_regions: int | None = None,
    difficulty: str | None = None,
    original_preview_path: str | None = None,
    outline_preview_path: str | None = None,
) -> None:
    """
    Build a 2-page PDF booklet:

        Page 1 (paper_size, e.g. A2/A3):
            - full-size outline only (high resolution for printing)

        Page 2 (A4):
            - top: downscaled original + downscaled outline side by side
            - bottom: palette with color tiles and names

    If num_regions and difficulty are provided, a short complexity summary
    is displayed between the preview images and the palette.
    """
    if pdf_name is None:
        pdf_name = f"output/{root}_booklet.pdf"

    # ---------- Page 1: outline only (full-res) ----------
    outline_img_full = load_image_reader(outline_path)

    size_dict = {
        "A1": A1,
        "A2": A2,
        "A3": A3,
        "A4": A4,
        "A5": A5,
    }

    ow_full, oh_full = outline_img_full.getSize()
    is_landscape = ow_full >= oh_full
    if is_landscape:
        page1_size = landscape(size_dict[paper_size])
    else:
        page1_size = portrait(size_dict[paper_size])

    c = canvas.Canvas(pdf_name, pagesize=page1_size)
    W1, H1 = page1_size

    margin = 5 * mm
    draw_centered_image(
        c,
        outline_img_full,
        W1 - 2 * margin,
        H1 - 2 * margin,
        margin,
        margin,
    )

    c.showPage()

    # Release reference to high-res outline before building the second page
    del outline_img_full

        # ---------- Page 2: A4, downscaled images ----------
    # Prefer small JPEG previews if they exist, otherwise fall back
    # to the original high-resolution files.
    root_path = Path(root)

    orig_source = root_path.with_name(root_path.name + "_preview_original.jpg")
    outline_source = root_path.with_name(root_path.name + "_preview_outline.jpg")

    if not orig_source.is_file():
        orig_source = Path(original_path)

    if not outline_source.is_file():
        outline_source = Path(outline_path)

    orig_img_small = load_image_reader_scaled(str(orig_source), max_long_px=1800)
    outline_img_small = load_image_reader_scaled(str(outline_source), max_long_px=1800)



    if is_landscape:
        page2_size = landscape(A4)
    else:
        page2_size = portrait(A4)

    c.setPageSize(page2_size)
    W2, H2 = page2_size

    margin_x = 8 * mm
    margin_y_top = 8 * mm
    margin_y_bottom = 8 * mm
    mid_gap = 6 * mm

    usable_h = H2 - margin_y_top - margin_y_bottom - mid_gap
    top_h = usable_h * 0.5
    bottom_h = usable_h - top_h

    # Top region: original + mini-outline side by side
    top_box_y = H2 - margin_y_top - top_h
    top_box_h = top_h
    half_w = (W2 - 3 * margin_x) / 2.0

    draw_centered_image(
        c,
        orig_img_small,
        half_w,
        top_box_h,
        margin_x,
        top_box_y,
    )

    draw_centered_image(
        c,
        outline_img_small,
        half_w,
        top_box_h,
        margin_x * 2.0 + half_w,
        top_box_y,
    )

    # Bottom region: palette
    pal_box_y = margin_y_bottom
    pal_box_h = bottom_h
    pal_box_x = margin_x
    pal_box_w = W2 - 2 * margin_x

    colors = read_palette_with_names(palette_csv_path)
    draw_big_palette(
        c,
        colors,
        pal_box_x,
        pal_box_y,
        pal_box_w,
        pal_box_h,
    )

    # Optional complexity summary in the middle gap between preview and palette
    if num_regions is not None and difficulty is not None:
        c.setFont("Helvetica", 9)
        footer_text = f"Regions: {num_regions}  •  Difficulty: {difficulty}"
        gap_center_y = pal_box_y + pal_box_h + (mid_gap - 3 * mm) * 0.5
        c.drawCentredString(W2 / 2.0, gap_center_y, footer_text)

    # Drop small-page images and palette data before finalizing the canvas
    del colors, orig_img_small, outline_img_small

    c.showPage()
    c.save()
    print(f"[10] PDF booklet saved as {pdf_name}")
