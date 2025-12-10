from __future__ import annotations

import csv
from typing import Tuple

import webcolors
from reportlab.lib.pagesizes import A3, A4, landscape, portrait
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
    # "light-sky-blue" -> "Light Sky Blue"
    return best.replace("-", " ").title()


def read_palette_with_names(palette_csv_path: str):
    """
    Read palette CSV:
      index, R, G, B, hex

    Returns
    -------
    list of (index, (R,G,B), hex, human_name), sorted by index.
    """
    colors = []
    with open(palette_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            idx, R, G, B, hex_code = row
            idx = int(idx)
            rgb = (int(R), int(G), int(B))
            name = closest_css3_name(rgb)
            colors.append((idx, rgb, hex_code, name))

    colors.sort(key=lambda t: t[0])
    return colors


def draw_centered_image(
    c: canvas.Canvas,
    path: str,
    box_w: float,
    box_h: float,
    x: float,
    y: float,
) -> None:
    """
    Draw an image inside (x, y, box_w, box_h) keeping aspect ratio.
    """
    img = ImageReader(path)
    iw, ih = img.getSize()

    scale = min(box_w / iw, box_h / ih)
    sw = iw * scale
    sh = ih * scale

    px = x + (box_w - sw) / 2
    py = y + (box_h - sh) / 2

    c.drawImage(img, px, py, sw, sh)


def draw_big_palette(
    c: canvas.Canvas,
    colors,   # list of (idx, (R,G,B), hex_code, name)
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
) -> None:
    """
    Draw palette inside (box_x, box_y, box_w, box_h):

      - left-to-right, top-to-bottom, indices 1..N
      - color swatch
      - first line: "N: #HEX"
      - second line: human-readable name (wrapped if needed)
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

        y_color = y_base + 2 * line_h + gap_between_text + inner_pad
        c.rect(x, y_color, swatch_w, swatch_h, stroke=1, fill=1)

        c.setFillColorRGB(0, 0, 0)

        text1 = f"{idx}: {hex_code}"
        c.drawString(
            x + inner_pad,
            y_base + line_h + gap_between_text,
            text1,
        )

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
) -> None:
    """
    Build a 2-page PDF:

      1) A3: only the outline (full-size coloring page).
      2) A4 (orientation follows original):
         - top: original + mini-outline side by side
         - bottom: palette tiles with names.
    """
    if pdf_name is None:
        pdf_name = f"output/{root}_booklet.pdf"

    # ------- Page 1: A3 outline only -------
    outline_img = ImageReader(outline_path)
    ow, oh = outline_img.getSize()

    if ow >= oh:
        page1_size = landscape(A3)
    else:
        page1_size = portrait(A3)

    c = canvas.Canvas(pdf_name, pagesize=page1_size)
    W1, H1 = page1_size

    margin = 5 * mm
    draw_centered_image(
        c,
        outline_path,
        W1 - 2 * margin,
        H1 - 2 * margin,
        margin,
        margin,
    )

    c.showPage()

    # ------- Page 2: A4, original orientation -------
    orig_img_reader = ImageReader(original_path)
    ow0, oh0 = orig_img_reader.getSize()
    if ow0 >= oh0:
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

    # Top: original + mini-outline side by side
    top_box_y = H2 - margin_y_top - top_h
    top_box_h = top_h
    half_w = (W2 - 3 * margin_x) / 2

    draw_centered_image(
        c,
        original_path,
        half_w,
        top_box_h,
        margin_x,
        top_box_y,
    )

    draw_centered_image(
        c,
        outline_path,
        half_w,
        top_box_h,
        margin_x * 2 + half_w,
        top_box_y,
    )

    # Bottom: palette
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

    c.showPage()
    c.save()
    print(f"[10] PDF booklet saved as {pdf_name}")
