from __future__ import annotations

import csv
import colorsys
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def build_ordered_palette(
    palette_final: List[Tuple[int, int, int]],
) -> Tuple[Dict[int, int], List[Tuple[int, Tuple[int, int, int]]]]:
    """
    Order colors for painting palette:

    - chromatic colors (by hue, then lightness)
    - then near-grays (by lightness)

    Returns
    -------
    color_id_to_paint_index : dict
        Mapping from color_id to paint index 1..N.
    paint_palette : list of (index, (R, G, B))
        Palette entries in painting order.
    """

    def rgb_to_hls_norm(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        r, g, b = rgb
        return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)  # H, L, S

    color_infos = []
    for cid, rgb in enumerate(palette_final):
        h, l, s = rgb_to_hls_norm(rgb)
        color_infos.append({"cid": cid, "rgb": rgb, "h": h, "l": l, "s": s})

    SAT_THRESHOLD = 0.15
    chroma = [c for c in color_infos if c["s"] >= SAT_THRESHOLD]
    grays = [c for c in color_infos if c["s"] < SAT_THRESHOLD]

    chroma.sort(key=lambda c: (c["h"], c["l"]))
    grays.sort(key=lambda c: c["l"])

    ordered = chroma + grays

    color_id_to_paint_index: Dict[int, int] = {}
    paint_palette: List[Tuple[int, Tuple[int, int, int]]] = []

    for idx, info in enumerate(ordered, start=1):
        cid = info["cid"]
        rgb = info["rgb"]
        color_id_to_paint_index[cid] = idx
        paint_palette.append((idx, rgb))

    return color_id_to_paint_index, paint_palette


def save_palette_csv(
    palette_csv_path: str,
    paint_palette: List[Tuple[int, Tuple[int, int, int]]],
) -> None:
    """
    Save palette to CSV with columns: index, R, G, B, hex.
    """
    with open(palette_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "R", "G", "B", "hex"])
        for idx, (r, g, b) in paint_palette:
            hex_code = f"#{r:02X}{g:02X}{b:02X}"
            writer.writerow([idx, r, g, b, hex_code])


def save_palette_image(
    palette_img_path: str,
    paint_palette: List[Tuple[int, Tuple[int, int, int]]],
) -> None:
    """
    Render palette as a simple PNG image:
      - color swatch
      - "index: hex" text
    """
    num_colors_paint = len(paint_palette)
    row_height = 40
    margin = 10
    swatch_width = 60
    swatch_height = 30
    img_width = 500
    img_height = num_colors_paint * row_height + 2 * margin

    palette_img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(palette_img)

    try:
        font_palette = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font_palette = ImageFont.load_default()

    for i, (idx_color, (r, g, b)) in enumerate(paint_palette, start=1):
        y_top = margin + (i - 1) * row_height
        y_bottom = y_top + swatch_height
        x_left = margin
        x_right = margin + swatch_width

        draw.rectangle(
            [x_left, y_top, x_right, y_bottom],
            fill=(r, g, b),
            outline=(0, 0, 0),
        )

        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        text = f"{idx_color}: {hex_code}"
        draw.text(
            (x_right + 10, y_top + 5),
            text,
            fill=(0, 0, 0),
            font=font_palette,
        )

    palette_img.save(palette_img_path)

