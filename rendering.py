from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import NEIGHBORS
from smoothing import dilate_mask


def render_outline_and_colored(
    final_regions: List[Dict],
    palette_final: List[Tuple[int, int, int]],
    color_id_to_paint_index: Dict[int, int],
    line_thickness_px: int,
    size: Tuple[int, int],
) -> Tuple[Image.Image, Image.Image]:
    """
    Render:
      - outline image: white background, black contours, numbers
      - colored image: filled with colors + same contours

    Parameters
    ----------
    final_regions : list of dict
        Regions from final segmentation.
    palette_final : list of (R, G, B)
        Palette corresponding to color_id in final_regions.
    color_id_to_paint_index : dict
        Mapping from color_id to paint index.
    line_thickness_px : int
        Contour thickness in pixels.
    size : (H, W)
        Output size in pixels.

    Returns
    -------
    outline_img : PIL.Image
    colored_img : PIL.Image
    """
    H, W = size
    outline_arr = np.full((H, W, 3), 255, dtype=np.uint8)
    colored_arr = np.full((H, W, 3), 255, dtype=np.uint8)

    # Fill with color
    for reg in final_regions:
        cid = reg["color_id"]
        if cid not in color_id_to_paint_index:
            continue
        color = palette_final[cid]
        for (yy, xx) in reg["pixels"]:
            colored_arr[yy, xx] = color

    # Build contours
    for reg in final_regions:
        min_y, max_y, min_x, max_x = reg["bbox"]
        H_loc = max_y - min_y + 1
        W_loc = max_x - min_x + 1

        local_mask = np.zeros((H_loc, W_loc), dtype=bool)
        for (yy, xx) in reg["pixels"]:
            local_mask[yy - min_y, xx - min_x] = True

        border_local = np.zeros_like(local_mask)
        for dy, dx in NEIGHBORS:
            ys = slice(max(0, -dy), min(H_loc, H_loc - dy))
            xs = slice(max(0, -dx), min(W_loc, W_loc - dx))
            ys2 = slice(
                max(0, -dy) + dy,
                min(H_loc, H_loc - dy) + dy,
            )
            xs2 = slice(
                max(0, -dx) + dx,
                min(W_loc, W_loc - dx) + dx,
            )

            m1 = local_mask[ys, xs]
            m2 = local_mask[ys2, xs2]
            border_local[ys, xs] |= (m1 & ~m2)

        border_thick = dilate_mask(border_local, line_thickness_px)
        ys_idx, xs_idx = np.where(border_thick)
        outline_arr[min_y + ys_idx, min_x + xs_idx] = (0, 0, 0)
        colored_arr[min_y + ys_idx, min_x + xs_idx] = (0, 0, 0)

    outline_img = Image.fromarray(outline_arr, mode="RGB")
    colored_img = Image.fromarray(colored_arr, mode="RGB")
    return outline_img, colored_img


def draw_numbers_on_outline(
    outline_img: Image.Image,
    final_regions: List[Dict],
    color_id_to_paint_index: Dict[int, int],
    font_size: int = 28,
) -> None:
    """
    Draw color numbers inside regions, trying to keep text within the area.

    Parameters
    ----------
    font_size : int
        Base font size in pixels (28 is about +30% compared to previous 22).
    """
    W, H = outline_img.size
    draw = ImageDraw.Draw(outline_img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()


    for reg in final_regions:
        cid = reg["color_id"]
        if cid not in color_id_to_paint_index:
            continue

        number = color_id_to_paint_index[cid]
        text = str(number)

        # Measure text size
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        cx = reg["cx"]
        cy = reg["cy"]

        min_y, max_y, min_x, max_x = reg["bbox"]
        H_loc = max_y - min_y + 1
        W_loc = max_x - min_x + 1

        # Local mask
        local_mask = np.zeros((H_loc, W_loc), dtype=bool)
        for (yy, xx) in reg["pixels"]:
            local_mask[yy - min_y, xx - min_x] = True

        # Desired centre in local coordinates
        cx_loc = cx - min_x
        cy_loc = cy - min_y

        best_pos = None

        # Search radius for candidate positions around centroid
        max_radius = max(10, min(H_loc, W_loc) // 3)

        for radius in range(max_radius + 1):
            found = False
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    cxx = cx_loc + dx
                    cyy = cy_loc + dy

                    tx_loc = cxx - tw // 2
                    ty_loc = cyy - th // 2

                    if tx_loc < 0 or ty_loc < 0:
                        continue
                    if tx_loc + tw > W_loc or ty_loc + th > H_loc:
                        continue

                    submask = local_mask[ty_loc : ty_loc + th, tx_loc : tx_loc + tw]
                    if submask.all():
                        tx = min_x + tx_loc
                        ty = min_y + ty_loc
                        best_pos = (tx, ty)
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if best_pos is not None:
            tx, ty = best_pos
        else:
            # Fallback: just centre within the image and clamp.
            tx = cx - tw // 2
            ty = cy - th // 2
            tx = max(0, min(W - tw, tx))
            ty = max(0, min(H - th, ty))

        draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
