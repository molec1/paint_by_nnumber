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
    font_size: int,
    min_feature_px: float,
) -> None:
    """
    Draw exactly one color number per region (if region is large enough).

    - Uses dynamic font_size (in px).
    - Places the number inside region using the region mask.
    - Has hard limits on search radius and step to avoid huge runtimes.
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

        # размер текста
        bbox_text = draw.textbbox((0, 0), text, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        # если область меньше текста — пропускаем
        if reg["area"] < tw * th * 1.2:
            continue

        cx = reg["cx"]
        cy = reg["cy"]

        min_y, max_y, min_x, max_x = reg["bbox"]
        H_loc = max_y - min_y + 1
        W_loc = max_x - min_x + 1

        # локальная маска области
        local_mask = np.zeros((H_loc, W_loc), dtype=bool)
        for (yy, xx) in reg["pixels"]:
            local_mask[yy - min_y, xx - min_x] = True

        # целевой центр в локальных координатах
        cx_loc = cx - min_x
        cy_loc = cy - min_y

        best_pos = None

        # отступ от границы области внутрь
        inner_margin = max(1, font_size // 6)

        # ограничения на поиск
        # не больше 20% диагонали области, но и не больше 80 пикселей
        max_dim = max(H_loc, W_loc)
        max_radius = int(min(max_dim * 0.2, 80))

        # шаг по сетке
        step = max(1, font_size // 3)

        for radius in range(0, max_radius + 1, step):
            found = False
            # грубая квадратная окрестность
            for dy in range(-radius, radius + 1, step):
                for dx in range(-radius, radius + 1, step):
                    cxx = cx_loc + dx
                    cyy = cy_loc + dy

                    tx_loc = cxx - tw // 2
                    ty_loc = cyy - th // 2

                    # текст должен помещаться в локальный bbox
                    if tx_loc < 0 or ty_loc < 0:
                        continue
                    if tx_loc + tw > W_loc or ty_loc + th > H_loc:
                        continue

                    # внутренний прямоугольник с отступами
                    ix0 = tx_loc + inner_margin
                    iy0 = ty_loc + inner_margin
                    ix1 = tx_loc + tw - inner_margin
                    iy1 = ty_loc + th - inner_margin

                    if ix1 <= ix0 or iy1 <= iy0:
                        ix0, iy0 = tx_loc, ty_loc
                        ix1, iy1 = tx_loc + tw, ty_loc + th

                    if ix0 < 0 or iy0 < 0 or ix1 > W_loc or iy1 > H_loc:
                        continue

                    submask = local_mask[iy0:iy1, ix0:ix1]
                    if submask.size == 0:
                        continue

                    # требуем, чтобы большая часть текста была внутри области
                    if submask.mean() < 0.97:
                        continue

                    # переводим в глобальные координаты
                    tx = min_x + tx_loc
                    ty = min_y + ty_loc
                    best_pos = (tx, ty)
                    found = True
                    break
                if found:
                    break
            if found:
                break

        if best_pos is None:
            # fallback: просто центрируем, но зажимаем внутрь картинки
            tx = cx - tw // 2
            ty = cy - th // 2
            tx = max(0, min(W - tw, tx))
            ty = max(0, min(H - th, ty))
        else:
            tx, ty = best_pos

        draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
