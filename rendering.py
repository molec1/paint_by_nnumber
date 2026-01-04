from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import LINE_THICKNESS_MM

from config import NEIGHBORS
from smoothing import dilate_mask

def render_outline_and_colored_highres(
    cluster_id_img: np.ndarray,  # (H, W) int color_id per pixel
    palette_final: List[Tuple[int, int, int]],
    color_id_to_paint_index: Dict[int, int],
    target_long_px: int,
) -> Tuple[Image.Image, Image.Image, np.ndarray, float]:
    """
    High-res render:
      - upscales cluster_id_img using NEAREST to match target_long_px (long side)
      - builds colored image by palette lookup
      - builds contours by label differences + dilation
    Returns:
      outline_img, colored_img, labels_big, scale
    """
    H, W = cluster_id_img.shape
    image_long_px = max(H, W)
    if target_long_px <= 0:
        raise ValueError("target_long_px must be > 0")

    scale = float(target_long_px) / float(image_long_px)
    H2 = int(round(H * scale))
    W2 = int(round(W * scale))

    # 1) Upscale labels (NEAREST is crucial)
    labels_pil = Image.fromarray(cluster_id_img.astype(np.int32), mode="I")
    labels_big_pil = labels_pil.resize((W2, H2), resample=Image.NEAREST)
    labels_big = np.array(labels_big_pil, dtype=np.int32)

    # 2) Colored fill via palette lookup
    pal = np.asarray(palette_final, dtype=np.uint8)  # (K, 3)
    # safety: clamp ids to palette range (shouldn't be needed, but robust)
    labels_safe = np.clip(labels_big, 0, len(pal) - 1)
    colored_arr = pal[labels_safe]  # (H2, W2, 3)

    # 3) Outline base (white)
    outline_arr = np.full((H2, W2, 3), 255, dtype=np.uint8)

    # 4) Contours from label differences
    border = np.zeros((H2, W2), dtype=bool)
    for dy, dx in NEIGHBORS:
        ys = slice(max(0, -dy), min(H2, H2 - dy))
        xs = slice(max(0, -dx), min(W2, W2 - dx))
        ys2 = slice(max(0, -dy) + dy, min(H2, H2 - dy) + dy)
        xs2 = slice(max(0, -dx) + dx, min(W2, W2 - dx) + dx)

        m1 = labels_big[ys, xs]
        m2 = labels_big[ys2, xs2]
        border[ys, xs] |= (m1 != m2)

    # Scale thickness to hi-res
    thickness_hi = max(1, int(round(300 * (LINE_THICKNESS_MM / 25.4))))

    border_thick = dilate_mask(border, thickness_hi)

    outline_arr[border_thick] = (0, 0, 0)
    colored_arr[border_thick] = (0, 0, 0)

    outline_img = Image.fromarray(outline_arr, mode="RGB")
    colored_img = Image.fromarray(colored_arr, mode="RGB")
    return outline_img, colored_img, labels_big, scale

def _load_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            return ImageFont.load_default()



def draw_numbers_on_outline_highres(
    outline_img: Image.Image,
    final_regions: List[Dict],
    labels_big: np.ndarray,  # (H2, W2) color_id map at hi-res
    color_id_to_paint_index: Dict[int, int],
    font_size: int,
    scale: float,
    max_repeats_per_region: int = 4,
    min_dist_factor: float = 7.0,
    area_per_label_factor: float = 30.0,
) -> None:
    """
    Draw up to `max_repeats_per_region` numbers per region, depending on region area.
    Enforces minimum distance between labels ~ min_dist_factor * font_size.
    """
    W2, H2 = outline_img.size
    draw = ImageDraw.Draw(outline_img)
    font = _load_font(font_size)

    for reg in final_regions:
        cid = reg["color_id"]
        if cid not in color_id_to_paint_index:
            continue

        number = color_id_to_paint_index[cid]
        text = str(number)

        bbox_text = draw.textbbox((0, 0), text, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        area_hi = float(reg["area"]) * (scale * scale)
        if area_hi < tw * th * 1.2:
            continue

        target = int(area_hi // (area_per_label_factor * float(tw * th)))
        target = max(1, min(max_repeats_per_region, target))

        cx = int(round(reg["cx"] * scale))
        cy = int(round(reg["cy"] * scale))

        min_y, max_y, min_x, max_x = reg["bbox"]
        min_y2 = max(0, int(np.floor(min_y * scale)))
        max_y2 = min(H2 - 1, int(np.ceil(max_y * scale)))
        min_x2 = max(0, int(np.floor(min_x * scale)))
        max_x2 = min(W2 - 1, int(np.ceil(max_x * scale)))

        local = labels_big[min_y2 : max_y2 + 1, min_x2 : max_x2 + 1]
        local_mask = (local == cid)
        H_loc, W_loc = local_mask.shape
        if H_loc <= 0 or W_loc <= 0:
            continue

        if local_mask.sum() < (tw * th * 2):
            target = 1

        inner_margin = max(1, font_size // 6)
        max_dim = max(H_loc, W_loc)
        max_radius = int(min(max_dim * 0.45, 220))
        step = max(1, font_size // 3)

        min_dist = float(min_dist_factor * font_size)
        min_dist2 = min_dist * min_dist

        ring1 = int(round(1.6 * min_dist))
        ring2 = int(round(2.4 * min_dist))

        anchors_1 = [(0, 0)]
        anchors_2 = [(ring1, 0), (-ring1, 0)]
        anchors_3 = [(ring1, 0), (-ring1, 0), (0, ring1)]
        anchors_4 = [(-ring1, -ring1), (ring1, -ring1), (-ring1, ring1), (ring1, ring1)]

        anchors_fallback = [
            (0, ring1),
            (0, -ring1),
            (ring2, 0),
            (-ring2, 0),
            (ring2, ring2),
            (ring2, -ring2),
            (-ring2, ring2),
            (-ring2, -ring2),
        ]

        if target == 1:
            anchor_offsets = anchors_1
        elif target == 2:
            anchor_offsets = anchors_2
        elif target == 3:
            anchor_offsets = anchors_3
        else:
            anchor_offsets = anchors_4

        anchor_offsets = anchor_offsets + anchors_fallback

        placed_centers: List[Tuple[int, int]] = []

        def fits_here(tx_loc: int, ty_loc: int) -> bool:
            if tx_loc < 0 or ty_loc < 0:
                return False
            if tx_loc + tw > W_loc or ty_loc + th > H_loc:
                return False

            ix0 = tx_loc + inner_margin
            iy0 = ty_loc + inner_margin
            ix1 = tx_loc + tw - inner_margin
            iy1 = ty_loc + th - inner_margin
            if ix1 <= ix0 or iy1 <= iy0:
                ix0, iy0 = tx_loc, ty_loc
                ix1, iy1 = tx_loc + tw, ty_loc + th

            submask = local_mask[iy0:iy1, ix0:ix1]
            if submask.size == 0:
                return False
            if submask.mean() < 0.97:
                return False

            cx_t = tx_loc + tw // 2
            cy_t = ty_loc + th // 2
            for (pcx, pcy) in placed_centers:
                dx = float(cx_t - pcx)
                dy = float(cy_t - pcy)
                if dx * dx + dy * dy < min_dist2:
                    return False

            return True

        for k in range(target):
            best_pos = None

            ax, ay = anchor_offsets[k] if k < len(anchor_offsets) else anchor_offsets[-1]
            anchor_x = int(round((cx - min_x2) + ax))
            anchor_y = int(round((cy - min_y2) + ay))

            for radius in range(0, max_radius + 1, step):
                found = False
                for dy in range(-radius, radius + 1, step):
                    for dx in range(-radius, radius + 1, step):
                        cxx = anchor_x + dx
                        cyy = anchor_y + dy

                        tx_loc = cxx - tw // 2
                        ty_loc = cyy - th // 2

                        if fits_here(tx_loc, ty_loc):
                            best_pos = (tx_loc, ty_loc)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

            if best_pos is None:
                break

            tx_loc, ty_loc = best_pos
            tx = min_x2 + tx_loc
            ty = min_y2 + ty_loc

            draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
            placed_centers.append((tx_loc + tw // 2, ty_loc + th // 2))