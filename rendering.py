from __future__ import annotations
from scipy import ndimage as ndi  # add import at top

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

    # 1) Upscale labels as uint8 (0..255). Your ids are 0..K-1, so this is safe.
    labels_pil = Image.fromarray(cluster_id_img.astype(np.uint8), mode="L")
    labels_big_pil = labels_pil.resize((W2, H2), resample=Image.NEAREST)
    labels_big = np.asarray(labels_big_pil, dtype=np.uint8)

    # 2) Colored fill via palette lookup
    pal = np.asarray(palette_final, dtype=np.uint8)
    labels_safe = np.minimum(labels_big, len(pal) - 1)  # faster than clip for uint
    colored_arr = pal[labels_safe]

    outline_arr = np.full((H2, W2, 3), 255, dtype=np.uint8)

    # 4) Contours from label differences (vectorized 4-neighborhood)
    border = np.zeros((H2, W2), dtype=bool)

    # horizontal differences
    diff_h = labels_big[:, 1:] != labels_big[:, :-1]
    border[:, 1:] |= diff_h
    border[:, :-1] |= diff_h

    # vertical differences
    diff_v = labels_big[1:, :] != labels_big[:-1, :]
    border[1:, :] |= diff_v
    border[:-1, :] |= diff_v

    # thickness (you keep your formula)
    thickness_hi = max(1, int(round(300 * (LINE_THICKNESS_MM / 25.4))))

    # fast dilation on C
    if thickness_hi > 1:
        structure = ndi.generate_binary_structure(2, 1)  # 4-connected
        border_thick = ndi.binary_dilation(border, structure=structure, iterations=thickness_hi)
    else:
        border_thick = border

    outline_arr[border_thick] = (0, 0, 0)
    colored_arr[border_thick] = (0, 0, 0)

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
    area_per_label_factor: float = 60.0,
) -> None:
    """
    Place labels inside regions.

    New approach:
      - For each region, decide how many labels `target` to place.
      - If target == 1: place one label using the general placer.
      - If target > 1: split the region bbox into a grid of sub-windows (m x k),
        then run the same single-label placer inside each sub-window.
      - This yields uniform coverage for elongated regions and robustness for complex shapes.

    This version avoids expensive rectangular erosions.
    Requires SciPy (distance transform).
    """
    from scipy import ndimage as ndi

    W2, H2 = outline_img.size
    draw = ImageDraw.Draw(outline_img)
    font = _load_font(font_size)

    text_metrics: Dict[str, Tuple[int, int]] = {}

    def _get_text_size(text: str) -> Tuple[int, int]:
        if text not in text_metrics:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = int(bbox[2] - bbox[0])
            th = int(bbox[3] - bbox[1])
            text_metrics[text] = (tw, th)
        return text_metrics[text]

    def _fits_rect(local_mask: np.ndarray, tx: int, ty: int, tw: int, th: int, inner_margin: int) -> bool:
        H_loc, W_loc = local_mask.shape
        if tx < 0 or ty < 0:
            return False
        if tx + tw > W_loc or ty + th > H_loc:
            return False

        ix0 = tx + inner_margin
        iy0 = ty + inner_margin
        ix1 = tx + tw - inner_margin
        iy1 = ty + th - inner_margin
        if ix1 <= ix0 or iy1 <= iy0:
            ix0, iy0 = tx, ty
            ix1, iy1 = tx + tw, ty + th

        sub = local_mask[iy0:iy1, ix0:ix1]
        if sub.size == 0:
            return False
        return float(sub.mean()) >= 0.97

    def _draw_text_at_center(
        local_mask: np.ndarray,
        min_x2: int,
        min_y2: int,
        cx: int,
        cy: int,
        text: str,
        tw: int,
        th: int,
        inner_margin: int,
    ) -> bool:
        H_loc, W_loc = local_mask.shape
        tx = int(cx - tw // 2)
        ty = int(cy - th // 2)

        tx = max(0, min(W_loc - tw, tx))
        ty = max(0, min(H_loc - th, ty))

        if not _fits_rect(local_mask, tx, ty, tw, th, inner_margin):
            return False

        draw.text((min_x2 + tx, min_y2 + ty), text, fill=(0, 0, 0), font=font)
        return True

    def _place_one_label_in_window(
        local_mask: np.ndarray,
        dist: np.ndarray,
        min_x2: int,
        min_y2: int,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        text: str,
        tw: int,
        th: int,
        inner_margin: int,
        prefer_center: bool = True,
        max_tries: int = 6,
    ) -> bool:
        """
        Place ONE label inside [x0:x1, y0:y1] window.
        We pick a candidate point by "deepness" (distance to boundary) and optionally
        add a tiny bias towards the window center to avoid drifting to one corner.
        """
        H_loc, W_loc = local_mask.shape
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(W_loc, int(x1))
        y1 = min(H_loc, int(y1))
        if x1 <= x0 or y1 <= y0:
            return False

        # Candidate mask inside the window
        win_mask = local_mask[y0:y1, x0:x1]
        if not win_mask.any():
            return False

        win_dist = dist[y0:y1, x0:x1].copy()

        if prefer_center:
            # Small bias towards window center (cheap, prevents "all in one corner")
            cy = (y1 - y0 - 1) / 2.0
            cx = (x1 - x0 - 1) / 2.0
            yy, xx = np.ogrid[0:(y1 - y0), 0:(x1 - x0)]
            d2 = (yy - cy) * (yy - cy) + (xx - cx) * (xx - cx)
            # Normalize and subtract a small term so center is slightly preferred
            # (keeps primary criterion = deepness)
            win_dist = win_dist - 0.002 * d2

        # Try several peaks (in case the max doesn't fit the text rectangle)
        work = np.where(win_mask, win_dist, -1e9)

        for _ in range(int(max_tries)):
            idx = int(np.argmax(work))
            best = float(work.ravel()[idx])
            if best < -1e8:
                break

            wy = idx // (x1 - x0)
            wx = idx % (x1 - x0)
            cx_loc = x0 + wx
            cy_loc = y0 + wy

            if _draw_text_at_center(local_mask, min_x2, min_y2, cx_loc, cy_loc, text, tw, th, inner_margin):
                return True

            # Suppress a neighborhood around this failed point and retry
            rr = max(4, int(round(0.8 * float(font_size))))
            yy0 = max(0, wy - rr)
            yy1 = min((y1 - y0), wy + rr + 1)
            xx0 = max(0, wx - rr)
            xx1 = min((x1 - x0), wx + rr + 1)
            work[yy0:yy1, xx0:xx1] = -1e9

        return False

    def _split_into_grid_windows(W_loc: int, H_loc: int, target: int) -> List[Tuple[int, int, int, int]]:
        """
        Return list of (x0, x1, y0, y1) windows.

        - If elongated: use m=target along long axis, k=1.
        - Else: choose m,k roughly to get >= target windows and cover bbox.
        """
        if target <= 1:
            return [(0, W_loc, 0, H_loc)]

        aspect = float(W_loc) / float(max(1, H_loc))
        elongated = aspect >= 6.0 or (1.0 / aspect) >= 6.0

        windows: List[Tuple[int, int, int, int]] = []

        if elongated:
            if W_loc >= H_loc:
                m, k = target, 1
            else:
                m, k = 1, target
        else:
            # Choose m,k close to sqrt while respecting aspect
            # m * k >= target
            base = float(target) ** 0.5
            m = max(1, int(round(base * (aspect ** 0.5))))
            k = max(1, int(np.ceil(target / float(m))))
            # Cap to avoid silly splits
            m = min(m, 12)
            k = min(k, 12)
            # Ensure coverage
            if m * k < target:
                k = min(12, int(np.ceil(target / float(m))))

        # Build windows in row-major order
        xs = [int(round(i * W_loc / float(m))) for i in range(m + 1)]
        ys = [int(round(j * H_loc / float(k))) for j in range(k + 1)]

        for j in range(k):
            for i in range(m):
                x0, x1 = xs[i], xs[i + 1]
                y0, y1 = ys[j], ys[j + 1]
                if x1 > x0 and y1 > y0:
                    windows.append((x0, x1, y0, y1))

        return windows

    # ---------------- main loop ----------------
    for reg in final_regions:
        cid = reg["color_id"]
        if cid not in color_id_to_paint_index:
            continue

        number = color_id_to_paint_index[cid]
        text = str(number)
        tw, th = _get_text_size(text)

        area_hi = float(reg["area"]) * (scale * scale)
        if area_hi < float(tw * th) * 1.2:
            continue

        target = int(area_hi // (area_per_label_factor * float(tw * th)))
        target = max(1, min(int(max_repeats_per_region), target))

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

        # Too small: do not try multiple placements
        if int(local_mask.sum()) < int(tw * th * 2):
            target = 1

        inner_margin = max(1, font_size // 6)

        # Precompute distance transform ONCE per region (fast enough, much faster than erosions)
        dist = ndi.distance_transform_edt(local_mask)

        windows = _split_into_grid_windows(W_loc=W_loc, H_loc=H_loc, target=target)

        placed = 0

        if target == 1:
            # Single placement: whole bbox window
            ok = _place_one_label_in_window(
                local_mask=local_mask,
                dist=dist,
                min_x2=min_x2,
                min_y2=min_y2,
                x0=0,
                x1=W_loc,
                y0=0,
                y1=H_loc,
                text=text,
                tw=tw,
                th=th,
                inner_margin=inner_margin,
                prefer_center=True,
                max_tries=10,
            )
            if ok:
                continue
            # If failed, give up quietly (better than boundary junk)
            continue

        # Multi placement: place once per sub-window, stop when reached `target`
        # Order windows to spread along the dominant axis first:
        # - elongated horizontal: left->right
        # - elongated vertical: top->bottom
        # - general: row-major is ok
        aspect = float(W_loc) / float(max(1, H_loc))
        elongated = aspect >= 6.0 or (1.0 / aspect) >= 6.0
        if elongated and W_loc >= H_loc:
            windows.sort(key=lambda w: (w[0] + w[1]) / 2.0)  # by x center
        elif elongated:
            windows.sort(key=lambda w: (w[2] + w[3]) / 2.0)  # by y center

        for (x0, x1, y0, y1) in windows:
            if placed >= target:
                break

            ok = _place_one_label_in_window(
                local_mask=local_mask,
                dist=dist,
                min_x2=min_x2,
                min_y2=min_y2,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                text=text,
                tw=tw,
                th=th,
                inner_margin=inner_margin,
                prefer_center=True,
                max_tries=8,
            )
            if ok:
                placed += 1

        # Optional mild fallback: if we placed less than target, try again on the full region
        # but do not exceed target. This helps complex shapes where some windows are empty.
        while placed < target:
            ok = _place_one_label_in_window(
                local_mask=local_mask,
                dist=dist,
                min_x2=min_x2,
                min_y2=min_y2,
                x0=0,
                x1=W_loc,
                y0=0,
                y1=H_loc,
                text=text,
                tw=tw,
                th=th,
                inner_margin=inner_margin,
                prefer_center=False,  # now prefer deepness over window-center bias
                max_tries=6,
            )
            if not ok:
                break
            placed += 1


