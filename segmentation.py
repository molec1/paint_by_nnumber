from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np

from config import NEIGHBORS


def segment_regions(cluster_id_img: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
    """
    First segmentation pass: connected components by color ID.

    Returns
    -------
    regions : list of dict
        Each region has keys: id, color_id, pixels, area.
    region_id_img : np.ndarray
        2D map of region IDs with same shape as cluster_id_img.
    """
    H, W = cluster_id_img.shape
    visited = np.zeros((H, W), dtype=bool)
    region_id_img = -np.ones((H, W), dtype=np.int32)
    regions: List[Dict] = []

    for y0 in range(H):
        for x0 in range(W):
            if visited[y0, x0]:
                continue

            cid = int(cluster_id_img[y0, x0])
            visited[y0, x0] = True

            region_idx = len(regions)
            queue = deque([(y0, x0)])
            pixels = []

            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))
                region_id_img[y, x] = region_idx

                for dy, dx in NEIGHBORS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                        if int(cluster_id_img[ny, nx]) == cid:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            regions.append(
                {
                    "id": region_idx,
                    "color_id": cid,
                    "pixels": pixels,
                    "area": len(pixels),
                }
            )

    return regions, region_id_img


def split_big_small_regions(
    regions: List[Dict],
    min_region_pixels: int,
) -> Tuple[Set[int], Set[int]]:
    """
    Split regions into big and small by area threshold.
    """
    big_region_ids: Set[int] = set()
    small_region_ids: Set[int] = set()

    for reg in regions:
        if reg["area"] >= min_region_pixels:
            big_region_ids.add(reg["id"])
        else:
            small_region_ids.add(reg["id"])

    return big_region_ids, small_region_ids


def build_adjacency_small_to_big(
    region_id_img: np.ndarray,
    big_region_ids: Set[int],
    small_region_ids: Set[int],
) -> Dict[int, Set[int]]:
    """
    Build adjacency graph: small region -> set of neighbouring big regions.
    """
    H, W = region_id_img.shape
    adj_small_to_big: Dict[int, Set[int]] = {rid: set() for rid in small_region_ids}

    for y in range(H):
        for x in range(W):
            r1 = int(region_id_img[y, x])
            for dy, dx in NEIGHBORS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    r2 = int(region_id_img[ny, nx])
                    if r1 == r2 or r1 < 0 or r2 < 0:
                        continue
                    if r1 in small_region_ids and r2 in big_region_ids:
                        adj_small_to_big[r1].add(r2)
                    if r2 in small_region_ids and r1 in big_region_ids:
                        adj_small_to_big[r2].add(r1)

    return adj_small_to_big


def merge_small_regions(
    cluster_id_img: np.ndarray,
    regions: List[Dict],
    palette: List[Tuple[int, int, int]],
    big_region_ids: Set[int],
    small_region_ids: Set[int],
    adj_small_to_big: Dict[int, Set[int]],
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Merge small regions into neighbouring big ones, using color distance
    in RGB space to choose the best target.

    Returns
    -------
    cluster_id_final : np.ndarray
        Map of color IDs after merging.
    palette_final : list of (R, G, B)
        Palette matching new IDs.
    """
    H, W = cluster_id_img.shape
    cluster_id_mod = cluster_id_img.copy()

    for sid in small_region_ids:
        reg_small = regions[sid]
        small_cid = reg_small["color_id"]
        small_color = np.array(palette[small_cid], dtype=np.int16)

        candidates = list(adj_small_to_big[sid])
        if not candidates:
            continue

        best_bid = None
        best_dist = None

        for bid in candidates:
            reg_big = regions[bid]
            big_cid = reg_big["color_id"]
            big_color = np.array(palette[big_cid], dtype=np.int16)
            dist = np.sum((small_color - big_color) ** 2)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_bid = bid

        if best_bid is None:
            continue

        target_cid = regions[best_bid]["color_id"]

        for (yy, xx) in reg_small["pixels"]:
            cluster_id_mod[yy, xx] = target_cid

    final_cids, inverse = np.unique(cluster_id_mod, axis=None, return_inverse=True)
    cluster_id_final = inverse.reshape(H, W)
    palette_final = [palette[int(c)] for c in final_cids]

    return cluster_id_final, palette_final


def segment_final_regions(cluster_id_final: np.ndarray) -> List[Dict]:
    """
    Second segmentation pass: final connected regions after all merges.

    Each region dict includes:
      - color_id
      - pixels
      - cx, cy : approximate centroid
      - bbox   : (min_y, max_y, min_x, max_x)
      - area   : number of pixels
    """
    H, W = cluster_id_final.shape
    visited = np.zeros((H, W), dtype=bool)
    final_regions: List[Dict] = []

    for y0 in range(H):
        for x0 in range(W):
            if visited[y0, x0]:
                continue

            cid = int(cluster_id_final[y0, x0])
            visited[y0, x0] = True

            queue = deque([(y0, x0)])
            pixels = []
            min_y = max_y = y0
            min_x = max_x = x0

            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))

                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x

                for dy, dx in NEIGHBORS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                        if int(cluster_id_final[ny, nx]) == cid:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            ys = np.array([p[0] for p in pixels])
            xs = np.array([p[1] for p in pixels])
            cy = int(round(ys.mean()))
            cx = int(round(xs.mean()))

            final_regions.append(
                {
                    "color_id": cid,
                    "pixels": pixels,
                    "cx": cx,
                    "cy": cy,
                    "bbox": (min_y, max_y, min_x, max_x),
                    "area": len(pixels),
                }
            )

    return final_regions


def clean_small_final_regions(
    cluster_id_final: np.ndarray,
    palette: List[Tuple[int, int, int]],
    min_final_region_pixels: int,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], List[Dict]]:
    """
    Optional refinement pass on the final map:

    - Segment into regions again.
    - Split into big/small by min_final_region_pixels.
    - Merge small regions into neighbouring big regions (by color distance).
    - Run final segmentation once more to get the definitive region list.

    Returns
    -------
    cluster_id_refined : np.ndarray
    palette_refined : list of (R, G, B)
    final_regions : list of dict
    """
    regions2, region_id2 = segment_regions(cluster_id_final)
    big2, small2 = split_big_small_regions(regions2, min_final_region_pixels)
    print(f"    [clean] final big regions: {len(big2)}, small regions: {len(small2)}")

    # If everything is either small or big, just return the segmentation as is.
    if not small2 or not big2:
        final_regions = segment_final_regions(cluster_id_final)
        return cluster_id_final, palette, final_regions

    adj2 = build_adjacency_small_to_big(region_id2, big2, small2)

    cluster_id_refined, palette_refined = merge_small_regions(
        cluster_id_final,
        regions2,
        palette,
        big2,
        small2,
        adj2,
    )

    final_regions = segment_final_regions(cluster_id_refined)
    return cluster_id_refined, palette_refined, final_regions



def merge_small_regions_with_fallback(
    cluster_id_img: np.ndarray,
    regions: list[dict],
    palette: list[tuple[int, int, int]],
    big_region_ids: set[int],
    small_region_ids: set[int],
    adj_small_to_big: dict[int, set[int]],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Merge all small regions into neighbours:
    - prefer big regions with closest color
    - if no big neighbours, merge into the largest neighbour (by area),
      even if it is also small.

    This guarantees progress: every small region is merged somewhere.
    """
    H, W = cluster_id_img.shape
    cluster_id_mod = cluster_id_img.copy()

    # precompute area for fast lookup
    area_by_id = {reg["id"]: reg["area"] for reg in regions}

    for sid in small_region_ids:
        reg_small = regions[sid]
        small_cid = reg_small["color_id"]
        small_color = np.array(palette[small_cid], dtype=np.int16)

        candidates_big = list(adj_small_to_big.get(sid, []))

        best_target_region_id = None

        if candidates_big:
            # normal path: pick closest big region by color
            best_dist = None
            for bid in candidates_big:
                reg_big = regions[bid]
                big_cid = reg_big["color_id"]
                big_color = np.array(palette[big_cid], dtype=np.int16)
                dist = np.sum((small_color - big_color) ** 2)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_target_region_id = bid
        else:
            # fallback: нет ни одного big-соседа
            # берём всех соседей (big+small), выбираем с максимальной площадью
            neighbour_ids = set()
            # обходим пиксели региона и собираем соседей
            for (yy, xx) in reg_small["pixels"]:
                for dy, dx in NEIGHBORS:
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        nid = int(regions[cluster_id_img[ny, nx]]["id"])
                        if nid != sid:
                            neighbour_ids.add(nid)

            if neighbour_ids:
                best_target_region_id = max(
                    neighbour_ids,
                    key=lambda rid: area_by_id.get(rid, 0),
                )

        if best_target_region_id is None:
            # совсем одиночный пиксель без соседей (теоретически не должен случаться,
            # но оставим защиту)
            continue

        target_cid = regions[best_target_region_id]["color_id"]

        for (yy, xx) in reg_small["pixels"]:
            cluster_id_mod[yy, xx] = target_cid

    final_cids, inverse = np.unique(cluster_id_mod, axis=None, return_inverse=True)
    cluster_id_final = inverse.reshape(H, W)
    palette_merged = [palette[int(c)] for c in final_cids]

    return cluster_id_final, palette_merged


def hard_cleanup_tiny_regions(
    cluster_id: np.ndarray,
    palette: list[tuple[int, int, int]],
    hard_min_pixels: int,
    max_iters: int = 3,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[dict]]:
    """
    Forcefully remove all regions smaller than hard_min_pixels by repeatedly
    merging them into neighbouring regions.

    Returns:
        cluster_id_cleaned, palette_cleaned, final_regions
    """
    current_map = cluster_id
    current_palette = palette

    for it in range(max_iters):
        regions, region_id_img = segment_regions(current_map)
        big_ids, small_ids = split_big_small_regions(regions, hard_min_pixels)

        print(
            f"[hard-clean {it}] regions={len(regions)}, "
            f"big={len(big_ids)}, small={len(small_ids)}"
        )

        if not small_ids:
            # nothing left to clean
            break

        # Важно: даже если big_ids пустой (все маленькие), всё равно строим
        # соседства и будем сливать маленькие в "самый большой сосед".
        adj = build_adjacency_small_to_big(region_id_img, big_ids, small_ids)

        # fallback: если у маленького нет big-соседа, будем сливать в
        # "наиболее крупного любого соседа". Для этого слегка расширим merge.
        current_map, current_palette = merge_small_regions_with_fallback(
            current_map,
            regions,
            current_palette,
            big_ids,
            small_ids,
            adj,
        )

    # финальная сегментация
    final_regions = segment_final_regions(current_map)
    return current_map, current_palette, final_regions