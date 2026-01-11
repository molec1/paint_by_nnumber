from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
from scipy import ndimage as ndi

from config import NEIGHBORS, CONNECTIVITY4


def segment_regions_scipy(cluster_id_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Connected components by color id using SciPy (fast, no Python BFS).

    Returns
    -------
    region_id_img : (H, W) int32
        Region id per pixel, 0..R-1.
    region_color_id : (R,) int32
        Color id (cluster) for each region.
    region_area : (R,) int32
        Pixel area per region.
    """
    H, W = cluster_id_img.shape
    structure = ndi.generate_binary_structure(2, 1 if CONNECTIVITY4 else 2)

    region_id_img = -np.ones((H, W), dtype=np.int32)
    region_color_id: list[int] = []
    region_area: list[int] = []

    offset = 0
    # iterate only colors that actually exist
    for cid in np.unique(cluster_id_img):
        mask = (cluster_id_img == cid)
        if not mask.any():
            continue

        labeled, n = ndi.label(mask, structure=structure)
        if n == 0:
            continue

        # areas: bincount over labeled (0 is background)
        areas = np.bincount(labeled.ravel())[1:].astype(np.int32)

        # map local labels 1..n -> global region ids offset..offset+n-1
        rid = labeled.astype(np.int32)
        rid[rid > 0] = (rid[rid > 0] - 1) + offset

        region_id_img[mask] = rid[mask]

        region_color_id.extend([int(cid)] * n)
        region_area.extend(areas.tolist())
        offset += n

    return region_id_img, np.asarray(region_color_id, dtype=np.int32), np.asarray(region_area, dtype=np.int32)


def build_region_adjacency(region_id_img: np.ndarray, num_regions: int) -> list[set[int]]:
    """
    Build undirected region adjacency from region_id_img by scanning right/down borders.
    """
    adj: list[set[int]] = [set() for _ in range(num_regions)]
    r = region_id_img

    # right neighbors
    a = r[:, :-1]
    b = r[:, 1:]
    m = (a != b) & (a >= 0) & (b >= 0)
    p1 = np.stack([a[m], b[m]], axis=1) if m.any() else np.empty((0, 2), dtype=np.int32)

    # down neighbors
    a = r[:-1, :]
    b = r[1:, :]
    m = (a != b) & (a >= 0) & (b >= 0)
    p2 = np.stack([a[m], b[m]], axis=1) if m.any() else np.empty((0, 2), dtype=np.int32)

    pairs = np.vstack([p1, p2])
    if pairs.size == 0:
        return adj

    # normalize ordering so (u,v) and (v,u) are the same before unique
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)

    for u, v in pairs:
        uu = int(u); vv = int(v)
        if uu != vv:
            adj[uu].add(vv)
            adj[vv].add(uu)

    return adj



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


def merge_small_regions_scipy(
    region_id_img: np.ndarray,
    region_color_id: np.ndarray,
    region_area: np.ndarray,
    palette: list[tuple[int, int, int]],
    min_region_pixels: int,
    allow_fallback: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Merge regions smaller than min_region_pixels into neighbor regions.
    Produces a new cluster_id map and compacted palette (0..K'-1).
    """
    num_regions = int(region_area.shape[0])
    adj = build_region_adjacency(region_id_img, num_regions)

    big = region_area >= min_region_pixels
    small_ids = np.flatnonzero(~big)
    if small_ids.size == 0:
        # already clean, just compact colors
        cluster_id_mod = region_color_id[region_id_img]
        final_cids, inv = np.unique(cluster_id_mod, return_inverse=True)
        return inv.reshape(cluster_id_mod.shape), [palette[int(c)] for c in final_cids]

    pal = np.asarray(palette, dtype=np.int16)  # (K,3) for fast distance
    region_to_cid = region_color_id.copy()

    for sid in small_ids.tolist():
        scid = int(region_to_cid[sid])
        sclr = pal[scid]

        neigh = adj[sid]
        if not neigh:
            continue

        # prefer big neighbors
        candidates = [nid for nid in neigh if big[nid]]
        if not candidates:
            if not allow_fallback:
                continue
            # fallback: merge into the largest neighbor (big or small)
            nid = max(neigh, key=lambda x: int(region_area[x]))
            region_to_cid[sid] = int(region_to_cid[nid])
            continue

        # pick closest-by-color big neighbor
        cand_cids = region_to_cid[np.asarray(candidates, dtype=np.int32)]
        cand_colors = pal[cand_cids]
        d2 = np.sum((cand_colors - sclr) ** 2, axis=1)
        best = candidates[int(np.argmin(d2))]
        region_to_cid[sid] = int(region_to_cid[best])

    # rebuild pixel map in one shot
    cluster_id_mod = region_to_cid[region_id_img]

    # compact palette to 0..K'-1
    final_cids, inv = np.unique(cluster_id_mod, return_inverse=True)
    cluster_id_final = inv.reshape(cluster_id_mod.shape)
    palette_final = [palette[int(c)] for c in final_cids]
    return cluster_id_final, palette_final


def segment_final_regions_scipy(cluster_id_final: np.ndarray, connectivity4: bool = True) -> list[dict]:
    """
    Final connected regions with bbox + centroid.
    Faster than center_of_mass + find_objects by using bincount-based stats.

    Returns dicts with keys: color_id, cx, cy, bbox, area.
    bbox format: (y0, y1, x0, x1), inclusive.
    """
    H, W = cluster_id_final.shape
    structure = ndi.generate_binary_structure(2, 1 if connectivity4 else 2)

    yy, xx = np.indices((H, W), dtype=np.int32)
    final_regions: list[dict] = []

    for cid in np.unique(cluster_id_final):
        mask = (cluster_id_final == cid)
        if not mask.any():
            continue

        labeled, n = ndi.label(mask, structure=structure)
        if n == 0:
            continue

        lab = labeled.ravel()
        # areas for labels 1..n
        area = np.bincount(lab)[1:].astype(np.int32)

        # centroid: sum(x)/area, sum(y)/area using weights
        sum_x = np.bincount(lab, weights=xx.ravel())[1:]
        sum_y = np.bincount(lab, weights=yy.ravel())[1:]
        cx = sum_x / np.maximum(area, 1)
        cy = sum_y / np.maximum(area, 1)

        # bbox: min/max x/y per label via bincount-style reduction
        # We compute mins/maxs by initializing with sentinel and using np.minimum/maximum with ufunc.at
        x_min = np.full(n + 1, W, dtype=np.int32)
        x_max = np.full(n + 1, -1, dtype=np.int32)
        y_min = np.full(n + 1, H, dtype=np.int32)
        y_max = np.full(n + 1, -1, dtype=np.int32)

        # only consider foreground pixels (label > 0)
        fg = lab > 0
        lab_fg = lab[fg]
        x_fg = xx.ravel()[fg]
        y_fg = yy.ravel()[fg]

        np.minimum.at(x_min, lab_fg, x_fg)
        np.maximum.at(x_max, lab_fg, x_fg)
        np.minimum.at(y_min, lab_fg, y_fg)
        np.maximum.at(y_max, lab_fg, y_fg)

        for i in range(1, n + 1):
            a = int(area[i - 1])
            if a <= 0:
                continue

            final_regions.append(
                {
                    "color_id": int(cid),
                    "cx": float(cx[i - 1]),
                    "cy": float(cy[i - 1]),
                    "bbox": (int(y_min[i]), int(y_max[i]), int(x_min[i]), int(x_max[i])),
                    "area": a,
                }
            )

    return final_regions


def clean_small_final_regions(
    cluster_id_final: np.ndarray,
    palette: list[tuple[int, int, int]],
    min_final_region_pixels: int,
    return_regions: bool = True,
):
    region_id_img, region_color_id, region_area = segment_regions_scipy(cluster_id_final)

    big_mask = region_area >= int(min_final_region_pixels)
    big_ids = np.flatnonzero(big_mask)
    small_ids = np.flatnonzero(~big_mask)

    print(f"    [clean] final big regions: {big_ids.size}, small regions: {small_ids.size}")

    if small_ids.size == 0 or big_ids.size == 0:
        if return_regions:
            final_regions = segment_final_regions_scipy(cluster_id_final)
            return cluster_id_final, palette, final_regions
        return cluster_id_final, palette

    cluster_id_refined, palette_refined = merge_small_regions_scipy(
        region_id_img=region_id_img,
        region_color_id=region_color_id,
        region_area=region_area,
        palette=palette,
        min_region_pixels=int(min_final_region_pixels),
        allow_fallback=True,
    )

    if return_regions:
        final_regions = segment_final_regions_scipy(cluster_id_refined)
        return cluster_id_refined, palette_refined, final_regions
    return cluster_id_refined, palette_refined


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

    for it in range(int(max_iters)):
        region_id_img, region_color_id, region_area = segment_regions_scipy(current_map)

        small_mask = region_area < int(hard_min_pixels)
        num_regions = int(region_area.size)
        num_small = int(np.count_nonzero(small_mask))
        num_big = int(num_regions - num_small)

        print(f"[hard-clean {it}] regions={num_regions}, big={num_big}, small={num_small}")

        if num_small == 0:
            break

        # Merge all regions smaller than hard_min_pixels. Use fallback to guarantee progress.
        current_map, current_palette = merge_small_regions_scipy(
            region_id_img=region_id_img,
            region_color_id=region_color_id,
            region_area=region_area,
            palette=current_palette,
            min_region_pixels=int(hard_min_pixels),
            allow_fallback=True,
        )

    return current_map, current_palette


def iterative_final_cleanup(
    cluster_id: np.ndarray,
    palette: list[tuple[int, int, int]],
    min_region_pixels: int,
    max_tiny_regions: int = 10,
    max_iters: int = 3,
    connectivity4: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[dict]]:
    """
    Combined final clean-up:

      * repeatedly merges all regions smaller than `min_region_pixels`
      * stops when number of remaining small regions is <= `max_tiny_regions`
        or when `max_iters` is reached
      * returns final connected regions ready for rendering

    This replaces the separate clean_small_final_regions + hard_cleanup_tiny_regions
    passes and runs segmentation/merging only as many times as actually needed.
    """
    current_map = cluster_id
    current_palette = palette

    for it in range(int(max_iters)):
        region_id_img, region_color_id, region_area = segment_regions_scipy(current_map)

        small_mask = region_area < int(min_region_pixels)
        num_regions = int(region_area.size)
        num_small = int(np.count_nonzero(small_mask))
        num_big = int(num_regions - num_small)

        print(
            f"[7/clean {it}] regions={num_regions}, big={num_big}, small={num_small}"
        )

        # Good enough: allow up to `max_tiny_regions` small pieces to survive
        if num_small <= int(max_tiny_regions) or num_big == 0:
            break

        # Merge all small regions into neighbours, palette is compacted inside
        current_map, current_palette = merge_small_regions_scipy(
            region_id_img=region_id_img,
            region_color_id=region_color_id,
            region_area=region_area,
            palette=current_palette,
            min_region_pixels=int(min_region_pixels),
            allow_fallback=True,
        )

    # Final regions with bboxes and centroids for number drawing
    final_regions = segment_final_regions_scipy(
        current_map,
        connectivity4=connectivity4,
    )
    return current_map, current_palette, final_regions
