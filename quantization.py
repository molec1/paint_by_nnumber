from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from colorspace import rgb_to_lab, lab_to_rgb
from smoothing import estimate_smoothing_radius_px, smooth_labels_radius


def quantize_kmeans_lab(
    orig_arr: np.ndarray,
    num_colors: int,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], np.ndarray]:
    """
    KMeans quantization in Lab space.

    Returns
    -------
    cluster_id_raw : np.ndarray (H, W)
        Cluster ID for each pixel.
    palette_colors : list of (R, G, B)
        RGB palette centers.
    quant_arr : np.ndarray (H, W, 3)
        Quantized RGB image.
    """
    H, W, _ = orig_arr.shape

    lab = rgb_to_lab(orig_arr)
    flat_lab = lab.reshape(-1, 3)

    n_pixels = flat_lab.shape[0]
    max_sample = 200_000
    if n_pixels > max_sample:
        idx = np.random.choice(n_pixels, max_sample, replace=False)
        sample = flat_lab[idx]
    else:
        sample = flat_lab

    kmeans = KMeans(n_clusters=num_colors, n_init=5, random_state=42)
    kmeans.fit(sample)

    labels = kmeans.predict(flat_lab)          # (N,)
    centers_lab = kmeans.cluster_centers_      # (K, 3)

    cluster_id_raw = labels.reshape(H, W)      # 0..K-1
    centers_rgb = lab_to_rgb(centers_lab)      # (K, 3) uint8
    palette_colors = [tuple(map(int, c)) for c in centers_rgb]

    palette_np = np.array(palette_colors, dtype=np.uint8)
    quant_arr = palette_np[cluster_id_raw]

    return cluster_id_raw, palette_colors, quant_arr


def smooth_cluster_map(
    cluster_id_raw,
    palette_colors,
    image_long_px,
    print_long_mm,
    min_feature_mm,
    area_factor,
    max_effective_dpi,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Smooth the cluster map with a radius linked to physical print size
    and recompute the palette for resulting cluster IDs.
    """
    num_initial_colors = len(palette_colors)

    radius = estimate_smoothing_radius_px(
    image_long_px=image_long_px,
    print_long_mm=print_long_mm,
    min_feature_mm=min_feature_mm,
    oversample=0.5,
    max_effective_dpi=max_effective_dpi,
)

    print(f"[2] Smoothing window radius: {radius} pixels for {print_long_mm:.0f} mm long side")

    cluster_id_smoothed = smooth_labels_radius(
        cluster_id_raw,
        num_labels=num_initial_colors,
        radius=radius,
        iterations=1,
    )

    final_cids, inverse2 = np.unique(
        cluster_id_smoothed, axis=None, return_inverse=True
    )
    H, W = cluster_id_raw.shape
    cluster_id_img = inverse2.reshape(H, W)  # 0..K'-1

    palette_final = [palette_colors[int(c)] for c in final_cids]
    return cluster_id_img, palette_final
