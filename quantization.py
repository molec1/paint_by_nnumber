from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

from colorspace import rgb_to_lab, lab_to_rgb
from smoothing import estimate_smoothing_radius_px, smooth_labels_radius_scipy


def quantize_kmeans_lab(
    orig_arr: np.ndarray,
    num_colors: int,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], np.ndarray]:
    """
    KMeans quantization in Lab space.

    Steps:
      1) Convert full image to Lab.
      2) Take a random subset of pixels for fitting k-means.
      3) Fit k-means on this subset.
      4) Train a decision tree on (Lab_subset -> kmeans.labels_).
      5) Use the tree to predict cluster IDs for all pixels (in batches).
      6) Build RGB palette and quantized image.

    This avoids building any large (N x K) distance matrices.
    """
    H, W, _ = orig_arr.shape

    # 1) RGB -> Lab (float32)
    lab = rgb_to_lab(orig_arr)          # (H, W, 3) float32
    flat_lab = lab.reshape(-1, 3)       # view, (N, 3)
    n_pixels = flat_lab.shape[0]

    # 2) Sample subset for k-means and tree training
    max_sample = 200_000
    if n_pixels > max_sample:
        sample_idx = np.random.choice(n_pixels, max_sample, replace=False)
        sample_lab = flat_lab[sample_idx]
    else:
        sample_lab = flat_lab

    # 3) Fit k-means on the subset
    kmeans = KMeans(
        n_clusters=num_colors,
        n_init=2,
        random_state=42,
    )
    kmeans.fit(sample_lab)

    # cluster labels for the subset (training targets for the tree)
    sample_labels = kmeans.labels_.astype(np.int8, copy=False)

    # 4) Train decision tree on (Lab -> cluster_id)
    tree = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=50,
        random_state=42,
    )
    tree.fit(sample_lab, sample_labels)

    # 5) Predict cluster for all pixels using the tree (batched for safety)
    labels_all = np.empty(n_pixels, dtype=np.int8)
    batch_size = 1_000_000

    for start in range(0, n_pixels, batch_size):
        end = min(n_pixels, start + batch_size)
        block = flat_lab[start:end]         # (B, 3) float32
        labels_all[start:end] = tree.predict(block).astype(np.int8, copy=False)

    cluster_id_raw = labels_all.reshape(H, W)

    # 6) Palette from k-means centers
    centers_lab = kmeans.cluster_centers_.astype(np.float32, copy=False)   # (K, 3)
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
    orig_arr,
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
    
    cluster_id_smoothed = smooth_labels_radius_scipy(
        cluster_id_raw,
        num_labels=num_initial_colors,
        radius=radius,
        iterations=2,
    )
    final_cids, inverse2 = np.unique(
        cluster_id_smoothed, axis=None, return_inverse=True
    )
    H, W = cluster_id_raw.shape
    cluster_id_img = inverse2.reshape(H, W)  # 0..K'-1

    palette_final = [palette_colors[int(c)] for c in final_cids]
    return cluster_id_img, palette_final
