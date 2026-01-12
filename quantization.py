from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

from colorspace import lab_to_rgb
from smoothing import estimate_smoothing_radius_px, smooth_labels_radius_scipy


def rgb_to_lab(
    rgb: np.ndarray,
    out_dtype: np.dtype = np.float16,
    block_rows: int = 256,
) -> np.ndarray:
    """
    Convert an sRGB uint8 image (H, W, 3) to CIE Lab with low peak memory.

    - Processes the image in row blocks to avoid large temporary arrays.
    - Uses float32 internally for math, then casts to out_dtype at the end.

    Parameters
    ----------
    rgb : np.ndarray
        Input image, shape (H, W, 3), dtype uint8, sRGB color space.
    out_dtype : np.dtype, optional
        Output dtype for Lab, e.g. np.float16 or np.float32.
    block_rows : int, optional
        Number of rows to process per block.

    Returns
    -------
    lab : np.ndarray
        Lab image, shape (H, W, 3), dtype=out_dtype.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "rgb must be (H, W, 3)"
    assert rgb.dtype == np.uint8, "rgb must be uint8"

    H, W, _ = rgb.shape

    # Allocate output Lab array once.
    lab = np.empty((H, W, 3), dtype=out_dtype)

    # Constants for sRGB -> XYZ (D65) and XYZ -> Lab.
    # sRGB to XYZ matrix (D65)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )

    # Reference white D65 in XYZ (normalized Y=1.0)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    # Lab constants
    delta = 6.0 / 29.0
    delta2 = delta * delta
    delta3 = delta2 * delta

    def f_func(t: np.ndarray) -> np.ndarray:
        """
        Helper for Lab nonlinearity f(t).
        Works in-place-friendly style: returns new array, but caller can reuse.
        """
        # t is float32
        out = np.empty_like(t, dtype=np.float32)
        mask = t > delta3
        # Where t is big enough
        out[mask] = np.cbrt(t[mask])
        # Where t is small
        out[~mask] = t[~mask] * (1.0 / (3.0 * delta2)) + (4.0 / 29.0)
        return out

    # Process image in row blocks
    for start in range(0, H, block_rows):
        end = min(H, start + block_rows)

        # 1) Extract chunk and convert to float32 [0, 1]
        # rgb_chunk: (R, G, B) in [0, 255], uint8
        rgb_chunk = rgb[start:end]  # view, uint8
        # float_chunk: float32 in [0, 1]
        float_chunk = rgb_chunk.astype(np.float32) * (1.0 / 255.0)

        # 2) sRGB gamma correction (inverse companding) in-place
        # linearize sRGB
        # small values: c / 12.92
        # large values: ((c + 0.055) / 1.055) ** 2.4
        for ch in range(3):
            c = float_chunk[..., ch]
            mask = c > 0.04045
            # big branch
            c_big = c[mask]
            c[mask] = ((c_big + 0.055) / 1.055) ** 2.4
            # small branch
            c_small = c[~mask]
            c[~mask] = c_small / 12.92

        # 3) Convert linear RGB to XYZ
        R = float_chunk[..., 0]
        G = float_chunk[..., 1]
        B = float_chunk[..., 2]

        X = M[0, 0] * R + M[0, 1] * G + M[0, 2] * B
        Y = M[1, 0] * R + M[1, 1] * G + M[1, 2] * B
        Z = M[2, 0] * R + M[2, 1] * G + M[2, 2] * B

        # 4) Normalize by reference white
        X /= Xn
        Y /= Yn
        Z /= Zn

        # 5) Apply f(t) for Lab
        fX = f_func(X)
        fY = f_func(Y)
        fZ = f_func(Z)

        # 6) Compute Lab
        L = 116.0 * fY - 16.0
        a = 500.0 * (fX - fY)
        b = 200.0 * (fY - fZ)

        # 7) Store into output (cast to out_dtype)
        lab_block = lab[start:end]
        lab_block[..., 0] = L.astype(out_dtype, copy=False)
        lab_block[..., 1] = a.astype(out_dtype, copy=False)
        lab_block[..., 2] = b.astype(out_dtype, copy=False)

    return lab

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
    lab = rgb_to_lab(orig_arr, out_dtype=np.float16)
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
    centers_lab = kmeans.cluster_centers_.astype(np.float32, copy=False)   # (K, 3)
    del kmeans

    # 4) Train decision tree on (Lab -> cluster_id)
    tree = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=50,
        random_state=42,
    )
    tree.fit(sample_lab, sample_labels)
    del sample_lab

    # 5) Predict cluster for all pixels using the tree (batched for safety)
    labels_all = np.empty(n_pixels, dtype=np.int8)
    batch_size = 1_000_000

    for start in range(0, n_pixels, batch_size):
        end = min(n_pixels, start + batch_size)
        block = flat_lab[start:end]         # (B, 3) float32
        labels_all[start:end] = tree.predict(block).astype(np.int8, copy=False)

    cluster_id_raw = labels_all.reshape(H, W)

    # 6) Palette from k-means centers
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
    cluster_id_img = inverse2.reshape(H, W) .astype(np.uint8, copy=False) # 0..K'-1

    palette_final = [palette_colors[int(c)] for c in final_cids]
    return cluster_id_img, palette_final
