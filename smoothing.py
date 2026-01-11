from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from colorspace import rgb_to_lab


def compute_strong_edges_lab(
    orig_arr: np.ndarray,
    quantile: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute strong edge masks between right and down neighbors using Lab distance.

    Returns
    -------
    edge_right : (H, W-1) bool
        True where edge between (y,x) and (y,x+1) is strong.
    edge_down : (H-1, W) bool
        True where edge between (y,x) and (y+1,x) is strong.
    """
    lab = rgb_to_lab(orig_arr).astype(np.float32)
    d_right = np.sum((lab[:, 1:, :] - lab[:, :-1, :]) ** 2, axis=2)
    d_down = np.sum((lab[1:, :, :] - lab[:-1, :, :]) ** 2, axis=2)

    # Use a global adaptive threshold based on image content.
    all_d = np.concatenate([d_right.reshape(-1), d_down.reshape(-1)], axis=0)
    thr = float(np.quantile(all_d, quantile))

    edge_right = d_right > thr
    edge_down = d_down > thr
    return edge_right, edge_down

def estimate_smoothing_radius_px(
    image_long_px: int,
    print_long_mm: float = 420.0,
    min_feature_mm: float = 2.0,
    oversample: float = 0.5,
    max_effective_dpi: int = 250,
) -> int:
    """
    Estimate smoothing window radius in pixels based on physical size.

    We assume an A3-like long side in millimetres and cap effective DPI
    so that over-detailed scans do not blow up processing.

    Returns
    -------
    int
        Radius of the smoothing window in pixels (at least 1).
    """
    max_effective_px = int(print_long_mm / 25.4 * max_effective_dpi)
    effective_long_px = min(image_long_px, max_effective_px)

    mm_per_px = print_long_mm / float(effective_long_px)
    min_feature_px = min_feature_mm / mm_per_px

    radius = int(round(min_feature_px * oversample / 2.0))
    return max(1, radius)


def smooth_labels_radius_scipy(
    labels: np.ndarray,
    num_labels: int,
    radius: int,
    iterations: int = 1,
) -> np.ndarray:
    """
    Fast majority vote smoothing in a (2*radius+1)x(2*radius+1) box window.

    Memory-optimized version:
      - no (num_labels, H, W) "votes" tensor
      - keep only best_score (H, W) and best_label (H, W)
    """
    if radius <= 0 or iterations <= 0 or num_labels <= 1:
        return labels

    labels = labels.astype(np.int32, copy=False)
    win = 2 * int(radius) + 1
    H, W = labels.shape

    out = labels
    for _ in range(int(iterations)):
        # Initialize with current labels so every pixel always has a label
        best_score = np.zeros((H, W), dtype=np.float32)
        best_label = out.copy()

        # For each label, compute local frequency and update argmax
        for k in range(num_labels):
            mask = (out == k)
            if not mask.any():
                continue

            # local "score" = mean of mask in window (0..1)
            score = ndi.uniform_filter(
                mask.astype(np.float32),
                size=win,
                mode="nearest",
            )

            better = score > best_score
            if not better.any():
                continue

            best_score[better] = score[better]
            best_label[better] = k

        out = best_label.astype(np.int32, copy=False)

    return out



def dilate_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    """
    Thicken a binary mask by roughly `thickness` pixels
    using simple 3x3 neighbourhood without OpenCV.
    """
    if thickness <= 1:
        return mask

    base = mask.copy()
    expanded = mask.copy()
    for _ in range(thickness - 1):
        tmp = expanded.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(base, shift=(dy, dx), axis=(0, 1))
                if dy > 0:
                    shifted[:dy, :] = False
                elif dy < 0:
                    shifted[dy:, :] = False
                if dx > 0:
                    shifted[:, :dx] = False
                elif dx < 0:
                    shifted[:, dx:] = False
                tmp |= shifted
        expanded = tmp
        base = expanded.copy()
    return expanded


def estimate_min_region_pixels(
    image_long_px: int,
    print_long_mm: float = 420.0,
    min_feature_mm: float = 2.0,
    area_factor: float = 4.0,
    max_effective_dpi: int = 250,
) -> int:
    """
    Estimate minimal region area in pixels that is considered
    paintable. Regions smaller than this will be merged into neighbours.
    """
    max_effective_px = int(print_long_mm / 25.4 * max_effective_dpi)
    effective_long_px = min(image_long_px, max_effective_px)

    mm_per_px = print_long_mm / float(effective_long_px)
    min_feature_px = min_feature_mm / mm_per_px

    area = area_factor * (min_feature_px ** 2)
    return max(1, int(round(area)))


def compute_edge_masks_lab(
    orig_arr: np.ndarray,
    q_weak: float = 0.75,
    q_strong: float = 0.94,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lab = rgb_to_lab(orig_arr).astype(np.float32)
    d_right = np.sum((lab[:, 1:, :] - lab[:, :-1, :]) ** 2, axis=2)
    d_down = np.sum((lab[1:, :, :] - lab[:-1, :, :]) ** 2, axis=2)

    all_d = np.concatenate([d_right.reshape(-1), d_down.reshape(-1)], axis=0)
    thr_weak = float(np.quantile(all_d, q_weak))
    thr_strong = float(np.quantile(all_d, q_strong))

    weak_right = d_right > thr_weak
    weak_down = d_down > thr_weak
    strong_right = d_right > thr_strong
    strong_down = d_down > thr_strong
    return weak_right, weak_down, strong_right, strong_down
