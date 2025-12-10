from __future__ import annotations

import numpy as np


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


def smooth_labels_radius(
    cluster_id: np.ndarray,
    num_labels: int,
    radius: int,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply a simple mode filter over a square window (2*radius+1)^2
    for label maps.

    Parameters
    ----------
    cluster_id : np.ndarray
        2D array of int32 cluster IDs [0..num_labels-1].
    num_labels : int
        Number of labels.
    radius : int
        Window radius in pixels.
    iterations : int
        Number of passes.

    Returns
    -------
    np.ndarray
        Smoothed label map of the same shape.
    """
    h, w = cluster_id.shape
    arr = cluster_id.copy()

    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]

    for _ in range(iterations):
        counts = np.zeros((num_labels, h, w), dtype=np.int16)

        for k in range(num_labels):
            mask = (arr == k)
            acc = np.zeros_like(mask, dtype=np.int16)

            for dy, dx in offsets:
                shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))

                if dy > 0:
                    shifted[:dy, :] = False
                elif dy < 0:
                    shifted[dy:, :] = False
                if dx > 0:
                    shifted[:, :dx] = False
                elif dx < 0:
                    shifted[:, dx:] = False

                acc += shifted.astype(np.int16)

            counts[k] = acc

        arr = counts.argmax(axis=0).astype(np.int32)

    return arr


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
