from __future__ import annotations

import numpy as np

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


def smooth_labels_radius(
    cluster_id: np.ndarray,
    num_labels: int,
    radius: int,
    iterations: int = 2,
    edge_right: np.ndarray | None = None,  # (H, W-1) bool
    edge_down: np.ndarray | None = None,   # (H-1, W) bool
) -> np.ndarray:
    """
    Apply a mode filter over a square window (2*radius+1)^2 for label maps.
    If edge masks are provided, votes across strong edges are ignored.
    """
    h, w = cluster_id.shape
    arr = cluster_id.copy()

    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if not (dy == 0 and dx == 0)
    ]

    use_edges = (edge_right is not None) and (edge_down is not None)

    for _ in range(iterations):
        counts = np.zeros((num_labels, h, w), dtype=np.int16)

        for k in range(num_labels):
            mask = (arr == k)

            # Start with self-vote to stabilize isolated pixels.
            acc = mask.astype(np.int16)

            for dy, dx in offsets:
                shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))

                # Zero out wrapped areas.
                if dy > 0:
                    shifted[:dy, :] = False
                elif dy < 0:
                    shifted[dy:, :] = False
                if dx > 0:
                    shifted[:, :dx] = False
                elif dx < 0:
                    shifted[:, dx:] = False
                if use_edges:
                    allow = np.ones((h, w), dtype=bool)

                    # Horizontal edge protection (edge_right has shape (h, w-1))
                    if dx > 0:
                        # Receiving pixels are x in [dx .. w-1]
                        # Block if the boundary just left of the receiving pixel is strong:
                        # boundary index is (x-1) in edge_right.
                        allow[:, dx:] &= ~edge_right[:, (dx - 1) :]

                        # Wrapped area should not contribute anyway.
                        allow[:, :dx] = False
                    elif dx < 0:
                        ddx = -dx
                        # Receiving pixels are x in [0 .. w-ddx-1]
                        # Block if the boundary just right of the receiving pixel is strong:
                        # boundary index is x in edge_right.
                        allow[:, : w - ddx] &= ~edge_right[:, : w - ddx]
                        allow[:, w - ddx :] = False

                    # Vertical edge protection (edge_down has shape (h-1, w))
                    if dy > 0:
                        # Receiving pixels are y in [dy .. h-1]
                        # Block if the boundary just above the receiving pixel is strong:
                        # boundary index is (y-1) in edge_down.
                        allow[dy:, :] &= ~edge_down[(dy - 1) :, :]
                        allow[:dy, :] = False
                    elif dy < 0:
                        ddy = -dy
                        # Receiving pixels are y in [0 .. h-ddy-1]
                        # Block if the boundary just below the receiving pixel is strong:
                        # boundary index is y in edge_down.
                        allow[: h - ddy, :] &= ~edge_down[: h - ddy, :]
                        allow[h - ddy :, :] = False

                    shifted = shifted & allow


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
