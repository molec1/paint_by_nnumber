from __future__ import annotations

import numpy as np


def rgb_to_lab(rgb_arr: np.ndarray) -> np.ndarray:
    """
    Convert sRGB uint8 [0..255] to CIE Lab (D65).

    Parameters
    ----------
    rgb_arr : np.ndarray
        uint8 array of shape (H, W, 3) or (N, 3).

    Returns
    -------
    np.ndarray
        float32 array with Lab values, same shape as input.
    """
    arr = rgb_arr.astype(np.float32) / 255.0

    # sRGB -> linear RGB
    mask = arr <= 0.04045
    arr_lin = np.empty_like(arr)
    arr_lin[mask] = arr[mask] / 12.92
    arr_lin[~mask] = ((arr[~mask] + 0.055) / 1.055) ** 2.4

    # linear RGB -> XYZ (sRGB, D65)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = arr_lin @ M.T

    # normalize by D65 white point
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t: np.ndarray) -> np.ndarray:
        t_cbrt = np.cbrt(t)
        return np.where(t > eps, t_cbrt, (kappa * t + 16.0) / 116.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    lab = np.stack([L, a, b], axis=-1)
    return lab.astype(np.float32)


def lab_to_rgb(lab_arr: np.ndarray) -> np.ndarray:
    """
    Convert CIE Lab (D65) to sRGB uint8 [0..255].

    Parameters
    ----------
    lab_arr : np.ndarray
        float32 array with shape (..., 3).

    Returns
    -------
    np.ndarray
        uint8 sRGB array with same leading shape.
    """
    L = lab_arr[..., 0]
    a = lab_arr[..., 1]
    b = lab_arr[..., 2]

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f_inv(t: np.ndarray) -> np.ndarray:
        t3 = t ** 3
        return np.where(t3 > eps, t3, (116.0 * t - 16.0) / kappa)

    x = f_inv(fx)
    y = f_inv(fy)
    z = f_inv(fz)

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x *= Xn
    y *= Yn
    z *= Zn

    # XYZ -> linear RGB
    M_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    xyz = np.stack([x, y, z], axis=-1)
    rgb_lin = xyz @ M_inv.T

    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)

    # linear RGB -> sRGB
    mask = rgb_lin <= 0.0031308
    rgb = np.empty_like(rgb_lin)
    rgb[mask] = 12.92 * rgb_lin[mask]
    rgb[~mask] = 1.055 * (rgb_lin[~mask] ** (1.0 / 2.4)) - 0.055

    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0 + 0.5).astype(np.uint8)
