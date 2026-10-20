"""Slope-based lighting for heightmaps."""

import numpy as np

# Direction towards the sun: from the north-west, 40 degrees above the horizon.
# x grows east, y grows south, z is up.
_SUN = np.array([-1.0, -1.0, 1.2])
SUN_DIR = _SUN / np.linalg.norm(_SUN)

# Light that reaches surfaces facing away from the sun (sky light).
AMBIENT = 0.35

# 4x4 ordered-dither thresholds in [0, 1).
_BAYER = (
    np.array(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32
    )
    + 0.5
) / 16


def lambert(heights: np.ndarray, spacing: float) -> np.ndarray:
    """Light level in [0, 1] for each interior cell of ``heights``.

    ``heights`` must have a one-cell border on every side: the result is two
    rows and two columns smaller. Using real neighbours instead of edge
    clamping is what keeps shading seamless across chunk borders.
    ``spacing`` is the world distance between neighbouring cells.
    """
    dzdx = (heights[1:-1, 2:] - heights[1:-1, :-2]) / (2 * spacing)
    dzdy = (heights[2:, 1:-1] - heights[:-2, 1:-1]) / (2 * spacing)
    # Surface normal of z = h(x, y) is (-dz/dx, -dz/dy, 1), normalised.
    inv_len = 1.0 / np.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)
    diffuse = (-dzdx * SUN_DIR[0] - dzdy * SUN_DIR[1] + SUN_DIR[2]) * inv_len
    return (AMBIENT + (1 - AMBIENT) * np.clip(diffuse, 0.0, 1.0)).astype(np.float32)


def slope(heights: np.ndarray, spacing: float, reach: int = 1) -> np.ndarray:
    """Steepness (rise over run) of each interior cell.

    Measured between the cells ``reach`` steps away on either side, so
    ``heights`` needs a border of ``reach`` cells and the result is that much
    smaller on every side. A wider reach gives the slope of the landscape
    rather than of each little bump.
    """
    r = reach
    inner = slice(r, heights.shape[0] - r), slice(r, heights.shape[1] - r)
    dzdx = (heights[inner[0], 2 * r :] - heights[inner[0], : -2 * r]) / (
        2 * r * spacing
    )
    dzdy = (heights[2 * r :, inner[1]] - heights[: -2 * r, inner[1]]) / (
        2 * r * spacing
    )
    return np.sqrt(dzdx * dzdx + dzdy * dzdy).astype(np.float32)


def quantize(
    level: np.ndarray, steps: int, ix: np.ndarray, iy: np.ndarray
) -> np.ndarray:
    """Map ``level`` in [0, 1] to integers ``0..steps-1`` with ordered dithering.

    ``ix``/``iy`` are integer grid positions; dithering on world-aligned
    positions keeps the pattern continuous across chunks.
    """
    threshold = _BAYER[iy & 3, ix & 3]
    return np.clip(np.floor(level * (steps - 1) + threshold), 0, steps - 1).astype(
        np.uint8
    )
