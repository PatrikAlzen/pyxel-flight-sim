"""Bridge between NumPy arrays and Pyxel images.

Pyxel stores each image as a flat buffer of palette indices (one byte per pixel,
row-major). Wrapping that buffer in a NumPy view lets us render whole frames with
vectorised array operations and skip per-pixel ``pset`` calls entirely.
"""

import numpy as np
import pyxel


def image_array(img: pyxel.Image) -> np.ndarray:
    """Return a writable ``(height, width)`` uint8 view of ``img``'s pixels.

    Writes to the array show up in the image immediately. The view does not keep
    ``img`` alive, so hold a reference to the image for as long as the view is
    used. Fetch a fresh view each frame instead of caching it, in case Pyxel
    reallocates the buffer.
    """
    return np.ctypeslib.as_array(img.data_ptr()).reshape(img.height, img.width)


def screen_array() -> np.ndarray:
    """Return a writable ``(SCREEN_H, SCREEN_W)`` view of the Pyxel screen."""
    return image_array(pyxel.screen)
