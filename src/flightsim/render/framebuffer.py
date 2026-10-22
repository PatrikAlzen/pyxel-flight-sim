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


def scale_onto_screen(pixels: np.ndarray, transparent: int | None = None) -> None:
    """Scale ``pixels`` up to fill the screen, each pixel a square block,
    leaving the screen as it was where they are ``transparent``. Much faster
    than a scaled ``pyxel.blt`` of the whole screen."""
    screen = screen_array()
    scale = screen.shape[0] // pixels.shape[0]
    big = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
    if transparent is None:
        screen[:] = big
    else:
        np.copyto(screen, big, where=big != transparent)
