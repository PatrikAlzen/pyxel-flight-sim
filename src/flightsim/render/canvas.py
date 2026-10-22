"""Where the HUD, menus and text are drawn: the screen, or a low-resolution
layer that is scaled up over the sharp world view.

The world is drawn at the screen's full resolution, but the interface is laid
out for the terrain's resolution (a third of it) and drawn to a layer that is
then scaled up on top. These functions take Pyxel's drawing calls and send
them to whichever is current, so interface code does not need to know.
"""

import numpy as np
import pyxel

from flightsim.render.framebuffer import image_array, screen_array

_target: pyxel.Image | None = None  # None: the screen


def use(image: pyxel.Image | None) -> None:
    """Draw to ``image`` from now on, or to the screen with None."""
    global _target
    _target = image


def _draw():
    return pyxel if _target is None else _target


def width() -> int:
    return pyxel.width if _target is None else _target.width


def height() -> int:
    return pyxel.height if _target is None else _target.height


def array() -> np.ndarray:
    """The current target's pixels as a writable (height, width) array."""
    return screen_array() if _target is None else image_array(_target)


def cls(col: int) -> None:
    _draw().cls(col)


def pset(x, y, col) -> None:
    _draw().pset(x, y, col)


def line(x1, y1, x2, y2, col) -> None:
    _draw().line(x1, y1, x2, y2, col)


def rect(x, y, w, h, col) -> None:
    _draw().rect(x, y, w, h, col)


def rectb(x, y, w, h, col) -> None:
    _draw().rectb(x, y, w, h, col)


def circ(x, y, r, col) -> None:
    _draw().circ(x, y, r, col)


def circb(x, y, r, col) -> None:
    _draw().circb(x, y, r, col)


def tri(x1, y1, x2, y2, x3, y3, col) -> None:
    _draw().tri(x1, y1, x2, y2, x3, y3, col)


def trib(x1, y1, x2, y2, x3, y3, col) -> None:
    _draw().trib(x1, y1, x2, y2, x3, y3, col)


def text(x, y, s, col) -> None:
    _draw().text(x, y, s, col)


def blt(x, y, img, u, v, w, h, colkey=None, rotate=None, scale=None) -> None:
    _draw().blt(x, y, img, u, v, w, h, colkey, rotate=rotate, scale=scale)


def clip(*args) -> None:
    _draw().clip(*args)


def dither(alpha: float) -> None:
    _draw().dither(alpha)
