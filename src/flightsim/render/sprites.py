"""Pixel-art sprites defined as text, so they live in version control.

Each character is one pixel; see ``_COLORS`` for the legend. ``.`` is
transparent.
"""

import pyxel

# Never used in the sprites, so it can mark transparent pixels.
TRANSPARENT = pyxel.COLOR_PINK

_COLORS = {
    ".": TRANSPARENT,
    "k": pyxel.COLOR_BLACK,
    "w": pyxel.COLOR_WHITE,
    "g": pyxel.COLOR_GRAY,
    "r": pyxel.COLOR_RED,
    "y": pyxel.COLOR_YELLOW,
    "n": pyxel.COLOR_NAVY,
}

# A small high-wing plane seen from behind and slightly above.
PLANE_REAR = [
    "...........kwk...........",
    "...........kwk...........",
    "...........krk...........",
    ".......kkkkkwkkkkk.......",
    ".......kgwwwwwwwgk.......",
    "kkkkkkkkkkkkwkkkkkkkkkkkk",
    "krwwwwwwwwwwwwwwwwwwwwwrk",
    "kkkkggggggknnnkggggggkkkk",
    "..........kwwwk..........",
    "..........kgggk..........",
    "...........kkk...........",
]


def make_image(rows: list[str]) -> pyxel.Image:
    """Pyxel image of a text sprite; blit it with ``colkey=TRANSPARENT``."""
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("sprite rows must all have the same width")
    img = pyxel.Image(width, len(rows))
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            img.pset(x, y, _COLORS[char])
    return img
