"""Pixel-art sprites defined as text, so they live in version control.

Each character is one pixel; see ``_COLORS`` for the legend. ``.`` is
transparent.
"""

import pyxel

from flightsim.render.palette import biome_color
from flightsim.terrain.biomes import Biome

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
    "o": pyxel.COLOR_ORANGE,
    "b": pyxel.COLOR_BROWN,
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


# The same plane after a crash: tail bent, right wing torn off, sooty and
# smouldering, with bits of debris beside it.
PLANE_WRECK = [
    "..............k..........",
    ".............kgk.........",
    "............kgk..........",
    ".......kkkkkkgkkk........",
    ".......kgwgkkwwgk........",
    "kkkkkkkkkkkkwkkkkk.......",
    "krwwwgwwkwwwwwgwkk..k....",
    "kkkkggkkkknbnkkk...k.k...",
    "..........kgykk..o.......",
    "..........kokbk..........",
    "...........kkk...........",
]

# An airfield hangar: curved roof, open door.
HANGAR = [
    "........kkkkkkkk........",
    ".....kkkggggggggkkk.....",
    "...kkgggggwwwwgggggkk...",
    "..kgggggggggggggggggggk.",
    ".kggggggggggggggggggggk.",
    ".kgggkkkkkkkkkkkkkkgggk.",
    ".kgggknnnnnnnnnnnnkgggk.",
    ".kgggknnnnnnnnnnnnkgggk.",
    ".kgggknnnnnnnnnnnnkgggk.",
    ".kgggknnnnnnnnnnnnkgggk.",
    ".kgggknnnnnnnnnnnnkgggk.",
    ".kkkkkkkkkkkkkkkkkkkkkk.",
]

# A windsock on its pole.
WINDSOCK = [
    "kowowo..",
    "kowow...",
    "k.......",
    "k.......",
    "k.......",
    "k.......",
    "k.......",
    "k.......",
    "k.......",
    "k.......",
]


def make_image(rows: list[str], legend: dict[str, int] | None = None) -> pyxel.Image:
    """Pyxel image of a text sprite; blit it with ``colkey=TRANSPARENT``."""
    legend = legend or _COLORS
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("sprite rows must all have the same width")
    img = pyxel.Image(width, len(rows))
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            img.pset(x, y, legend[char])
    return img


# Things in the world use the terrain's palette ramps, so they darken at dusk
# and night along with the ground. Upper case = lit side, lower = shadow.
def _shade(biome: Biome, shade: int) -> int:
    return int(biome_color(biome, shade))


WORLD_COLORS = {
    ".": TRANSPARENT,
    "L": _shade(Biome.FOREST, 4),  # leaves
    "l": _shade(Biome.FOREST, 1),
    "N": _shade(Biome.TAIGA, 4),  # needles
    "n": _shade(Biome.TAIGA, 1),
    "S": _shade(Biome.SNOW, 5),
    "s": _shade(Biome.SNOW, 2),
    "t": _shade(Biome.DESERT, 0),  # trunk
    "W": _shade(Biome.SAND, 5),  # walls
    "w": _shade(Biome.SAND, 3),
    "d": _shade(Biome.DESERT, 1),  # door
    "g": _shade(Biome.WATER, 1),  # window
    "R": _shade(Biome.ROOF, 4),  # red stripes
    "r": _shade(Biome.ROOF, 2),
    "k": _shade(Biome.ROCK, 0),
    "y": pyxel.COLOR_YELLOW,  # a lamp shines in any light
}

# Roof colours for the house variants: red tiles, dark tiles, slate, blue.
ROOF_COLORS = [
    (_shade(Biome.ROOF, 4), _shade(Biome.ROOF, 2)),
    (_shade(Biome.ROOF, 2), _shade(Biome.ROOF, 0)),
    (_shade(Biome.ROCK, 3), _shade(Biome.ROCK, 1)),
    (_shade(Biome.WATER, 4), _shade(Biome.WATER, 1)),
]

BROADLEAF = [
    "...lLl...",
    "..lLLLl..",
    ".lLLLLLl.",
    "lLLLlLLLl",
    "lLLlLLLLl",
    "llLLLLLll",
    ".llLLLll.",
    "..lllll..",
    "....t....",
    "....t....",
    "....t....",
    "...ttt...",
]

SPRUCE = [
    "...N...",
    "..NNN..",
    "..NnN..",
    ".NNnNN.",
    "..NnN..",
    ".NNNNN.",
    "NNNnNNN",
    ".NNnNn.",
    "NNNNNNn",
    "NNnnNnn",
    "...t...",
    "...t...",
    "..ttt..",
]

SNOWY_SPRUCE = [
    "...S...",
    "..SSS..",
    "..NnS..",
    ".SSnNS.",
    "..NnN..",
    ".SSSNS.",
    "NNNnSSN",
    ".SNnNn.",
    "SSSNNSS",
    "NNnnNnn",
    "...t...",
    "...t...",
    "..ttt..",
]

# Front of a house; "R"/"r" are replaced by the roof colour of each variant.
HOUSE = [
    "....RRRR....",
    "..RRRRRRRR..",
    ".RRRRRRRRrr.",
    "RRRRRRRRRrrr",
    ".WWWWWWWWww.",
    ".WWgWWWgWww.",
    ".WWgWddWWww.",
    ".WWWWddWWww.",
    ".WWWWddWWww.",
]

LIGHTHOUSE = [
    "..yy..",
    ".kyyk.",
    ".kkkk.",
    "..SS..",
    "..Rr..",
    "..Ss..",
    "..Rr..",
    ".SSss.",
    ".RRrr.",
    ".SSss.",
    ".RRrr.",
    ".SSss.",
    "SSSsss",
]


def house_images() -> list[pyxel.Image]:
    """One house image per roof colour."""
    images = []
    for lit, shadow in ROOF_COLORS:
        legend = {**WORLD_COLORS, "R": lit, "r": shadow}
        images.append(make_image(HOUSE, legend))
    return images
