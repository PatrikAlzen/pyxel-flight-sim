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
    "e": pyxel.COLOR_GREEN,
    "l": pyxel.COLOR_LIGHT_BLUE,
}

# A small high-wing plane seen from behind and slightly above. Navigation
# lights tell the views apart: red on the left wing tip, green on the right.
PLANE_REAR = [
    "...........kwk...........",
    "...........kwk...........",
    "...........krk...........",
    ".......kkkkkwkkkkk.......",
    ".......kgwwwwwwwgk.......",
    "kkkkkkkkkkkkwkkkkkkkkkkkk",
    "krwwwwwwwwwwwwwwwwwwwwwek",
    "kkkkggggggknnnkggggggkkkk",
    "..........kwwwk..........",
    "..........kgggk..........",
    "...........kkk...........",
]


# The same plane coming towards you: propeller disc, windscreen and wheels,
# the fin peeking over the wing, and the lights swapped round.
PLANE_FRONT = [
    "...........kwk...........",
    "...........kwk...........",
    "kkkkkkkkkkkkkkkkkkkkkkkkk",
    "kewwwwwwwwwwwwwwwwwwwwwrk",
    "kkkkggggggkkkkkggggggkkkk",
    ".........knnnnnk.........",
    ".........kwwwwwk.........",
    "........kyyykyyyk........",
    ".......kyyyyyyyyyk.......",
    "........kyyykyyyk........",
    "......k...kwwwk...k......",
    ".....kkk...kkk...kkk.....",
]

# The same plane from straight above, nose up: wings, cabin roof, tailplane.
PLANE_TOP = [
    "............y............",
    "...........kgk...........",
    "...........kwk...........",
    "kkkkkkkkkkkkkkkkkkkkkkkkk",
    "rwwwwwwwwwwnnnwwwwwwwwwwe",
    "kkkkkkkkkkkwwwkkkkkkkkkkk",
    "...........kwk...........",
    "...........kwk...........",
    "...........kwk...........",
    "........kkkkwkkkk........",
    "........kwwwwwwwk........",
    "........kkkkrkkkk........",
]

# The same plane from its left side, flying to the right: tail fin, high wing
# seen edge on, cabin windows, propeller and wheels.
PLANE_SIDE = [
    "kkk......................",
    "kwwk.....................",
    "krwwk...kkkkkkkkkkkkk....",
    ".kwwwk..kgggggggggggk....",
    ".kwwwwkkkkkkkkkkkkkkkkk..",
    ".kwwwwwwwwwwwwwnnnwnnwwkk",
    "..kkwwwwwwwwwwwnnnwnnwwky",
    "....kkkgggggggggggggggkky",
    "..........k......k.....k.",
    ".........kkk....kkk......",
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
    ".kggggggggggggggggggggk.",
    ".kggggggggggggggggggggk.",
    ".kggggggggggggggggggggk.",
    ".kggggggggggggggggggggk.",
    ".kggggggggggggggggggggk.",
    ".kggggggggggggggggggggk.",
    ".kkkkkkkkkkkkkkkkkkkkkk.",
]

# A control tower: dark glass cab with a flat roof on a white shaft.
TOWER = [
    "kkkkkkkkk",
    "knlnnnlnk",
    "knnnnnnnk",
    ".kgggggk.",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
    "..kwwgk..",
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


def _rows(rows: list[str], width: int = 25) -> list[str]:
    """Sprite rows padded with transparency to ``width``, centred."""
    return [row.center(width, ".") for row in rows]


# A helicopter, the same five views as the plane: rotor blades as a dark line.
HELI_REAR = _rows([
    "kkkkkkkkkkkkkkkkkkkkkkkkk",
    "k",
    "kkkkk",
    "kwwwwwk",
    "kwwwwwwwk",
    "kwwgggwwk",
    "kwwwwwk",
    "kkkkk",
    "k",
    "k.....k.....k",
    "kkkkkkkkkkkkkkk",
])  # fmt: skip
HELI_FRONT = _rows([
    "kkkkkkkkkkkkkkkkkkkkkkkkk",
    "k",
    "kkkkk",
    "kwnnnwk",
    "kwnnnnnwk",
    "kwwwwwwwk",
    "kwwwwwk",
    "kkkkk",
    "k.....k.....k",
    "kkkkkkkkkkkkkkk",
])  # fmt: skip
HELI_SIDE = [
    row.ljust(25, ".")
    for row in [
        "..kkkkkkkkkkkkkkkkkkkkkkk",
        "...............k",
        "k............kkwwk",
        "kwk.......kkkwwwnnk",
        "kwwkkkkkkkwwwwwwnnnk",
        "kgwwwwwwwwwwwwwwwnnnk",
        ".kk.....kgwwwwwwwwwwk",
        ".........kkkkkkkkkkk",
        "..........k......k",
        ".........kkkkkkkkkkkk",
    ]
]  # fmt: skip
HELI_TOP = _rows([
    "k",
    "k",
    "kkkkk",
    "kwwnwwk",
    "kkkkkkkkkkwwwwwkkkkkkkkkk",
    "kwwwwwk",
    "kwwwk",
    "k",
    "kwk",
    "kwk",
    "kwk",
    "kkgkk",
    "kkk",
])  # fmt: skip
HELI_WRECK = [
    row.ljust(25, ".")
    for row in [
        "..........k",
        ".........k",
        "........kkwwwk",
        "...kkkkkwwwwnnk",
        "kkkwwwwwwwwwwnnk",
        ".kk..kgwwwwwwwwk",
        "......kkkkkkkkk.....k",
        "....kkkkkkkk..k.k",
    ]
]  # fmt: skip


def helicopter_sprites(body: str = "w", trim: str = "g"):
    """A helicopter's views (rear, front, side, top and wreck) in ``body`` and
    ``trim`` colours in place of white and grey."""
    views = {
        "rear": HELI_REAR,
        "front": HELI_FRONT,
        "side": HELI_SIDE,
        "top": HELI_TOP,
        "wreck": HELI_WRECK,
    }
    colours = str.maketrans({"w": body, "g": trim})
    return {
        view: [row.translate(colours) for row in rows] for view, rows in views.items()
    }


# Floats replacing the wheels, row by row, for a seaplane's views.
_FLOATS = {
    "rear": {
        9: ".....k....kgggk....k.....",
        10: "....kwk....kkk....kwk....",
    },
    "front": {
        10: "......k...kwwwk...k......",
        11: ".....kwk.........kwk.....",
    },
    "side": {
        8: "........k.......k........",
        9: "...kwwwwwwwwwwwwwwwwwk...",
    },
    "top": {
        6: ".......k...kwk...k.......",
        7: ".......k...kwk...k.......",
        8: ".......k...kwk...k.......",
    },
}


def plane_sprites(body: str = "w", trim: str = "g", floats: bool = False):
    """The plane's views (rear, front, side, top and wreck) as sprite rows,
    with ``body`` and ``trim`` colours in place of white and grey."""
    views = {
        "rear": PLANE_REAR,
        "front": PLANE_FRONT,
        "side": PLANE_SIDE,
        "top": PLANE_TOP,
        "wreck": PLANE_WRECK,
    }
    result = {}
    for view, rows in views.items():
        rows = list(rows)
        if floats:
            for i, row in _FLOATS.get(view, {}).items():
                rows[i] = row
        colours = str.maketrans({"w": body, "g": trim})
        result[view] = [row.translate(colours) for row in rows]
    return result


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
    "..Rr..",
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
