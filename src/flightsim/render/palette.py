"""Game palette: Pyxel's 16 default colours followed by shaded terrain ramps.

Indices 0-15 keep the default Pyxel colours so ``pyxel.COLOR_*`` still works for
text and HUD. Every biome then gets ``SHADES`` consecutive indices running from
shadow to full sunlight, so a terrain pixel's colour index is
``biome_color(biome, shade)``.
"""

import numpy as np
import pyxel

from flightsim.terrain.biomes import Biome

SHADES = 6
TERRAIN_BASE = 16

# (shadow, sunlit) RGB per biome; intermediate shades are interpolated.
RAMPS: dict[Biome, tuple[int, int]] = {
    Biome.DEEP_WATER: (0x14244E, 0x23407A),
    Biome.WATER: (0x2A5A96, 0x4A8ACA),
    Biome.SAND: (0x8F7A4E, 0xF0DCA0),
    Biome.GRASS: (0x2C501C, 0x8CC04A),
    Biome.FOREST: (0x14301A, 0x4A863A),
    Biome.STEPPE: (0x5E5A28, 0xC8BE6E),
    Biome.DESERT: (0x8A5C32, 0xEEBE7E),
    Biome.TAIGA: (0x163428, 0x3C6E50),
    Biome.ROCK: (0x34302C, 0x9C948A),
    Biome.SNOW: (0x7E8EAA, 0xFFFFFF),
}


def biome_color(biome: np.ndarray | int, shade: np.ndarray | int) -> np.ndarray:
    """Palette index for a biome at a shade level (0 = darkest, SHADES - 1 = brightest)."""
    return (TERRAIN_BASE + np.asarray(biome) * SHADES + np.asarray(shade)).astype(
        np.uint8
    )


def _lerp_rgb(a: int, b: int, t: float) -> int:
    channels = [
        round(((a >> s) & 0xFF) + (((b >> s) & 0xFF) - ((a >> s) & 0xFF)) * t)
        for s in (16, 8, 0)
    ]
    return (channels[0] << 16) | (channels[1] << 8) | channels[2]


def colors() -> list[int]:
    """The full palette as 24-bit RGB values."""
    result = list(pyxel.DEFAULT_COLORS)
    for biome in Biome:
        dark, light = RAMPS[biome]
        result += [_lerp_rgb(dark, light, i / (SHADES - 1)) for i in range(SHADES)]
    return result


def install() -> None:
    """Replace Pyxel's palette with the game palette. Call after ``pyxel.init``."""
    pyxel.colors[:] = colors()
