"""Game palette: Pyxel's 16 default colours, shaded terrain ramps, fog and sky.

Layout of the 256 palette slots:

- ``0..15``: Pyxel's default colours, so ``pyxel.COLOR_*`` still works for
  text and HUD.
- ``TERRAIN_BASE..``: ``FOG_LEVELS`` copies of the terrain colours. Copy 0 is
  clear; each later copy is blended further towards ``FOG_COLOR``. Within a
  copy every biome has ``SHADES`` consecutive indices from shadow to full sun,
  so a terrain colour is ``biome_color(biome, shade)`` and its fogged version
  is ``color + level * TERRAIN_COLORS``.
- ``SKY_BASE..``: ``SKY_COLORS`` sky shades from the horizon (equal to the fog
  colour, so fully fogged terrain melts into the sky) up to the zenith.
"""

from dataclasses import dataclass

import numpy as np
import pyxel

from flightsim.terrain.biomes import Biome

SHADES = 6
TERRAIN_BASE = 16
TERRAIN_COLORS = len(Biome) * SHADES

# How far each fog copy is blended towards the fog colour. Fog level
# FOG_LEVELS (one past the last copy) means fully fogged: SKY_BASE.
FOG_BLEND = [0.0, 0.35, 0.68]
FOG_LEVELS = len(FOG_BLEND)
FOG_COLOR = 0xBCD6EC

SKY_BASE = TERRAIN_BASE + TERRAIN_COLORS * FOG_LEVELS
SKY_COLORS = 8
SKY_ZENITH = 0x3A64B0

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
    Biome.RUNWAY: (0x26262C, 0xE6E6DC),  # asphalt; the top shade is the paint
}


@dataclass(frozen=True)
class Lighting:
    """Time of day, applied by rebuilding the palette rather than the image."""

    name: str
    tint: tuple[float, float, float]  # multiplies terrain RGB
    fog: int  # haze colour, also the sky at the horizon
    zenith: int  # sky colour straight up


DAY = Lighting("DAY", (1.0, 1.0, 1.0), FOG_COLOR, SKY_ZENITH)
DUSK = Lighting("DUSK", (1.0, 0.72, 0.58), 0xE8A27A, 0x3A2E6A)
NIGHT = Lighting("NIGHT", (0.3, 0.36, 0.55), 0x1E2A48, 0x05070F)
LIGHTINGS = [DAY, DUSK, NIGHT]


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


def _tint(rgb: int, tint: tuple[float, float, float]) -> int:
    channels = [
        min(255, round(((rgb >> s) & 0xFF) * t)) for s, t in zip((16, 8, 0), tint)
    ]
    return (channels[0] << 16) | (channels[1] << 8) | channels[2]


def colors(lighting: Lighting = DAY) -> list[int]:
    """The full palette as 24-bit RGB values."""
    terrain = []
    for biome in Biome:
        dark, light = RAMPS[biome]
        terrain += [
            _tint(_lerp_rgb(dark, light, i / (SHADES - 1)), lighting.tint)
            for i in range(SHADES)
        ]

    result = list(pyxel.DEFAULT_COLORS)
    for blend in FOG_BLEND:
        result += [_lerp_rgb(c, lighting.fog, blend) for c in terrain]
    result += [
        _lerp_rgb(lighting.fog, lighting.zenith, i / (SKY_COLORS - 1))
        for i in range(SKY_COLORS)
    ]
    return result


def install(lighting: Lighting = DAY) -> None:
    """Replace Pyxel's palette with the game palette. Call after ``pyxel.init``.

    Everything on screen changes at once, which is what makes switching the
    time of day free.
    """
    pyxel.colors[:] = colors(lighting)
