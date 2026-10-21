"""Biome types and the rules that pick one per terrain cell."""

from enum import IntEnum

import numpy as np


class Biome(IntEnum):
    DEEP_WATER = 0
    WATER = 1
    SAND = 2
    GRASS = 3
    FOREST = 4
    STEPPE = 5
    DESERT = 6
    TAIGA = 7
    ROCK = 8
    SNOW = 9
    RUNWAY = 10  # never chosen by classify(); painted on by airfields

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").lower()


def classify(
    ground: np.ndarray,
    moisture: np.ndarray,
    slope: np.ndarray,
    jitter: np.ndarray | float = 0.0,
    *,
    sea_level: float,
    deep_water: float,
    beach_height: float,
    tree_line: float,
    snow_line: float,
    rock_slope: float,
) -> np.ndarray:
    """Biome per cell from terrain height, moisture in [0, 1] and slope (rise/run).

    ``jitter`` is added to the height when choosing between land biomes, to
    break up straight contour lines. Water is always decided by the true
    height, so it matches the flat water surface exactly.
    """
    depth = sea_level - ground
    above = ground + jitter - sea_level
    lowland = np.select(
        [moisture < 0.25, moisture < 0.42, moisture < 0.62],
        [Biome.DESERT, Biome.STEPPE, Biome.GRASS],
        Biome.FOREST,
    )
    highland = np.where(moisture < 0.45, Biome.STEPPE, Biome.TAIGA)
    biome = np.select(
        [
            depth > deep_water,
            depth > 0,
            (above < beach_height) & (slope < 0.6),
            (above > snow_line) & (slope < rock_slope * 1.6),
            (slope > rock_slope) | (above > tree_line),
            above > tree_line * 0.6,
        ],
        [Biome.DEEP_WATER, Biome.WATER, Biome.SAND, Biome.SNOW, Biome.ROCK, highland],
        lowland,
    )
    return biome.astype(np.uint8)
