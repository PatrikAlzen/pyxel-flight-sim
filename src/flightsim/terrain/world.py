"""Terrain kept in memory around the camera, ready for rendering and collision."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from flightsim.terrain.generator import TerrainGenerator

TILE = 128


@dataclass
class TerrainLevel:
    """A square window of terrain at one level of detail, stored as a torus.

    The cell at world position (x, y) lives at index
    ``[floor(y / step) % size, floor(x / step) % size]``. Lookups never need to
    know where the window starts, and moving the window only means overwriting
    the cells that scrolled out with the ones that scrolled in.
    """

    heights: np.ndarray  # (size, size) float32, visible surface height
    colors: np.ndarray  # (size, size) uint8, palette indices
    step: float

    def __post_init__(self) -> None:
        size = self.heights.shape[0]
        if self.heights.shape != (size, size) or size & (size - 1):
            raise ValueError("terrain level must be square with a power-of-two size")
        self.mask = size - 1

    @property
    def size(self) -> int:
        return self.heights.shape[0]

    @property
    def extent(self) -> float:
        """Side length of the window in world units."""
        return self.size * self.step

    @classmethod
    def empty(cls, size: int, step: float) -> TerrainLevel:
        return cls(
            np.zeros((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.uint8),
            step,
        )

    def fill_tiles(
        self, generator: TerrainGenerator, center_x: float, center_y: float
    ) -> Iterator[float]:
        """Generate the window centred on (center_x, center_y) one tile at a time.

        Yields the fraction done after each tile, so callers can spread the work
        over several frames and show progress.
        """
        tile = min(TILE, self.size)
        gx0 = int(np.floor(center_x / self.step)) - self.size // 2
        gy0 = int(np.floor(center_y / self.step)) - self.size // 2
        gx0 -= gx0 % tile
        gy0 -= gy0 % tile
        total = (self.size // tile) ** 2
        done = 0
        for gy in range(gy0, gy0 + self.size, tile):
            for gx in range(gx0, gx0 + self.size, tile):
                sample = generator.sample(
                    gx * self.step, gy * self.step, tile, tile, self.step
                )
                rows = slice(gy & self.mask, (gy & self.mask) + tile)
                cols = slice(gx & self.mask, (gx & self.mask) + tile)
                self.heights[rows, cols] = sample.height
                self.colors[rows, cols] = sample.color
                done += 1
                yield done / total

    def fill(
        self, generator: TerrainGenerator, center_x: float, center_y: float
    ) -> None:
        """Generate the whole window centred on (center_x, center_y) at once."""
        for _ in self.fill_tiles(generator, center_x, center_y):
            pass

    def indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Array indices of the cells containing world points (x, y)."""
        ix = np.floor(np.asarray(x) / self.step).astype(np.int64) & self.mask
        iy = np.floor(np.asarray(y) / self.step).astype(np.int64) & self.mask
        return iy, ix

    def height_at(self, x: float, y: float) -> float:
        iy, ix = self.indices(x, y)
        return float(self.heights[iy, ix])
