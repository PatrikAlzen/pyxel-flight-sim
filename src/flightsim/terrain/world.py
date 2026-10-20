"""Terrain kept in memory around the camera, ready for rendering and collision.

The world is endless: each level of detail is a fixed-size window that follows
the camera. When the camera crosses a tile border, only the row or column of
tiles that scrolled into the window is generated, by background processes, and
it overwrites the tiles that scrolled out.
"""

import math
import multiprocessing
from collections.abc import Iterator
from dataclasses import dataclass, field
from multiprocessing.connection import Connection

import numpy as np

from flightsim.terrain.generator import TerrainGenerator, TerrainSettings

TILE = 128

# Marks a ring-buffer slot that holds no tile yet.
_NO_TILE = np.iinfo(np.int64).min


@dataclass
class TerrainLevel:
    """A square window of terrain at one level of detail, stored as a torus.

    The cell at world position (x, y) lives at index
    ``[floor(y / step) % size, floor(x / step) % size]``. Lookups never need to
    know where the window starts, and moving the window only means overwriting
    the tiles that scrolled out with the ones that scrolled in.

    The window is centred on the nearest tile corner to the camera, so the
    camera is never more than half a tile from the centre.
    """

    heights: np.ndarray  # (size, size) float32, visible surface height
    colors: np.ndarray  # (size, size) uint8, palette indices
    step: float
    # World tile coordinates (tx, ty) stored in each slot, or _NO_TILE.
    slots: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        size = self.heights.shape[0]
        if self.heights.shape != (size, size) or size & (size - 1):
            raise ValueError("terrain level must be square with a power-of-two size")
        self.mask = size - 1
        self.tile = min(TILE, size)
        self.tiles = size // self.tile
        self.slots = np.full((self.tiles, self.tiles, 2), _NO_TILE, dtype=np.int64)

    @property
    def size(self) -> int:
        return self.heights.shape[0]

    @property
    def extent(self) -> float:
        """Side length of the window in world units."""
        return self.size * self.step

    @property
    def tile_extent(self) -> float:
        """Side length of one tile in world units."""
        return self.tile * self.step

    @property
    def safe_radius(self) -> float:
        """Distance from the camera that is always inside the window.

        The camera can be half a tile off-centre, and another half tile is kept
        in reserve so newly needed tiles have time to generate before they are
        within this radius.
        """
        return self.extent / 2 - self.tile_extent

    @classmethod
    def empty(cls, size: int, step: float) -> TerrainLevel:
        return cls(
            np.zeros((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.uint8),
            step,
        )

    def window(self, x: float, y: float) -> tuple[int, int]:
        """First tile (tx, ty) of the window wanted for a camera at (x, y)."""
        half = self.tiles // 2
        return (
            round(x / self.tile_extent) - half,
            round(y / self.tile_extent) - half,
        )

    def wants(self, tx: int, ty: int, x: float, y: float) -> bool:
        tx0, ty0 = self.window(x, y)
        return tx0 <= tx < tx0 + self.tiles and ty0 <= ty < ty0 + self.tiles

    def missing(self, x: float, y: float) -> list[tuple[int, int]]:
        """Tiles of the wanted window that are not stored yet."""
        tx0, ty0 = self.window(x, y)
        tx = np.arange(tx0, tx0 + self.tiles)
        ty = np.arange(ty0, ty0 + self.tiles)
        stored = self.slots[ty[:, None] % self.tiles, tx[None, :] % self.tiles]
        absent = (stored[..., 0] != tx[None, :]) | (stored[..., 1] != ty[:, None])
        rows, cols = np.nonzero(absent)
        return [(int(tx[c]), int(ty[r])) for r, c in zip(rows, cols)]

    def store(self, tx: int, ty: int, heights: np.ndarray, colors: np.ndarray) -> None:
        sy, sx = ty % self.tiles, tx % self.tiles
        rows = slice(sy * self.tile, (sy + 1) * self.tile)
        cols = slice(sx * self.tile, (sx + 1) * self.tile)
        self.heights[rows, cols] = heights
        self.colors[rows, cols] = colors
        self.slots[sy, sx] = (tx, ty)

    def generate(
        self, generator: TerrainGenerator, tx: int, ty: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Heights and colours for world tile (tx, ty)."""
        return generate_tile(generator, self.tile, self.step, tx, ty)

    def fill(self, generator: TerrainGenerator, x: float, y: float) -> None:
        """Generate every missing tile around (x, y) right away."""
        for tx, ty in self.missing(x, y):
            self.store(tx, ty, *self.generate(generator, tx, ty))

    def indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Array indices of the cells containing world points (x, y)."""
        ix = np.floor(np.asarray(x) / self.step).astype(np.int64) & self.mask
        iy = np.floor(np.asarray(y) / self.step).astype(np.int64) & self.mask
        return iy, ix

    def height_at(self, x: float, y: float) -> float:
        iy, ix = self.indices(x, y)
        return float(self.heights[iy, ix])


def generate_tile(
    generator: TerrainGenerator, tile: int, step: float, tx: int, ty: int
) -> tuple[np.ndarray, np.ndarray]:
    """Heights and colours for world tile (tx, ty) of a level with this tile size and step."""
    s = generator.sample(tx * tile * step, ty * tile * step, tile, tile, step)
    return s.height, s.color


# Jobs handed to each worker at once: one being generated, one ready to start.
_JOBS_PER_WORKER = 2

Job = tuple[int, int, int]  # (level index, tx, ty)


class TerrainWorld:
    """All levels of detail around the camera, kept up to date in the background.

    Tiles are generated in worker processes rather than threads: Pyxel's main
    loop holds the GIL while it waits for the next frame, which starves Python
    threads almost completely. Communication uses plain pipes that the main
    thread polls in ``update``, so no helper threads are needed in this process
    either. All writes to the level arrays happen in ``update``, between frames.

    Call ``update`` once per frame and ``close`` when done.
    """

    def __init__(
        self,
        generator: TerrainGenerator,
        levels: list[tuple[int, float]],
        workers: int = 2,
    ) -> None:
        self.generator = generator
        self.levels = [TerrainLevel.empty(size, step) for size, step in levels]
        self._queue: list[Job] = []  # most urgent first
        self._workers: list[tuple[multiprocessing.Process, Connection, list[Job]]] = []
        for i in range(workers):
            here, there = multiprocessing.Pipe()
            process = multiprocessing.Process(
                target=_work,
                args=(there, generator.seed, generator.settings, levels),
                name=f"terrain-{i}",
                daemon=True,
            )
            process.start()
            there.close()
            self._workers.append((process, here, []))

    @property
    def pending(self) -> int:
        """Tiles still waiting to be generated or installed."""
        return len(self._queue) + sum(len(jobs) for _, _, jobs in self._workers)

    def height_at(self, x: float, y: float) -> float:
        return self.levels[0].height_at(x, y)

    def stale_tiles(self, x: float, y: float) -> int:
        """Missing tiles that reach into a level's ``safe_radius`` around (x, y).

        These are drawn with old data from the opposite side of the ring
        buffer; anything above zero means generation is not keeping up.
        """
        count = 0
        for level in self.levels:
            for tx, ty in level.missing(x, y):
                left, top = tx * level.tile_extent, ty * level.tile_extent
                nearest_x = min(max(x, left), left + level.tile_extent)
                nearest_y = min(max(y, top), top + level.tile_extent)
                if math.hypot(nearest_x - x, nearest_y - y) < level.safe_radius:
                    count += 1
        return count

    def fill(self, x: float, y: float) -> Iterator[float]:
        """Generate everything around (x, y) in this process, yielding progress."""
        todo = [(level, tile) for level in self.levels for tile in level.missing(x, y)]
        for i, (level, (tx, ty)) in enumerate(todo):
            level.store(tx, ty, *level.generate(self.generator, tx, ty))
            yield (i + 1) / len(todo)

    def update(self, x: float, y: float) -> None:
        """Install finished tiles and hand out the ones missing around (x, y)."""
        for _, conn, jobs in self._workers:
            while conn.poll():
                li, tx, ty, heights, colors = conn.recv()
                jobs.remove((li, tx, ty))
                if self.levels[li].wants(tx, ty, x, y):
                    self.levels[li].store(tx, ty, heights, colors)

        started = {job for _, _, jobs in self._workers for job in jobs}
        queue = []
        for li, level in enumerate(self.levels):
            for tx, ty in level.missing(x, y):
                if (li, tx, ty) in started:
                    continue
                centre = (
                    (tx + 0.5) * level.tile_extent,
                    (ty + 0.5) * level.tile_extent,
                )
                queue.append((li, math.dist(centre, (x, y)), tx, ty))
        queue.sort()
        self._queue = [(li, tx, ty) for li, _, tx, ty in queue]

        for _, conn, jobs in self._workers:
            while self._queue and len(jobs) < _JOBS_PER_WORKER:
                job = self._queue.pop(0)
                conn.send(job)
                jobs.append(job)

    def close(self) -> None:
        for process, conn, _ in self._workers:
            try:
                conn.send(None)
            except OSError:
                pass
        for process, conn, _ in self._workers:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
            conn.close()
        self._workers = []


def _work(
    conn: Connection,
    seed: int,
    settings: TerrainSettings,
    levels: list[tuple[int, float]],
) -> None:
    """Worker process: generate tiles on request until told to stop.

    Also stops quietly if the game exits without calling ``close`` (Pyxel's
    quit can end the process without running Python's exit handlers), which
    shows up here as a closed pipe.
    """
    generator = TerrainGenerator(seed, settings)
    tiles = [(min(TILE, size), step) for size, step in levels]
    try:
        while (job := conn.recv()) is not None:
            li, tx, ty = job
            tile, step = tiles[li]
            conn.send((li, tx, ty, *generate_tile(generator, tile, step, tx, ty)))
    except EOFError, BrokenPipeError, KeyboardInterrupt:
        pass
