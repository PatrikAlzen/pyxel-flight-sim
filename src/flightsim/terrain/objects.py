"""Things standing on the terrain: trees, houses and lighthouses.

Objects are placed when a tile of the finest terrain level is generated, in
the same worker process, and travel with the tile. Trees use a jittered grid:
one candidate per small grid cell, moved a random amount within its cell and
kept with a chance that depends on the biome there. The random numbers come
from a hash of (seed, cell), so every tree belongs to exactly one tile and is
the same in every process. Houses and lighthouses come from ``Settlements``.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from flightsim.terrain.biomes import Biome


class Kind(IntEnum):
    TREE = 0
    HOUSE = 1
    LIGHTHOUSE = 2


class Tree(IntEnum):
    BROADLEAF = 0
    SPRUCE = 1
    SNOWY_SPRUCE = 2


# Size of each kind in world units (trees are scaled per tree on top of this).
HEIGHT = {Kind.TREE: 10.0, Kind.HOUSE: 7.0, Kind.LIGHTHOUSE: 22.0}
RADIUS = {Kind.TREE: 1.6, Kind.HOUSE: 5.0, Kind.LIGHTHOUSE: 3.0}


@dataclass(frozen=True)
class ObjectSettings:
    tree_cell: float = 8.0
    # Chance of a tree in a grid cell, by biome.
    tree_chance: tuple[tuple[Biome, float], ...] = (
        (Biome.FOREST, 0.75),
        (Biome.TAIGA, 0.7),
        (Biome.GRASS, 0.06),
        (Biome.STEPPE, 0.02),
    )
    tree_scale: tuple[float, float] = (0.8, 1.25)
    snowy_above: float = 95.0  # spruces higher than this carry snow
    # Clear ground around runways, beyond the runway's own length and width.
    runway_clearance: tuple[float, float] = (40.0, 70.0)


@dataclass
class Objects:
    position: np.ndarray  # (n, 3) float32: x, y and the ground they stand on
    kind: np.ndarray  # (n,) uint8, Kind
    variant: np.ndarray  # (n,) uint8, e.g. Tree, or a house's roof colour
    scale: np.ndarray  # (n,) float32

    def __len__(self) -> int:
        return len(self.kind)

    @classmethod
    def empty(cls) -> Objects:
        return cls(
            np.zeros((0, 3), np.float32),
            np.zeros(0, np.uint8),
            np.zeros(0, np.uint8),
            np.zeros(0, np.float32),
        )

    @classmethod
    def concat(cls, parts: list[Objects]) -> Objects:
        if not parts:
            return cls.empty()
        return cls(
            np.concatenate([p.position for p in parts]),
            np.concatenate([p.kind for p in parts]),
            np.concatenate([p.variant for p in parts]),
            np.concatenate([p.scale for p in parts]),
        )

    def select(self, mask: np.ndarray) -> Objects:
        return Objects(
            self.position[mask], self.kind[mask], self.variant[mask], self.scale[mask]
        )

    @property
    def height(self) -> np.ndarray:
        return (
            np.array([HEIGHT[Kind(k)] for k in range(len(Kind))])[self.kind]
            * self.scale
        )

    @property
    def radius(self) -> np.ndarray:
        return (
            np.array([RADIUS[Kind(k)] for k in range(len(Kind))])[self.kind]
            * self.scale
        )


def hit(objects: Objects, x: float, y: float, bottom: float) -> Kind | None:
    """What a plane whose lowest point is at height ``bottom`` above (x, y) flies into."""
    if not len(objects):
        return None
    offset = objects.position[:, :2] - np.array([x, y], dtype=np.float32)
    inside = np.hypot(offset[:, 0], offset[:, 1]) < objects.radius
    low = bottom < objects.position[:, 2] + objects.height
    hits = np.flatnonzero(inside & low)
    return Kind(int(objects.kind[hits[0]])) if len(hits) else None


def hash01(seed: int, salt: int, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random numbers in [0, 1) for integer grid cells.

    A splitmix64-style mix of (seed, salt, ix, iy): the same inputs always give
    the same number, in any process and in any order.
    """
    x = np.asarray(ix, np.int64).view(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    x ^= np.asarray(iy, np.int64).view(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
    x ^= np.uint64((seed * 1_000_003 + salt * 7_919) & 0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * 2.0**-53


def place_objects(generator, sample, settings: ObjectSettings | None = None) -> Objects:
    """Everything standing inside a generated terrain sample."""
    s = settings or ObjectSettings()
    width = sample.height.shape[1] * sample.step
    depth = sample.height.shape[0] * sample.step
    x0, y0 = sample.x0, sample.y0
    trees = _trees(generator, sample, s, x0, y0, width, depth)
    built = _settlement_objects(generator, x0, y0, x0 + width, y0 + depth)
    return Objects.concat([trees, built])


def _trees(generator, sample, s: ObjectSettings, x0, y0, width, depth) -> Objects:
    cell = s.tree_cell
    cx = np.arange(np.ceil(x0 / cell), np.ceil((x0 + width) / cell)).astype(np.int64)
    cy = np.arange(np.ceil(y0 / cell), np.ceil((y0 + depth) / cell)).astype(np.int64)
    gx, gy = np.meshgrid(cx, cy)
    seed = generator.seed
    px = (gx + hash01(seed, 1, gx, gy)) * cell
    py = (gy + hash01(seed, 2, gx, gy)) * cell
    inside = (px < x0 + width) & (py < y0 + depth)
    col = np.clip(
        ((px - x0) / sample.step).astype(np.int64), 0, sample.height.shape[1] - 1
    )
    row = np.clip(
        ((py - y0) / sample.step).astype(np.int64), 0, sample.height.shape[0] - 1
    )
    biome = sample.biome[row, col]
    chance = np.zeros(len(Biome))
    for b, p in s.tree_chance:
        chance[b] = p
    keep = inside & (hash01(seed, 3, gx, gy) < chance[biome])

    du, dv = s.runway_clearance
    for field in generator.airfields.overlapping(x0, y0, x0 + width, y0 + depth):
        u, v = field.local(px, py)
        keep &= ~(
            (np.abs(u) < field.length / 2 + du) & (np.abs(v) < field.width / 2 + dv)
        )

    px, py, gx, gy = px[keep], py[keep], gx[keep], gy[keep]
    ground = sample.height[row[keep], col[keep]]
    spruce = biome[keep] == Biome.TAIGA
    snowy = spruce & (ground > generator.settings.sea_level + s.snowy_above)
    variant = np.where(
        snowy, Tree.SNOWY_SPRUCE, np.where(spruce, Tree.SPRUCE, Tree.BROADLEAF)
    )
    low, high = s.tree_scale
    scale = low + (high - low) * hash01(seed, 4, gx, gy)
    return Objects(
        np.stack([px, py, ground], axis=1).astype(np.float32),
        np.full(len(px), Kind.TREE, np.uint8),
        variant.astype(np.uint8),
        scale.astype(np.float32),
    )


def _settlement_objects(generator, x0, y0, x1, y1) -> Objects:
    """Houses and lighthouses whose position falls inside [x0, x1) x [y0, y1)."""
    rows = []
    settlements = generator.settlements
    reach = settlements.settings.house_spacing * settlements.settings.village_houses[1]
    for village in settlements.villages(x0 - reach, y0 - reach, x1 + reach, y1 + reach):
        for hx, hy, ground, variant in village.houses:
            if x0 <= hx < x1 and y0 <= hy < y1:
                rows.append((hx, hy, ground, Kind.HOUSE, variant))
    for light in settlements.lighthouses(x0, y0, x1, y1):
        if x0 <= light.x < x1 and y0 <= light.y < y1:
            rows.append((light.x, light.y, light.z, Kind.LIGHTHOUSE, 0))
    if not rows:
        return Objects.empty()
    data = np.array(rows, dtype=np.float64)
    return Objects(
        data[:, :3].astype(np.float32),
        data[:, 3].astype(np.uint8),
        data[:, 4].astype(np.uint8),
        np.ones(len(rows), np.float32),
    )
