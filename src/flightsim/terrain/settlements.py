"""Villages and lighthouses, placed across the world the same way for every seed.

Like airfields, each kind divides the world into large cells, and each cell
gets its own random numbers from (seed, cell), so placement never depends on
which part of the world was visited first. A cell tries a few spots against
the terrain and keeps one that fits, if any:

- a village wants flat lowland that is not dry, with water within reach, and
  lines its houses up along a road;
- a lighthouse wants the tip of a cape: a low spot of land mostly surrounded
  by sea.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flightsim.terrain.airfields import Airfields

Field = Callable[[np.ndarray, np.ndarray], np.ndarray]

_SYLLABLES = [
    "NOR", "VIK", "STA", "BRO", "HOLM", "DAL", "BERG", "SUND", "LUND", "TORP",
    "HED", "MAR", "KUL", "LIN", "RO", "SKO", "FORS", "NES", "HAM", "ED",
]  # fmt: skip


@dataclass(frozen=True)
class SettlementSettings:
    village_cell: float = 640.0
    village_chance: float = 0.55
    village_candidates: int = 6
    village_houses: tuple[int, int] = (4, 12)
    house_spacing: float = 15.0
    house_offset: float = 9.0  # from the middle of the road
    village_height: tuple[float, float] = (3.0, 45.0)  # above sea level
    village_relief: float = 6.0
    village_moisture: float = 0.3
    water_within: float = 220.0
    airfield_clearance: float = 400.0

    lighthouse_cell: float = 768.0
    lighthouse_chance: float = 0.8
    lighthouse_candidates: int = 16
    lighthouse_height: tuple[float, float] = (2.0, 30.0)
    cape_radius: float = 60.0
    cape_sea_share: float = 0.55  # of points around it that must be sea


@dataclass(frozen=True)
class Village:
    number: int
    name: str
    x: float
    y: float
    houses: tuple[tuple[float, float, float, int], ...]  # (x, y, ground, variant)


@dataclass(frozen=True)
class Lighthouse:
    number: int
    name: str
    x: float
    y: float
    z: float


def make_name(rng: np.random.Generator) -> str:
    first, second = rng.choice(len(_SYLLABLES), size=2, replace=False)
    return _SYLLABLES[first] + _SYLLABLES[second]


def _ring(
    x: float, y: float, radius: float, count: int
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.linspace(0, math.tau, count, endpoint=False)
    return x + radius * np.cos(angle), y + radius * np.sin(angle)


class Settlements:
    def __init__(
        self,
        seed: int,
        elevation: Field,
        moisture: Field,
        sea_level: float,
        airfields: Airfields,
        settings: SettlementSettings | None = None,
    ) -> None:
        self.seed = seed
        self.elevation = elevation
        self.moisture = moisture
        self.sea_level = sea_level
        self.airfields = airfields
        self.settings = settings or SettlementSettings()
        self._villages: dict[tuple[int, int], Village | None] = {}
        self._lighthouses: dict[tuple[int, int], Lighthouse | None] = {}

    def _height(self, x, y) -> np.ndarray:
        return np.asarray(self.elevation(np.asarray(x, float), np.asarray(y, float)))

    def villages(self, x0: float, y0: float, x1: float, y1: float) -> list[Village]:
        """Villages whose cells overlap the rectangle (houses may reach a little outside)."""
        return self._in_cells(
            self.settings.village_cell,
            self._villages,
            self._place_village,
            x0,
            y0,
            x1,
            y1,
        )

    def lighthouses(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> list[Lighthouse]:
        return self._in_cells(
            self.settings.lighthouse_cell,
            self._lighthouses,
            self._place_lighthouse,
            x0,
            y0,
            x1,
            y1,
        )

    def _in_cells(self, cell, cache, place, x0, y0, x1, y1) -> list:
        found = []
        for cy in range(math.floor(y0 / cell), math.floor(y1 / cell) + 1):
            for cx in range(math.floor(x0 / cell), math.floor(x1 / cell) + 1):
                if (cx, cy) not in cache:
                    cache[cx, cy] = place(cx, cy)
                if cache[cx, cy] is not None:
                    found.append(cache[cx, cy])
        return found

    def _rng(self, salt: int, cx: int, cy: int) -> np.random.Generator:
        return np.random.default_rng(
            [self.seed & 0xFFFFFFFF, salt, cx + 2**31, cy + 2**31]
        )

    def _place_village(self, cx: int, cy: int) -> Village | None:
        s = self.settings
        rng = self._rng(23, cx, cy)
        if rng.random() > s.village_chance:
            return None
        n = s.village_candidates
        margin = s.house_spacing * s.village_houses[1] / 2 + 20
        lo, hi = cx * s.village_cell + margin, (cx + 1) * s.village_cell - margin
        xs = rng.uniform(lo, hi, size=n)
        lo, hi = cy * s.village_cell + margin, (cy + 1) * s.village_cell - margin
        ys = rng.uniform(lo, hi, size=n)

        # Test every candidate at once: each terrain query has a large fixed
        # cost, so a few big ones are much cheaper than many small ones.
        sea = self.sea_level
        h = self._height(xs, ys)
        angle = np.linspace(0, math.tau, 8, endpoint=False)
        around = self._height(
            xs[:, None] + 50 * np.cos(angle), ys[:, None] + 50 * np.sin(angle)
        )
        angle = np.linspace(0, math.tau, 16, endpoint=False)
        radius = np.array([0.6, 1.0])[:, None] * s.water_within
        wet = self._height(
            xs[:, None, None] + radius * np.cos(angle),
            ys[:, None, None] + radius * np.sin(angle),
        )
        ok = (h - sea >= s.village_height[0]) & (h - sea <= s.village_height[1])
        ok &= np.abs(around - h[:, None]).max(axis=1) <= s.village_relief
        ok &= np.asarray(self.moisture(xs, ys)) >= s.village_moisture
        ok &= (wet < sea).any(axis=(1, 2))

        c = s.airfield_clearance
        for i in np.flatnonzero(ok):
            x, y = float(xs[i]), float(ys[i])
            fields = self.airfields.overlapping(x - c, y - c, x + c, y + c)
            if any(math.hypot(f.x - x, f.y - y) < c for f in fields):
                continue
            houses = self._houses(rng, x, y, float(h[i]))
            if len(houses) >= s.village_houses[0]:
                number = 100 + (cx * 7907 + cy * 104723 + self.seed * 17) % 900
                return Village(int(number), make_name(rng), x, y, houses)
        return None

    def _houses(self, rng, x: float, y: float, centre_height: float) -> tuple:
        """Houses on alternating sides of a straight road through (x, y)."""
        s = self.settings
        count = int(rng.integers(s.village_houses[0], s.village_houses[1] + 1))
        road = rng.uniform(0, math.pi)
        along = np.array([math.sin(road), -math.cos(road)])
        across = np.array([math.cos(road), math.sin(road)])
        k = np.arange(count) - (count - 1) / 2
        side = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
        jitter = rng.uniform(-2.0, 2.0, size=(count, 2))
        px = (
            x
            + k * s.house_spacing * along[0]
            + side * s.house_offset * across[0]
            + jitter[:, 0]
        )
        py = (
            y
            + k * s.house_spacing * along[1]
            + side * s.house_offset * across[1]
            + jitter[:, 1]
        )
        ground = self._height(px, py)
        variants = rng.integers(0, 4, size=count)
        ok = (ground > self.sea_level + 1.5) & (np.abs(ground - centre_height) < 6.0)
        return tuple(
            (float(a), float(b), float(c), int(v))
            for a, b, c, v, keep in zip(px, py, ground, variants, ok)
            if keep
        )

    def _place_lighthouse(self, cx: int, cy: int) -> Lighthouse | None:
        s = self.settings
        rng = self._rng(29, cx, cy)
        if rng.random() > s.lighthouse_chance:
            return None
        n = s.lighthouse_candidates
        xs = (cx + rng.uniform(0.1, 0.9, size=n)) * s.lighthouse_cell
        ys = (cy + rng.uniform(0.1, 0.9, size=n)) * s.lighthouse_cell
        heights = self._height(xs, ys) - self.sea_level
        angle = np.linspace(0, math.tau, 16, endpoint=False)
        rx = xs[:, None] + s.cape_radius * np.cos(angle)
        ry = ys[:, None] + s.cape_radius * np.sin(angle)
        sea_share = (self._height(rx, ry) < self.sea_level).mean(axis=1)
        low, high = s.lighthouse_height
        ok = (heights >= low) & (heights <= high) & (sea_share >= s.cape_sea_share)
        if not ok.any():
            return None
        best = int(np.argmax(np.where(ok, sea_share, -1)))
        number = 100 + (cx * 6151 + cy * 92821 + self.seed * 13) % 900
        name = make_name(rng) + " LIGHT"
        return Lighthouse(
            int(number),
            name,
            float(xs[best]),
            float(ys[best]),
            float(heights[best] + self.sea_level),
        )
