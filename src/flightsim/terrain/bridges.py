"""Bridges across the rivers, placed the same way for every seed.

Like villages and lighthouses, the world is divided into cells and each cell
gets its own random numbers from (seed, cell). A cell looks for spots near a
river (preferring ones close to a village), steps onto the river's centre
line, turns straight across it and walks out along the ground on both sides:
the deck goes at the lowest height that leaves room to fly under and whose
ends both rest on the valley sides within a sensible span. Where the ground
under the deck drops away, pillars hold it up, except over the river itself,
which stays open.
"""

import math
from dataclasses import dataclass

import numpy as np

from flightsim.terrain.settlements import make_name

PILLAR_RADIUS = 1.5


@dataclass(frozen=True)
class BridgeSettings:
    cell: float = 1500.0
    chance: float = 0.9
    grid: int = 12  # candidate points per side of a cell
    candidates: int = 8  # river spots tried per cell
    clearance: tuple[float, float] = (14.0, 25.0)  # deck underside above the water
    span: tuple[float, float] = (30.0, 140.0)
    width: float = 6.0
    thickness: float = 2.0
    railing: float = 1.2
    embed: float = 3.0  # the deck runs this far into the bank at each end
    pillar_spacing: float = 26.0
    open_middle: float = 16.0  # no pillars this close to the river's centre
    village_reach: float = 900.0  # a village this close names the bridge
    airfield_clearance: float = 400.0


@dataclass(frozen=True)
class Bridge:
    number: int
    name: str
    x: float  # middle of the deck
    y: float
    ax: float  # unit vector along the deck (across the river)
    ay: float
    half_span: float
    deck: float  # height of the deck's top surface
    thickness: float
    width: float
    railing: float
    pillars: tuple[tuple[float, float, float], ...]  # (x, y, ground)

    @property
    def bottom(self) -> float:
        return self.deck - self.thickness

    def local(self, x, y):
        """Distance along the deck from its middle, and across it (along the river)."""
        dx, dy = np.asarray(x) - self.x, np.asarray(y) - self.y
        return dx * self.ax + dy * self.ay, -dx * self.ay + dy * self.ax

    def hit(self, x: float, y: float, z: float, margin: float = 1.0) -> bool:
        """Whether a point (the plane) is inside the deck, its railings or a pillar."""
        along, across = self.local(x, y)
        if (
            abs(float(along)) <= self.half_span
            and abs(float(across)) <= self.width / 2 + margin
            and self.bottom - margin <= z <= self.deck + self.railing + margin
        ):
            return True
        return any(
            math.hypot(x - px, y - py) < PILLAR_RADIUS + margin and z < self.bottom
            for px, py, _ in self.pillars
        )

    def passed_under(self, start, end) -> bool:
        """Whether moving from ``start`` to ``end`` (x, y, z) went under the deck."""
        a0, c0 = self.local(start[0], start[1])
        a1, c1 = self.local(end[0], end[1])
        if c0 * c1 >= 0:
            return False  # did not cross the line the deck runs along
        t = c0 / (c0 - c1)
        along = a0 + (a1 - a0) * t
        z = start[2] + (end[2] - start[2]) * t
        return bool(abs(along) < self.half_span - self.width and z < self.bottom)


class Bridges:
    def __init__(
        self, seed: int, generator, settings: BridgeSettings | None = None
    ) -> None:
        self.seed = seed
        self.generator = generator  # a TerrainGenerator: rivers, heights, villages
        self.settings = settings or BridgeSettings()
        self._cells: dict[tuple[int, int], Bridge | None] = {}

    def overlapping(self, x0: float, y0: float, x1: float, y1: float) -> list[Bridge]:
        """Bridges whose cells overlap the rectangle (a deck reaches ~75 outside)."""
        cell = self.settings.cell
        found = []
        for cy in range(math.floor(y0 / cell), math.floor(y1 / cell) + 1):
            for cx in range(math.floor(x0 / cell), math.floor(x1 / cell) + 1):
                if (cx, cy) not in self._cells:
                    self._cells[cx, cy] = self._place(cx, cy)
                if self._cells[cx, cy] is not None:
                    found.append(self._cells[cx, cy])
        return found

    def near(self, x: float, y: float, radius: float) -> list[Bridge]:
        reach = radius + self.settings.span[1]
        found = self.overlapping(x - reach, y - reach, x + reach, y + reach)
        return [b for b in found if math.hypot(b.x - x, b.y - y) <= reach]

    def _place(self, cx: int, cy: int) -> Bridge | None:
        s = self.settings
        g = self.generator
        sea = g.settings.sea_level
        rng = np.random.default_rng(
            [self.seed & 0xFFFFFFFF, 37, cx + 2**31, cy + 2**31]
        )
        if rng.random() > s.chance:
            return None
        # Points in a jittered grid over the cell; keep those deep in a valley.
        k = s.grid
        u = (np.arange(k)[None, :] + rng.uniform(0.2, 0.8, (k, k))) / k
        v = (np.arange(k)[:, None] + rng.uniform(0.2, 0.8, (k, k))) / k
        xs, ys = ((cx + u) * s.cell).ravel(), ((cy + v) * s.cell).ravel()
        _, valley = g._natural(xs, ys)
        spots = np.flatnonzero(valley > 0.6)
        if not len(spots):
            return None
        x0, y0 = cx * s.cell, cy * s.cell
        reach = s.village_reach
        villages = g.settlements.villages(
            x0 - reach, y0 - reach, x0 + s.cell + reach, y0 + s.cell + reach
        )
        if villages:
            vx = np.array([vil.x for vil in villages])
            vy = np.array([vil.y for vil in villages])
            closest = np.hypot(xs[spots, None] - vx, ys[spots, None] - vy).min(axis=1)
        else:
            closest = np.full(len(spots), np.inf)
        order = spots[np.lexsort((rng.random(len(spots)), closest))]
        for spot in order[: s.candidates]:
            bridge = self._try(cx, cy, xs[spot], ys[spot], villages, rng, sea)
            if bridge is not None:
                return bridge
        return None

    def _try(self, cx, cy, x, y, villages, rng, sea) -> Bridge | None:
        s = self.settings
        g = self.generator
        px, py, ax, ay = g.river_crossing(x, y)
        inside = (
            cx * s.cell <= px < (cx + 1) * s.cell
            and cy * s.cell <= py < (cy + 1) * s.cell
        )
        if not inside or float(g.elevation(np.array([px]), np.array([py]))[0]) >= sea:
            return None  # belongs to another cell, or not on water after all
        clear = s.airfield_clearance
        if g.airfields.overlapping(px - clear, py - clear, px + clear, py + clear):
            return None
        half = s.span[1] / 2 + s.embed
        t = np.arange(-half, half + 1.0, 1.0)
        ground = g.elevation(px + t * ax, py + t * ay)
        for clearance in np.arange(s.clearance[0], s.clearance[1] + 0.5, 1.0):
            bottom = sea + clearance
            deck = bottom + s.thickness
            solid = ground >= deck
            right = t[(t > 0) & solid]
            left = t[(t < 0) & solid]
            if not len(right) or not len(left):
                continue
            a, b = left.max(), right.min()
            span = b - a
            if span < s.span[0]:
                continue  # too short: a higher deck reaches further up the banks
            if span > s.span[1]:
                break  # higher only gets longer
            return self._build(
                cx, cy, px, py, ax, ay, a, b, deck, t, ground, villages, rng
            )
        return None

    def _build(self, cx, cy, px, py, ax, ay, a, b, deck, t, ground, villages, rng):
        s = self.settings
        sea = self.generator.settings.sea_level
        middle = (a + b) / 2
        bx, by = px + middle * ax, py + middle * ay
        bottom = deck - s.thickness
        pillars = []
        for p in np.arange(
            a + s.pillar_spacing, b - s.pillar_spacing / 2, s.pillar_spacing
        ):
            under = float(np.interp(p, t, ground))
            if abs(p) > s.open_middle and under < bottom - 4:
                pillars.append((px + p * ax, py + p * ay, max(under, sea)))
        near = [
            vil for vil in villages
            if math.hypot(vil.x - bx, vil.y - by) <= s.village_reach
        ]  # fmt: skip
        if near:
            village = min(near, key=lambda vil: math.hypot(vil.x - bx, vil.y - by))
            name = f"{village.name} BRIDGE"
        else:
            name = make_name(rng) + " BRIDGE"
        return Bridge(
            number=(cx + 2**15) * 2**16 + (cy + 2**15),
            name=name,
            x=float(bx),
            y=float(by),
            ax=float(ax),
            ay=float(ay),
            half_span=float((b - a) / 2 + s.embed),
            deck=float(deck),
            thickness=s.thickness,
            width=s.width,
            railing=s.railing,
            pillars=tuple((float(x), float(y), float(z)) for x, y, z in pillars),
        )
