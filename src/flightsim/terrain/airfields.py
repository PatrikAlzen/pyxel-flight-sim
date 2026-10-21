"""Airfields: flat runways placed across the world, the same for every seed.

The world is divided into large cells. Each cell gets its own random numbers
from (seed, cell), so what it contains never depends on the order cells are
looked at, or on which process looks. A cell tries a few spots and runway
directions on the natural terrain and keeps the flattest one, if any is flat
enough. The runway, its flat surroundings and the blend back into the natural
terrain all fit inside the cell, so a piece of terrain only ever needs the
airfields of the cells it overlaps.

Everything here is pointwise: flattening and runway markings depend only on
world position, so terrain tiles stay seamless.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flightsim.terrain.noise import smoothstep

Elevation = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class AirfieldSettings:
    cell: float = 2048.0
    chance: float = 0.85  # share of cells that try at all, before flatness decides
    length: float = 300.0
    width: float = 16.0
    apron: float = 14.0  # flat ground beyond the runway edge on every side
    blend: float = 70.0  # distance over which flat ground eases back into nature
    max_relief: float = 12.0  # highest minus lowest natural height over the flat area
    min_height: float = 3.0  # above sea level
    max_height: float = 80.0  # lowland only
    centres: int = 8  # candidate spots per cell
    headings: int = 6  # candidate runway directions per spot


@dataclass(frozen=True)
class Airfield:
    number: int
    x: float
    y: float
    heading: float  # direction of the runway, radians clockwise from north
    length: float
    width: float
    elevation: float

    @property
    def along(self) -> tuple[float, float]:
        return math.sin(self.heading), -math.cos(self.heading)

    @property
    def across(self) -> tuple[float, float]:
        return math.cos(self.heading), math.sin(self.heading)

    def local(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Runway coordinates: u along the runway, v across it, from its centre."""
        dx, dy = np.asarray(x) - self.x, np.asarray(y) - self.y
        (ax, ay), (cx, cy) = self.along, self.across
        return dx * ax + dy * ay, dx * cx + dy * cy

    def point(self, u: float, v: float = 0.0) -> tuple[float, float]:
        """World position of runway coordinates (u, v)."""
        (ax, ay), (cx, cy) = self.along, self.across
        return self.x + u * ax + v * cx, self.y + u * ay + v * cy

    def on_runway(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        u, v = self.local(x, y)
        return (np.abs(u) <= self.length / 2) & (np.abs(v) <= self.width / 2)

    def markings(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Painted parts of the runway: centre line, edge lines and threshold bars."""
        u, v = self.local(x, y)
        half_l, half_w = self.length / 2, self.width / 2
        au, av = np.abs(u), np.abs(v)
        centre = (av < 0.7) & ((u + half_l) % 24 < 12) & (au < half_l - 20)
        edges = av > half_w - 0.8
        bars = (au > half_l - 14) & (au < half_l - 4) & ((v + half_w) % 3 < 1.6)
        return self.on_runway(x, y) & (centre | edges | (bars & (av < half_w - 1.5)))

    def props(self) -> list[tuple[str, float, float]]:
        """Buildings beside the runway: (kind, x, y)."""
        hangar = self.point(0.0, self.width / 2 + 34)
        windsock = self.point(self.length / 4, -(self.width / 2 + 10))
        return [("hangar", *hangar), ("windsock", *windsock)]


class Airfields:
    def __init__(
        self,
        seed: int,
        natural: Elevation,
        sea_level: float,
        settings: AirfieldSettings | None = None,
    ) -> None:
        self.seed = seed
        self.natural = natural
        self.sea_level = sea_level
        self.settings = settings or AirfieldSettings()
        self._cells: dict[tuple[int, int], Airfield | None] = {}

    @property
    def reach(self) -> float:
        """Furthest distance from a runway's centre that its flattening touches."""
        s = self.settings
        return math.hypot(
            s.length / 2 + s.apron + s.blend, s.width / 2 + s.apron + s.blend
        )

    def in_cell(self, cx: int, cy: int) -> Airfield | None:
        key = (cx, cy)
        if key not in self._cells:
            self._cells[key] = self._place(cx, cy)
        return self._cells[key]

    def overlapping(self, x0: float, y0: float, x1: float, y1: float) -> list[Airfield]:
        """Airfields whose cells overlap the rectangle [x0, x1] x [y0, y1]."""
        cell = self.settings.cell
        found = []
        for cy in range(math.floor(y0 / cell), math.floor(y1 / cell) + 1):
            for cx in range(math.floor(x0 / cell), math.floor(x1 / cell) + 1):
                field = self.in_cell(cx, cy)
                if field is not None:
                    found.append(field)
        return found

    def nearest(self, x: float, y: float, rings: int = 3) -> Airfield | None:
        """Closest airfield within ``rings`` cells of (x, y), or None."""
        r = rings * self.settings.cell
        fields = self.overlapping(x - r, y - r, x + r, y + r)
        return min(fields, key=lambda f: math.hypot(f.x - x, f.y - y), default=None)

    def runway_at(self, x: float, y: float) -> Airfield | None:
        for field in self.overlapping(x, y, x, y):
            if field.on_runway(x, y):
                return field
        return None

    def flatten(self, x: np.ndarray, y: np.ndarray, height: np.ndarray) -> np.ndarray:
        """Natural heights with the ground under and around runways levelled."""
        x, y = np.asarray(x), np.asarray(y)
        fields = self.overlapping(x.min(), y.min(), x.max(), y.max())
        if not fields:
            return height
        s = self.settings
        for field in fields:
            u, v = field.local(x, y)
            du = np.maximum(np.abs(u) - (s.length / 2 + s.apron), 0.0)
            dv = np.maximum(np.abs(v) - (s.width / 2 + s.apron), 0.0)
            weight = 1.0 - smoothstep(0.0, s.blend, np.hypot(du, dv))
            if weight.any():
                height = height + (field.elevation - height) * weight
        return height

    def _place(self, cx: int, cy: int) -> Airfield | None:
        s = self.settings
        rng = np.random.default_rng([self.seed & 0xFFFFFFFF, 7, cx + 2**31, cy + 2**31])
        if rng.random() > s.chance:
            return None
        margin = self.reach + 8
        centres = rng.uniform(margin, s.cell - margin, size=(s.centres, 2))
        centres += (cx * s.cell, cy * s.cell)
        headings = (np.arange(s.headings) + rng.random()) * math.pi / s.headings

        # Natural heights over the whole flat area of every candidate at once.
        u = np.linspace(-(s.length / 2 + s.apron), s.length / 2 + s.apron, 13)
        v = np.linspace(-(s.width / 2 + s.apron), s.width / 2 + s.apron, 3)
        uu, vv = np.meshgrid(u, v)
        uu, vv = uu.ravel(), vv.ravel()
        h = headings[None, :, None]
        ax, ay = np.sin(h), -np.cos(h)
        px = centres[:, 0, None, None] + uu * ax + vv * np.cos(h)
        py = centres[:, 1, None, None] + uu * ay + vv * np.sin(h)
        heights = self.natural(px, py)  # (centres, headings, points)

        relief = heights.max(axis=2) - heights.min(axis=2)
        lowest = heights.min(axis=2) - self.sea_level
        mean = heights.mean(axis=2) - self.sea_level
        ok = (
            (relief <= s.max_relief) & (lowest >= s.min_height) & (mean <= s.max_height)
        )
        if not ok.any():
            return None
        i, j = np.unravel_index(np.argmin(np.where(ok, relief, np.inf)), relief.shape)
        number = 100 + (cx * 7919 + cy * 104729 + self.seed * 31) % 900
        return Airfield(
            number=int(number),
            x=float(centres[i, 0]),
            y=float(centres[i, 1]),
            heading=float(headings[j]),
            length=s.length,
            width=s.width,
            elevation=float(heights[i, j].mean()),
        )


def landing_report(
    field: Airfield, x: float, y: float, sink: float, deadstick: bool = False
) -> str:
    """One line grading a touchdown on ``field`` by sink rate and centring;
    ``deadstick`` when the engine was out."""
    _, v = field.local(x, y)
    if sink < 1.5:
        quality = "BUTTER"
    elif sink < 3.5:
        quality = "SMOOTH"
    elif sink < 6.0:
        quality = "FIRM"
    else:
        quality = "HARD"
    if deadstick:
        quality = "DEADSTICK " + quality
    return f"FIELD {field.number}: {quality} ({sink:.1f} SINK, {abs(float(v)):.0f} OFF CENTRE)"
