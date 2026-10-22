"""Helicopter pads beside some villages and lighthouses.

A village or lighthouse gets a pad with a chance drawn from (seed, where it
is), so pads are the same in every process and whatever is looked at
first. A few spots around it are tried and the flattest dry one, clear of
the houses, is kept. The pad is a slab whose top sits just above the highest
ground under it, so it never sinks into a slope; helicopters land on it and
refuel there.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HelipadSettings:
    half: float = 7.0  # half the pad's side
    thickness: float = 0.4  # above the highest ground under it
    village_chance: float = 0.8
    lighthouse_chance: float = 0.7
    village_distance: tuple[float, float] = (40.0, 90.0)  # from the village middle
    lighthouse_distance: tuple[float, float] = (22.0, 60.0)
    candidates: int = 20
    max_relief: float = 4.0  # highest minus lowest ground under the pad
    house_clearance: float = 14.0


@dataclass(frozen=True)
class Helipad:
    number: int
    name: str
    x: float
    y: float
    top: float  # height of the landing surface
    heading: float  # way the painted H faces
    half: float

    def contains(self, x, y) -> np.ndarray:
        return (np.abs(np.asarray(x) - self.x) <= self.half) & (
            np.abs(np.asarray(y) - self.y) <= self.half
        )


class Helipads:
    def __init__(self, seed: int, generator, settings: HelipadSettings | None = None):
        self.seed = seed
        self.generator = generator  # a TerrainGenerator: heights and settlements
        self.settings = settings or HelipadSettings()
        self._pads: dict[tuple[str, int, int], Helipad | None] = {}

    def overlapping(self, x0: float, y0: float, x1: float, y1: float) -> list[Helipad]:
        """Pads of the villages and lighthouses in and around the rectangle."""
        s = self.settings
        reach = max(s.village_distance[1], s.lighthouse_distance[1]) + s.half
        settlements = self.generator.settlements
        box = (x0 - reach, y0 - reach, x1 + reach, y1 + reach)
        found = []
        for village in settlements.villages(*box):
            found.append(self._pad("village", village))
        for light in settlements.lighthouses(*box):
            found.append(self._pad("lighthouse", light))
        return [
            p
            for p in found
            if p is not None
            and x0 - s.half <= p.x <= x1 + s.half
            and y0 - s.half <= p.y <= y1 + s.half
        ]

    def near(self, x: float, y: float, radius: float) -> list[Helipad]:
        pads = self.overlapping(x - radius, y - radius, x + radius, y + radius)
        return [p for p in pads if math.hypot(p.x - x, p.y - y) <= radius]

    def at(self, x: float, y: float) -> Helipad | None:
        """The pad whose surface covers (x, y), if any."""
        return next(
            (p for p in self.near(x, y, self.settings.half * 1.5) if p.contains(x, y)),
            None,
        )

    def _pad(self, kind: str, place) -> Helipad | None:
        # Village and lighthouse numbers are short and can repeat; their
        # positions do not.
        key = (kind, round(place.x), round(place.y))
        if key not in self._pads:
            self._pads[key] = self._place(kind, place)
        return self._pads[key]

    def _place(self, kind: str, place) -> Helipad | None:
        s = self.settings
        g = self.generator
        salt = 61 if kind == "village" else 67
        px, py = round(place.x), round(place.y)
        rng = np.random.default_rng(
            [self.seed & 0xFFFFFFFF, salt, px + 2**31, py + 2**31]
        )
        chance = s.village_chance if kind == "village" else s.lighthouse_chance
        if rng.random() > chance:
            return None
        low, high = s.village_distance if kind == "village" else s.lighthouse_distance
        angle = rng.uniform(0, math.tau, s.candidates)
        reach = rng.uniform(low, high, s.candidates)
        xs = place.x + reach * np.cos(angle)
        ys = place.y + reach * np.sin(angle)
        # Ground under each candidate pad: its middle, corners and edge middles.
        offsets = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]) * s.half
        gx = xs[:, None] + offsets[:, 0]
        gy = ys[:, None] + offsets[:, 1]
        ground = g.elevation(gx, gy)
        relief = ground.max(axis=1) - ground.min(axis=1)
        dry = ground.min(axis=1) > g.settings.sea_level + 1.5
        ok = dry & (relief <= s.max_relief)
        houses = getattr(place, "houses", ())
        if houses:
            hx = np.array([h[0] for h in houses])
            hy = np.array([h[1] for h in houses])
            nearest = np.hypot(xs[:, None] - hx, ys[:, None] - hy).min(axis=1)
            ok &= nearest > s.house_clearance + s.half
        if not ok.any():
            return None
        best = int(np.argmin(np.where(ok, relief, np.inf)))
        name = place.name.removesuffix(" LIGHT")
        return Helipad(
            number=((px & 0xFFFF) << 16 | (py & 0xFFFF)) * 2 + (kind == "lighthouse"),
            name=f"{name} HELIPAD",
            x=float(xs[best]),
            y=float(ys[best]),
            top=float(ground[best].max()) + s.thickness,
            heading=float(rng.uniform(0, math.tau)),
            half=s.half,
        )
