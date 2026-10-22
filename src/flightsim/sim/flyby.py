"""Close passes by things that stand still, such as a control tower.

A pass counts once the plane has come close and is drawing away again, the
same way as a near miss with traffic: measured between frames, so a fast
plane cannot skip past. After counting, the same thing only counts again
once the plane has been well away from it.
"""

import math
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from flightsim.sim.traffic import closest_approach


@dataclass(frozen=True)
class FlybySettings:
    distance: float = 30.0  # passing closer than this...
    speed: float = 60.0  # ...at least this fast counts
    reset: float = 150.0  # further away than this: the next pass counts again


class Flybys:
    def __init__(self, settings: FlybySettings | None = None) -> None:
        self.settings = settings or FlybySettings()
        self.closest: dict[Hashable, float] = {}
        self.counted: set[Hashable] = set()

    def clear(self) -> None:
        self.closest.clear()
        self.counted.clear()

    def update(
        self,
        key: Hashable,
        start: np.ndarray,
        end: np.ndarray,
        target: np.ndarray,
        speed: float,
    ) -> bool:
        """Whether the plane, moving from ``start`` to ``end`` at ``speed``, has
        just made a close pass by ``target`` (known by ``key``)."""
        s = self.settings
        target = np.asarray(target, float)
        gap = closest_approach(np.asarray(end, float), target, start, target)
        if gap > s.reset:
            self.closest.pop(key, None)
            self.counted.discard(key)
            return False
        closest = min(self.closest.get(key, math.inf), gap)
        self.closest[key] = closest
        drawing_away = gap > closest + 1.0
        if key in self.counted or not drawing_away:
            return False
        if closest < s.distance and speed >= s.speed:
            self.counted.add(key)
            return True
        return False
