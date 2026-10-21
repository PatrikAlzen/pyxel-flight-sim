"""Time trials: a course of rings starting from an airfield, timed against a record.

A course is generated from (seed, airfield number), so the same field always
offers the same course. It leaves along the runway and then wanders: each next
ring is placed a few hundred units on, in whichever of a few candidate
directions has the lowest terrain along the way, so courses follow valleys
and coastlines rather than climbing over mountains.

A ring counts as passed when the plane's movement during a frame crosses the
ring's plane, forwards, inside its radius. That works at any speed and frame
rate, since nothing can jump over it between two frames.

Nothing here knows about Pyxel; terrain comes in as an elevation function.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from flightsim.terrain.airfields import Airfield

Elevation = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CourseSettings:
    rings: tuple[int, int] = (8, 15)  # inclusive range of ring count
    spacing: tuple[float, float] = (300.0, 600.0)
    first_ring: float = 350.0  # beyond the end of the runway
    first_height: float = 35.0  # above the runway
    clearance: tuple[float, float] = (40.0, 120.0)  # above the terrain on the way
    candidates: int = 5
    # Difficulty grows with distance from the world's origin, up to this far.
    hard_distance: float = 20000.0
    radius: tuple[float, float] = (24.0, 14.0)  # easy, hard
    max_turn: tuple[float, float] = (math.radians(45), math.radians(80))  # easy, hard
    path_samples: int = 12


@dataclass(frozen=True)
class Ring:
    centre: tuple[float, float, float]
    normal: tuple[float, float, float]  # unit vector, the direction to fly through
    radius: float

    def points(self, count: int = 24) -> np.ndarray:
        """``count`` points around the ring, as rows of world coordinates."""
        n = np.asarray(self.normal)
        side = np.cross(n, [0.0, 0.0, 1.0])
        if np.linalg.norm(side) < 1e-6:  # flying straight up or down
            side = np.array([1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        up = np.cross(side, n)
        angle = np.linspace(0, math.tau, count, endpoint=False)[:, None]
        return np.asarray(self.centre) + self.radius * (
            np.cos(angle) * side + np.sin(angle) * up
        )


@dataclass(frozen=True)
class Course:
    field_number: int
    rings: tuple[Ring, ...]


def passes(ring: Ring, start: np.ndarray, end: np.ndarray) -> bool:
    """Whether moving from ``start`` to ``end`` flies through ``ring``, forwards."""
    centre, normal = np.asarray(ring.centre), np.asarray(ring.normal)
    before = float((np.asarray(start) - centre) @ normal)
    after = float((np.asarray(end) - centre) @ normal)
    if not (before < 0.0 <= after):
        return False
    t = before / (before - after)
    crossing = np.asarray(start) + t * (np.asarray(end) - np.asarray(start))
    return float(np.linalg.norm(crossing - centre)) <= ring.radius


def build_course(
    elevation: Elevation,
    airfield: Airfield,
    seed: int,
    settings: CourseSettings | None = None,
) -> Course:
    s = settings or CourseSettings()
    rng = np.random.default_rng([seed & 0xFFFFFFFF, 11, airfield.number])
    hardness = min(1.0, math.hypot(airfield.x, airfield.y) / s.hard_distance)
    radius = s.radius[0] + (s.radius[1] - s.radius[0]) * hardness
    max_turn = s.max_turn[0] + (s.max_turn[1] - s.max_turn[0]) * hardness
    count = int(rng.integers(s.rings[0], s.rings[1] + 1))

    def ground(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(
            elevation(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        )

    # The first ring is straight off the end of the runway.
    heading = airfield.heading
    x, y = airfield.point(airfield.length / 2 + s.first_ring)
    z = max(airfield.elevation + s.first_height, float(ground(x, y)) + s.first_height)
    centres = [(x, y, z)]

    for _ in range(count - 1):
        turns = rng.uniform(-max_turn, max_turn, size=s.candidates)
        distance = rng.uniform(*s.spacing)
        headings = heading + turns
        t = np.linspace(0.0, 1.0, s.path_samples)[None, :]
        px = x + np.sin(headings)[:, None] * distance * t
        py = y - np.cos(headings)[:, None] * distance * t
        highest = ground(px, py).max(axis=1)  # highest terrain on each way
        best = int(np.argmin(highest))
        heading = float(headings[best])
        x, y = float(px[best, -1]), float(py[best, -1])
        z = float(highest[best]) + rng.uniform(*s.clearance)
        centres.append((x, y, z))

    rings = []
    previous = (*airfield.point(airfield.length / 2), airfield.elevation)
    for centre in centres:
        direction = np.subtract(centre, previous)
        direction /= np.linalg.norm(direction)
        rings.append(Ring(centre, tuple(float(v) for v in direction), radius))
        previous = centre
    return Course(airfield.number, tuple(rings))


class Phase(Enum):
    READY = auto()  # waiting at the runway for the plane to start rolling
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Race:
    """One attempt at a course, timed and compared with the record splits."""

    course: Course
    record: list[float] | None = None  # split times of the best run so far
    phase: Phase = Phase.READY
    elapsed: float = 0.0
    next_ring: int = 0
    splits: list[float] = field(default_factory=list)
    # Plane positions every ``ghost_interval`` seconds, for replaying this run.
    ghost: list[list[float]] = field(default_factory=list)
    ghost_interval: float = 1 / 3

    @property
    def remaining(self) -> int:
        return len(self.course.rings) - self.next_ring

    @property
    def time(self) -> float | None:
        """Finishing time, once finished."""
        return self.splits[-1] if self.phase is Phase.FINISHED else None

    @property
    def is_record(self) -> bool:
        return self.time is not None and (
            self.record is None or self.time < self.record[-1]
        )

    def delta(self, ring: int) -> float | None:
        """Time against the record at ring ``ring`` (negative = ahead), if both exist."""
        if self.record is None or ring >= len(self.splits) or ring >= len(self.record):
            return None
        return self.splits[ring] - self.record[ring]

    def update(
        self, dt: float, start: np.ndarray, end: np.ndarray, moving: bool
    ) -> int | None:
        """Advance by one frame in which the plane moved from ``start`` to ``end``.

        The clock starts once the plane is ``moving``. Returns the index of a
        ring passed in this frame, if any.
        """
        if self.phase is Phase.READY and moving:
            self.phase = Phase.RUNNING
        if self.phase is not Phase.RUNNING:
            return None
        self.elapsed += dt
        if len(self.ghost) <= self.elapsed / self.ghost_interval:
            self.ghost.append([round(float(v), 1) for v in end])
        ring = self.course.rings[self.next_ring]
        if not passes(ring, start, end):
            return None
        passed = self.next_ring
        self.splits.append(self.elapsed)
        self.next_ring += 1
        if self.next_ring == len(self.course.rings):
            self.phase = Phase.FINISHED
        return passed


def ghost_position(
    ghost: list[list[float]], elapsed: float, interval: float
) -> np.ndarray | None:
    """Where a recorded run was at ``elapsed`` seconds, interpolated; None after its end."""
    i = elapsed / interval
    if not ghost or i >= len(ghost) - 1:
        return None
    k = int(i)
    a, b = np.asarray(ghost[k]), np.asarray(ghost[k + 1])
    return a + (b - a) * (i - k)


def format_time(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:04.1f}"
