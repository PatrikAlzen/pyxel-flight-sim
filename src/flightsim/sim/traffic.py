"""Other aircraft flying between airfields, on a timetable.

Nothing here is simulated. Airfields close enough to each other may have a
route between them, decided by the pair and the seed alone. A route is a
fixed path (take-off roll, climb, cruise above the terrain, approach,
landing roll) with a time for every point on it, and flights leave on a
timetable: every ``period`` seconds from a seeded phase, both ways. So where
any flight is at any game time is a lookup, the same whenever and wherever
it is asked for, and only flights near the player are ever looked up.
"""

import math
from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class TrafficSettings:
    pair_range: tuple[float, float] = (1500.0, 4500.0)  # airfield distance
    pair_chance: float = 0.75  # share of pairs in range that have a route
    period: tuple[float, float] = (100.0, 220.0)  # seconds between departures
    search: float = 6000.0  # airfields this close to the player may send traffic
    cruise_margin: float = 150.0  # above the terrain, well away from the runways
    min_clearance: float = 30.0  # anywhere in the air
    slope: float = 0.2  # steepest climb or descent, height per distance
    climb_out: float = 500.0  # straight ahead after lifting off
    final_length: float = 900.0  # straight in to the runway
    runway_reach: float = 300.0  # no terrain margin is needed this close
    turn_radius: float = 600.0
    takeoff_roll: float = 250.0
    landing_roll: float = 200.0
    roll_speed: float = 30.0  # average on the ground
    cruise_speed: float = 90.0
    final_speed: float = 65.0
    gear_height: float = 1.5
    draw_radius: float = 3000.0
    max_planes: int = 8
    builds_per_frame: int = 1  # routes laid out per frame (~5 ms each)
    step: float = 40.0  # world units between the points of a path


@dataclass(frozen=True)
class Flight:
    """One aircraft right now."""

    ident: tuple  # stays the same for the whole flight
    kind: int  # index into planes.TYPES
    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    on_ground: bool


class Route:
    """A path from one runway to another with a time for every point."""

    def __init__(
        self, points: np.ndarray, speeds: np.ndarray, ground: np.ndarray
    ) -> None:
        self.points = points  # (n, 3)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.times = np.concatenate([[0.0], np.cumsum(lengths / speeds)])
        self.ground = ground  # (n - 1,) whether each segment is on a runway
        self.duration = float(self.times[-1])

    def position(self, t: float) -> np.ndarray:
        t = min(max(t, 0.0), self.duration)
        i = min(
            int(np.searchsorted(self.times, t, side="right")) - 1, len(self.ground) - 1
        )
        span = self.times[i + 1] - self.times[i]
        k = 0.0 if span <= 0 else (t - self.times[i]) / span
        return self.points[i] + (self.points[i + 1] - self.points[i]) * k

    def pose(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Position, forward and up vectors, and whether on the ground, at ``t``."""
        here = self.position(t)
        ahead, behind = self.position(t + 1.0), self.position(t - 1.0)
        forward = ahead - behind
        norm = float(np.linalg.norm(forward))
        forward = forward / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])
        i = min(
            int(np.searchsorted(self.times, t, side="right")) - 1, len(self.ground) - 1
        )
        on_ground = bool(self.ground[max(i, 0)])
        bank = 0.0
        if not on_ground:
            # Bank into turns: the turn rate from the heading a moment either side.
            h0 = _heading(here - self.position(t - 2.0))
            h1 = _heading(self.position(t + 2.0) - here)
            rate = (h1 - h0 + math.pi) % math.tau - math.pi
            speed = norm / 2.0
            bank = max(-0.7, min(0.7, math.atan(rate / 2.0 * speed / 30.0)))
        right = np.cross(forward, UP)
        norm = float(np.linalg.norm(right))
        right = right / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        level_up = np.cross(right, forward)
        up = level_up * math.cos(bank) - right * math.sin(bank)
        return here, forward, up, on_ground


def closest_approach(
    here: np.ndarray,
    other: np.ndarray,
    here_before: np.ndarray,
    other_before: np.ndarray,
) -> float:
    """Least distance between two aircraft moving in straight lines from their
    ``before`` positions to their current ones over the same time. Head on they
    close by more than a wingspan per frame, so the end positions alone could
    pass each other by."""
    start = other_before - here_before
    move = (other - here) - start
    length = float(move @ move)
    t = 0.0 if length < 1e-12 else min(1.0, max(0.0, -float(start @ move) / length))
    return float(np.linalg.norm(start + move * t))


def midair(gap: float, span: float, other_span: float, share: float) -> bool:
    """Whether two aircraft ``gap`` apart touch: closer than ``share`` of their
    half-spans added (sprites are drawn a little larger than solid)."""
    return gap < (span + other_span) / 2 * share


def _heading(v: np.ndarray) -> float:
    return math.atan2(v[0], -v[1])


class Traffic:
    def __init__(
        self, seed: int, generator, types: int, settings: TrafficSettings | None = None
    ) -> None:
        self.seed = seed
        self.generator = generator  # a TerrainGenerator: airfields and heights
        self.types = types  # how many aircraft types there are to choose from
        self.settings = settings or TrafficSettings()
        self._routes: dict[tuple, Route | None] = {}
        self._schedule: dict[tuple, tuple[float, float] | None] = {}
        self._pending: list[tuple] = []

    # --- which routes exist -------------------------------------------------

    @staticmethod
    def _key(field) -> tuple[int, int]:
        return round(field.x), round(field.y)

    def pairs_near(self, x: float, y: float) -> list[tuple]:
        """Airfield pairs with a route, among airfields near (x, y)."""
        s = self.settings
        r = s.search
        fields = self.generator.airfields.overlapping(x - r, y - r, x + r, y + r)
        fields.sort(key=self._key)
        pairs = []
        for i, a in enumerate(fields):
            for b in fields[i + 1 :]:
                d = math.hypot(a.x - b.x, a.y - b.y)
                if s.pair_range[0] <= d <= s.pair_range[1] and self.schedule(a, b):
                    pairs.append((a, b))
        return pairs

    def schedule(self, a, b) -> tuple[float, float] | None:
        """(period, phase) of the route between two airfields, or None if the
        pair has no route. Depends only on the pair and the seed."""
        key = (*self._key(a), *self._key(b))
        if key not in self._schedule:
            rng = np.random.default_rng(
                [self.seed & 0xFFFFFFFF, 53, *(v + 2**31 for v in key)]
            )
            if rng.random() > self.settings.pair_chance:
                self._schedule[key] = None
            else:
                period = float(rng.uniform(*self.settings.period))
                self._schedule[key] = (period, float(rng.uniform(0, period)))
        return self._schedule[key]

    # --- flights --------------------------------------------------------------

    def flights(
        self, x: float, y: float, time: float, build: bool = True
    ) -> list[Flight]:
        """Aircraft within ``draw_radius`` of (x, y) at game ``time``, nearest
        first, at most ``max_planes``. Routes not laid out yet are queued and a
        few are built per call (all of them with ``build=False`` never)."""
        s = self.settings
        found = []
        for a, b in self.pairs_near(x, y):
            period, phase = self.schedule(a, b)
            for direction, (start, end) in enumerate(((a, b), (b, a))):
                route = self.route(start, end, build)
                if route is None:
                    continue
                offset = phase + direction * period / 2
                first = math.ceil((time - route.duration - offset) / period)
                last = math.floor((time - offset) / period)
                for k in range(first, last + 1):
                    t = time - (offset + k * period)
                    here = route.position(t)
                    if math.hypot(here[0] - x, here[1] - y) > s.draw_radius:
                        continue
                    found.append((a, b, direction, k, route, t, here))
        if build:
            self._build_some()
        found.sort(key=lambda f: math.hypot(f[6][0] - x, f[6][1] - y))
        flights = []
        for a, b, direction, k, route, t, _ in found[: s.max_planes]:
            ident = (*self._key(a), *self._key(b), direction, k)
            kind = int(
                np.random.default_rng(
                    [self.seed & 0xFFFFFFFF, 59, *(abs(v) for v in ident)]
                ).integers(0, self.types)
            )
            position, forward, up, on_ground = route.pose(t)
            flights.append(Flight(ident, kind, position, forward, up, on_ground))
        return flights

    def route(self, start, end, build: bool = True) -> Route | None:
        key = (*self._key(start), *self._key(end))
        if key in self._routes:
            return self._routes[key]
        if key not in [p[0] for p in self._pending]:
            self._pending.append((key, start, end))
        if not build:
            # Callers that need every route now (tests) lay it out at once.
            self._routes[key] = self._lay_out(start, end)
            return self._routes[key]
        return None

    def _build_some(self) -> None:
        for _ in range(min(self.settings.builds_per_frame, len(self._pending))):
            key, start, end = self._pending.pop(0)
            if key not in self._routes:
                self._routes[key] = self._lay_out(start, end)

    # --- laying out a route ---------------------------------------------------

    def _lay_out(self, a, b) -> Route | None:
        """The shortest workable path from runway ``a`` to runway ``b``, taking
        off and landing either way along each, or None if the terrain around
        them is too high to climb over and come down again in time.

        Seen from above, a path is the take-off roll, a straight climb-out, a
        curve round to a point on ``b``'s extended centre line, the final
        approach and the landing roll. The height along it is as low as it
        can be while staying above the terrain (by ``min_clearance`` close to
        the runways, growing to ``cruise_margin`` further out), never climbing
        or descending more steeply than ``slope``, and level with each runway
        at its end.
        """
        shapes = [
            shape
            for out_sign in (1, -1)
            for into_sign in (1, -1)
            if (shape := self._shape(a, b, out_sign, into_sign)) is not None
        ]
        if not shapes:
            return None
        # One terrain query for all of them: each query has a large fixed cost.
        xy = np.concatenate([shape[0] for shape in shapes])
        g = self.generator
        terrain = np.maximum(g.elevation(xy[:, 0], xy[:, 1]), g.settings.sea_level)
        splits = np.cumsum([len(shape[0]) for shape in shapes])[:-1]
        options = [
            route
            for shape, ground in zip(shapes, np.split(terrain, splits))
            if (route := self._profile(*shape, ground)) is not None
        ]
        return min(options, key=lambda r: r.duration, default=None)

    def _shape(self, a, b, out_sign: int, into_sign: int):
        """The path seen from above, taking off along ``a`` in direction
        ``out_sign`` and landing along ``b`` in direction ``into_sign``:
        (points, take-off height, landing height), or None if the runways
        point past each other and leave no room to turn in."""
        s = self.settings
        towards = np.array([b.x - a.x, b.y - a.y])
        towards /= np.linalg.norm(towards)
        out = np.array(a.along) * out_sign
        into = np.array(b.along) * into_sign
        start = np.array([a.x, a.y]) - out * (a.length / 2 - 5)  # just inside
        liftoff = start + out * s.takeoff_roll
        climb_out = liftoff + out * s.climb_out
        touchdown = np.array([b.x, b.y]) - into * b.length / 2
        approach = touchdown - into * s.final_length
        stop = touchdown + into * s.landing_roll
        if np.dot(approach - climb_out, towards) < s.turn_radius:
            return None
        air = _curve([liftoff, climb_out, approach, touchdown], s.turn_radius, s.step)
        xy = np.concatenate(
            [
                _line(start, liftoff, s.step),
                air[1:],
                _line(touchdown, stop, s.step)[1:],
            ]
        )
        return xy, a.elevation + s.gear_height, b.elevation + s.gear_height

    def _profile(self, xy, za: float, zb: float, terrain: np.ndarray) -> Route | None:
        """Heights along the path ``xy`` over ``terrain``, or None if no profile
        within the slope clears it."""
        s = self.settings
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dist = np.concatenate([[0.0], np.cumsum(steps)])
        lift, down = s.takeoff_roll, dist[-1] - s.landing_roll

        # Lowest height allowed at each point, then the lowest profile that
        # clears all of them without getting steeper than the slope.
        need = np.full(len(xy), -np.inf)
        away = (dist > lift + s.runway_reach) & (dist < down - s.runway_reach)
        # The margin grows from the least clearance near the runways to the
        # full cruise margin, no faster than a climb can follow.
        from_runway = np.minimum(dist - lift, down - dist) - s.runway_reach
        margin = np.clip(
            s.min_clearance + 0.6 * s.slope * from_runway,
            s.min_clearance,
            s.cruise_margin,
        )
        need[away] = terrain[away] + margin[away]
        need[dist <= lift] = za
        need[dist >= down] = zb
        cruise = float(need.max())
        climb = np.minimum(za + s.slope * (dist - lift), zb + s.slope * (down - dist))
        profile = np.maximum(np.minimum(cruise, climb), _cone(dist, need, s.slope))
        ground = (dist <= lift) | (dist >= down)
        profile[ground] = np.where(dist[ground] <= lift, za, zb)
        if np.any(np.abs(np.diff(profile)) > s.slope * steps + 1e-6):
            return None  # it would have to climb or dive too steeply
        if (profile - terrain)[away].min() < s.min_clearance - 1e-6:
            return None
        points = np.column_stack([xy, profile])
        middle = (dist[:-1] + dist[1:]) / 2
        on_ground = (middle < lift) | (middle > down)
        speeds = np.where(
            on_ground,
            s.roll_speed,
            np.where(middle > down - s.final_length, s.final_speed, s.cruise_speed),
        )
        return Route(points, speeds, on_ground)


def _line(a: np.ndarray, b: np.ndarray, step: float = 20.0) -> np.ndarray:
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    return a + np.outer(np.linspace(0, 1, n), b - a)


def _curve(corners: list[np.ndarray], radius: float, step: float = 20.0) -> np.ndarray:
    """Points every ~``step`` along straight legs through ``corners``, with each
    inner corner replaced by a curve so the heading turns smoothly."""
    points = [corners[0][None, :]]
    cursor = corners[0]
    for i in range(1, len(corners) - 1):
        prev, here, nxt = corners[i - 1], corners[i], corners[i + 1]
        d_in, d_out = here - prev, nxt - here
        l_in, l_out = float(np.linalg.norm(d_in)), float(np.linalg.norm(d_out))
        cos = float(np.dot(d_in, d_out) / max(l_in * l_out, 1e-9))
        turn = math.acos(max(-1.0, min(1.0, cos)))
        cut = min(radius * math.tan(min(turn, 3.0) / 2), l_in * 0.45, l_out * 0.45)
        p0, p2 = here - d_in / l_in * cut, here + d_out / l_out * cut
        points.append(_line(cursor, p0, step)[1:])
        u = np.linspace(0, 1, max(2, int(cut * 2 / step)) + 1)[1:, None]
        points.append((1 - u) ** 2 * p0 + 2 * (1 - u) * u * here + u**2 * p2)
        cursor = p2
    points.append(_line(cursor, corners[-1], step)[1:])
    return np.concatenate(points)


def _cone(dist: np.ndarray, need: np.ndarray, slope: float) -> np.ndarray:
    """The lowest heights above every ``need`` whose slope stays within ``slope``:
    max over j of need[j] - slope * |dist - dist[j]|, in two linear sweeps."""
    out = need.copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] - slope * (dist[i] - dist[i - 1]))
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1] - slope * (dist[i + 1] - dist[i]))
    return out
