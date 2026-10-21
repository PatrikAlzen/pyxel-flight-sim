"""Small goals that reward trying everything the game has.

Each achievement is a key, a name and a description. They are earned three
ways:

- detectors that watch the flight frame by frame (a loop, flying inverted, a
  low pass, following a river...), fed a ``Sample`` so they can be tested with
  made-up flight paths;
- game events (a graded landing, a crash, passing under a bridge, a near
  miss);
- notes from the game about things it already tracks (airfields visited,
  lighthouses found, photos taken, time trials finished).

What is earned, and the counts on the way there, live in the player's profile
(``Profile.earned`` and ``Profile.progress``), so they carry across worlds and
sessions. Nothing here knows about Pyxel.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from flightsim.sim.events import Event, EventKind


@dataclass(frozen=True)
class Achievement:
    key: str
    name: str
    description: str
    hidden: bool = False  # shown as ??? until earned


ACHIEVEMENTS = [
    # Exploring.
    Achievement("FIRST_FLIGHT", "FIRST FLIGHT", "Take off from the ground"),
    Achievement("VISIT_5", "REGULAR", "Land at 5 different airfields"),
    Achievement("VISIT_10", "FREQUENT FLYER", "Land at 10 different airfields"),
    Achievement("VISIT_25", "AIRFIELD COLLECTOR", "Land at 25 different airfields"),
    Achievement("LIGHTHOUSES_5", "KEEPER", "Discover 5 lighthouses"),
    Achievement("DISTANCE_100", "LONG HAUL", "Fly 100 km in one world"),
    # Landings.
    Achievement("BUTTER", "BUTTER", "Touch down with almost no sink"),
    Achievement("DEADSTICK", "DEADSTICK", "Land on a runway with the engine out"),
    Achievement(
        "CROSSWIND", "CROSSWIND", "Land with over 12 of wind across the runway"
    ),
    Achievement("SPLASHDOWN", "SPLASHDOWN", "Land a seaplane on water"),
    # Aerobatics.
    Achievement("LOOP", "LOOP THE LOOP", "Fly a full loop"),
    Achievement(
        "LOOP_LOW", "LOW LOOP", "Fly a loop that dips below 50 above the ground"
    ),
    Achievement("INVERTED", "UPSIDE DOWN", "Fly inverted for 10 seconds"),
    Achievement(
        "BARREL_ROLL", "DO A BARREL ROLL", "Roll all the way round in 6 seconds"
    ),
    Achievement("LOW_PASS", "LOW PASS", "Over 120 fast, under 10 high, for 3 seconds"),
    # Weather.
    Achievement("IN_THE_CLOUDS", "HEAD IN THE CLOUDS", "Fly 3 seconds inside a cloud"),
    Achievement("STORM_CHASER", "STORM CHASER", "Fly through a storm for a minute"),
    # Time trials.
    Achievement("FINISH_TRIAL", "ON THE CLOCK", "Finish a time trial"),
    Achievement("BEAT_RECORD", "PERSONAL BEST", "Beat your own time-trial record"),
    Achievement("FAST_TRIAL", "SPEED DEMON", "Finish a time trial in under 60 seconds"),
    # Rivers, bridges, traffic, photos.
    Achievement("RIVER_RUN", "RIVER RUN", "Follow a river for 5 km below 50"),
    Achievement("UNDER_BRIDGE", "UNDER THE BRIDGE", "Fly under a bridge"),
    Achievement("BRIDGES_5", "BRIDGE TROLL", "Fly under 5 different bridges"),
    Achievement("NEAR_MISS", "TOO CLOSE", "Pass within 30 of another aircraft"),
    Achievement("PHOTOGRAPHER", "PHOTOGRAPHER", "Take 10 photos"),
    Achievement("HARD_LESSONS", "HARD LESSONS", "Fly into a lighthouse", hidden=True),
]
BY_KEY = {a.key: a for a in ACHIEVEMENTS}
FULL_TURN = math.tau - 0.01  # a whole circle, give or take rounding


@dataclass(frozen=True)
class AchievementSettings:
    loop_window: float = 20.0  # seconds to turn a full circle in pitch
    loop_wings: float = 0.5  # |right wing's z| above this (30 degrees) breaks it
    loop_low: float = 50.0
    loop_turning: float = 0.3  # radians per second of pitch that count as looping
    inverted_up: float = -0.8  # up vector's z below this is inverted
    inverted_time: float = 10.0
    roll_window: float = 6.0
    low_pass_speed: float = 120.0
    low_pass_height: float = 10.0
    low_pass_time: float = 3.0
    cloud_time: float = 3.0
    storm_time: float = 60.0
    river_height: float = 50.0
    river_distance: float = 5000.0
    world_distance: float = 100_000.0
    crosswind: float = 12.0
    trial_fast: float = 60.0
    jump: float = 50.0  # moved further in a frame: respawned, not flown


@dataclass
class Sample:
    """The plane in one frame, as the detectors see it."""

    position: np.ndarray  # x, y, altitude
    forward: np.ndarray
    up: np.ndarray
    speed: float
    agl: float  # height above the ground
    flying: bool  # in the air (not landed, not crashed)
    in_cloud: bool = False
    storm: bool = False
    over_river: bool = False


def _turned(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from ``a`` to ``b`` about ``axis``."""
    return math.atan2(float(np.cross(a, b) @ axis), float(a @ b))


@dataclass
class _Window:
    """Rotation summed over the last ``seconds``."""

    seconds: float
    steps: deque = field(default_factory=deque)  # (time, angle, agl)
    total: float = 0.0

    def add(self, time: float, angle: float, agl: float) -> None:
        self.steps.append((time, angle, agl))
        self.total += angle
        while self.steps and self.steps[0][0] < time - self.seconds:
            self.total -= self.steps.popleft()[1]

    def clear(self) -> None:
        self.steps.clear()
        self.total = 0.0

    def lowest(self, turning: float) -> float:
        """Least height above the ground while actually turning (by more than
        ``turning`` a step), so level flight before a loop does not count."""
        return min(
            (agl for _, angle, agl in self.steps if abs(angle) > turning),
            default=math.inf,
        )


class Tracker:
    """Watches samples, events and notes; returns the keys of new achievements."""

    def __init__(
        self,
        earned: dict[str, dict],
        progress: dict,
        seed: int,
        settings: AchievementSettings | None = None,
    ) -> None:
        self.earned = earned  # the profile's, updated in place
        self.progress = progress
        self.seed = seed
        self.settings = settings or AchievementSettings()
        self.dirty = False  # progress changed since the profile was saved
        self.time = 0.0
        self.last: Sample | None = None
        self.loop = _Window(self.settings.loop_window)
        self.roll = _Window(self.settings.roll_window)
        self.inverted = self.low_pass = self.cloud = self.storm = 0.0
        self.river = 0.0

    # --- earning ---------------------------------------------------------

    def _earn(self, *keys: str) -> list[str]:
        new = [k for k in keys if k not in self.earned]
        for key in new:
            self.earned[key] = {"date": date.today().isoformat(), "seed": self.seed}
            self.dirty = True
        return new

    def _collect(self, name: str, item: str) -> int:
        """Add ``item`` to the progress list ``name``; how many it holds now."""
        items = self.progress.setdefault(name, [])
        if item not in items:
            items.append(item)
            self.dirty = True
        return len(items)

    def _count(self, name: str, amount: float = 1.0) -> float:
        self.progress[name] = self.progress.get(name, 0) + amount
        self.dirty = True
        return self.progress[name]

    # --- flight detectors ------------------------------------------------

    def update(self, sample: Sample, dt: float) -> list[str]:
        s = self.settings
        self.time += dt
        last, self.last = self.last, sample
        new = []
        if last is not None and not last.flying and sample.flying:
            new += self._earn("FIRST_FLIGHT")
        jumped = last is not None and (
            float(np.linalg.norm(sample.position - last.position)) > s.jump
        )
        if not sample.flying or last is None or not last.flying or jumped:
            self.loop.clear()
            self.roll.clear()
            self.inverted = self.low_pass = self.cloud = self.storm = 0.0
            self.river = 0.0
            return new

        # Loop: the nose turns a full circle about the wings while they stay
        # roughly level (so a steep turn does not count).
        right = np.cross(sample.forward, sample.up)
        if abs(right[2]) > s.loop_wings:
            self.loop.clear()
        else:
            self.loop.add(
                self.time, _turned(last.forward, sample.forward, right), sample.agl
            )
            if abs(self.loop.total) >= FULL_TURN:
                new += self._earn("LOOP")
                if self.loop.lowest(s.loop_turning * dt) < s.loop_low:
                    new += self._earn("LOOP_LOW")
                self.loop.clear()

        # Barrel roll: the canopy turns a full circle about the nose.
        self.roll.add(
            self.time, _turned(last.up, sample.up, sample.forward), sample.agl
        )
        if abs(self.roll.total) >= FULL_TURN:
            new += self._earn("BARREL_ROLL")
            self.roll.clear()

        self.inverted = self.inverted + dt if sample.up[2] < s.inverted_up else 0.0
        if self.inverted >= s.inverted_time:
            new += self._earn("INVERTED")
        low = sample.speed > s.low_pass_speed and sample.agl < s.low_pass_height
        self.low_pass = self.low_pass + dt if low else 0.0
        if self.low_pass >= s.low_pass_time:
            new += self._earn("LOW_PASS")
        self.cloud = self.cloud + dt if sample.in_cloud else 0.0
        if self.cloud >= s.cloud_time:
            new += self._earn("IN_THE_CLOUDS")
        self.storm = self.storm + dt if sample.storm else 0.0
        if self.storm >= s.storm_time:
            new += self._earn("STORM_CHASER")

        moved = float(np.hypot(*(sample.position[:2] - last.position[:2])))
        on_river = sample.over_river and sample.agl < s.river_height
        self.river = self.river + moved if on_river else 0.0
        if self.river >= s.river_distance:
            new += self._earn("RIVER_RUN")
        if self._count(f"distance:{self.seed}", moved) >= s.world_distance:
            new += self._earn("DISTANCE_100")
        return new

    # --- events and notes ------------------------------------------------

    def on_event(self, event: Event) -> list[str]:
        new = []
        match event.kind:
            case EventKind.LANDING:
                if "BUTTER" in event.text:
                    new += self._earn("BUTTER")
                if "DEADSTICK" in event.text and "FIELD" in event.text:
                    new += self._earn("DEADSTICK")
                if event.text.startswith("SPLASHDOWN"):
                    new += self._earn("SPLASHDOWN")
            case EventKind.CRASH if event.text == "HIT THE LIGHTHOUSE":
                new += self._earn("HARD_LESSONS")
            case EventKind.BRIDGE:
                new += self._earn("UNDER_BRIDGE")
            case EventKind.NEAR_MISS:
                new += self._earn("NEAR_MISS")
        return new

    def landed_at(self, field_number: int, crosswind: float) -> list[str]:
        """A touchdown on airfield ``field_number`` with ``crosswind`` across it."""
        new = []
        visited = self._collect("airfields", f"{self.seed}:{field_number}")
        for count, key in ((5, "VISIT_5"), (10, "VISIT_10"), (25, "VISIT_25")):
            if visited >= count:
                new += self._earn(key)
        if crosswind > self.settings.crosswind:
            new += self._earn("CROSSWIND")
        return new

    def discovered_lighthouse(self, number: int) -> list[str]:
        found = self._collect("lighthouses", f"{self.seed}:{number}")
        return self._earn("LIGHTHOUSES_5") if found >= 5 else []

    def flew_under_bridge(self, number: int) -> list[str]:
        bridges = self._collect("bridges", f"{self.seed}:{number}")
        return self._earn("BRIDGES_5") if bridges >= 5 else []

    def took_photo(self) -> list[str]:
        return self._earn("PHOTOGRAPHER") if self._count("photos") >= 10 else []

    def finished_trial(self, seconds: float, record: float | None) -> list[str]:
        """A time trial finished in ``seconds``; ``record`` is the best before."""
        new = self._earn("FINISH_TRIAL")
        if record is not None and seconds < record:
            new += self._earn("BEAT_RECORD")
        if seconds < self.settings.trial_fast:
            new += self._earn("FAST_TRIAL")
        return new
