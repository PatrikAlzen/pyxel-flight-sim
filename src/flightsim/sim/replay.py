"""Recording the last minute of flight and playing it back.

``Recorder`` samples the plane ten times a second into a ring buffer, and
keeps the events (crash, landing, ring) of that minute with their times. A
jump in position, such as a respawn or a trip to a saved place, starts a new
recording, so a replay never shows the plane teleporting.

``Recording`` is a frozen copy that can be asked where the plane was at any
time: positions are interpolated linearly, the forward and up vectors by
normalised interpolation. ``Playback`` steps through a recording at a chosen
speed and reports the events it passes.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from flightsim.sim.aircraft import Aircraft, FlightState
from flightsim.sim.events import Event, EventKind

STATES = list(FlightState)


@dataclass(frozen=True)
class ReplaySettings:
    interval: float = 0.1  # seconds between samples
    length: float = 60.0  # seconds kept
    jump: float = 50.0  # moved further than this between samples: a new recording
    after_crash: float = 4.0  # seconds of the crash scene kept after impact
    speeds: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    throttle: float
    speed: float
    state: FlightState


class Recorder:
    def __init__(self, settings: ReplaySettings | None = None) -> None:
        self.settings = s = settings or ReplaySettings()
        self.samples: deque[np.ndarray] = deque(maxlen=round(s.length / s.interval))
        self.events: deque[tuple[float, Event]] = deque()
        self.time = 0.0  # seconds recorded, also between samples
        self.next_sample = 0.0

    def clear(self) -> None:
        self.samples.clear()
        self.events.clear()
        self.next_sample = self.time

    def record(self, plane: Aircraft, dt: float) -> None:
        """Advance by ``dt`` and sample the plane when a sample is due."""
        s = self.settings
        self.time += dt
        if plane.state is FlightState.CRASHED and plane.state_time > s.after_crash:
            return  # keep the crash in the buffer instead of a minute of wreck
        if self.samples:
            last = self.samples[-1]
            moved = np.linalg.norm(last[1:4] - (plane.x, plane.y, plane.altitude))
            revived = (
                STATES[int(last[12])] is FlightState.CRASHED
                and plane.state is not FlightState.CRASHED
            )
            if moved > s.jump or revived:
                self.clear()
        if self.time + 1e-9 < self.next_sample:
            return
        self.next_sample += s.interval  # keep the cadence instead of drifting
        if self.next_sample <= self.time:
            self.next_sample = self.time + s.interval
        self.samples.append(
            np.array(
                [
                    self.time,
                    plane.x,
                    plane.y,
                    plane.altitude,
                    *plane.forward,
                    *plane.up,
                    plane.throttle,
                    plane.speed,
                    STATES.index(plane.state),
                ],
                dtype=float,
            )
        )
        start = self.samples[0][0]
        while self.events and self.events[0][0] < start:
            self.events.popleft()

    def mark(self, event: Event) -> None:
        """Remember an event at the current time (an ``EventLog`` listener)."""
        self.events.append((self.time, event))

    @property
    def duration(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return float(self.samples[-1][0] - self.samples[0][0])

    def recording(self) -> Recording | None:
        """A copy of what is in the buffer, or None if there is too little."""
        if len(self.samples) < 2:
            return None
        rows = np.array(self.samples)
        start = rows[0, 0]
        rows[:, 0] -= start
        events = [(t - start, e) for t, e in self.events if t >= start]
        return Recording(rows, events)


class Recording:
    """A recorded flight, times from 0 to ``duration`` seconds."""

    def __init__(self, rows: np.ndarray, events: list[tuple[float, Event]]) -> None:
        self.times = rows[:, 0]
        self.positions = rows[:, 1:4]
        self.forwards = rows[:, 4:7]
        self.ups = rows[:, 7:10]
        self.throttles = rows[:, 10]
        self.speeds = rows[:, 11]
        self.states = rows[:, 12].astype(int)
        self.events = events

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def at(self, t: float) -> Pose:
        t = min(max(t, 0.0), self.duration)
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        i = min(max(i, 0), len(self.times) - 2)
        span = self.times[i + 1] - self.times[i]
        k = 0.0 if span <= 0 else (t - self.times[i]) / span

        def lerp(a: np.ndarray) -> np.ndarray:
            return a[i] + (a[i + 1] - a[i]) * k

        forward = lerp(self.forwards)
        forward /= np.linalg.norm(forward)
        up = lerp(self.ups)
        up -= forward * float(up @ forward)
        up /= np.linalg.norm(up)
        state = STATES[self.states[i + 1] if k >= 1.0 else self.states[i]]
        return Pose(
            lerp(self.positions),
            forward,
            up,
            float(lerp(self.throttles)),
            float(lerp(self.speeds)),
            state,
        )

    def crash(self) -> tuple[float, Event] | None:
        """When and where the plane crashed in this recording, if it did."""
        return next(((t, e) for t, e in self.events if e.kind is EventKind.CRASH), None)


class Playback:
    def __init__(self, recording: Recording, settings: ReplaySettings | None = None):
        self.settings = settings or ReplaySettings()
        self.recording = recording
        self.time = 0.0
        self.speed_index = self.settings.speeds.index(1.0)
        self.paused = False

    @property
    def speed(self) -> float:
        return self.settings.speeds[self.speed_index]

    @property
    def finished(self) -> bool:
        return self.time >= self.recording.duration

    def step(self, dt: float) -> list[Event]:
        """Advance by ``dt`` of real time; the events passed on the way."""
        if self.paused or self.finished:
            return []
        before = self.time
        self.time = min(self.time + dt * self.speed, self.recording.duration)
        return [e for t, e in self.recording.events if before < t <= self.time]

    def seek(self, seconds: float) -> None:
        """Jump forwards or backwards; events on the way are skipped."""
        self.time = min(max(self.time + seconds, 0.0), self.recording.duration)

    def faster(self, steps: int) -> None:
        last = len(self.settings.speeds) - 1
        self.speed_index = min(max(self.speed_index + steps, 0), last)

    def pose(self) -> Pose:
        return self.recording.at(self.time)


@dataclass(frozen=True)
class TowerSettings:
    lead: float = 1.5  # seconds: stand where the plane is about to be...
    side: float = 45.0  # ...this far to one side of its path
    height: float = 6.0  # above the ground
    below: float = 40.0  # but at most this far below a plane flying higher up
    # Move on once the plane has flown past and is this far beyond the camera,
    # or, if it turned away instead, once it is this far off at all.
    passed: float = 120.0
    lost: float = 600.0


def tower_due(
    tower: np.ndarray | None,
    position: np.ndarray,
    forward: np.ndarray,
    settings: TowerSettings | None = None,
) -> bool:
    """Whether the tower camera should move to a new spot: the plane has gone
    by it (the tower is behind the nose) and is ``passed`` away, or it is
    ``lost`` away. A spot is chosen ahead of the plane, so it starts well over
    ``passed`` away and must not count as passed already."""
    if tower is None:
        return True
    s = settings or TowerSettings()
    offset = np.asarray(tower, float) - np.asarray(position, float)
    distance = float(np.linalg.norm(offset))
    behind = float(offset[:2] @ np.asarray(forward, float)[:2]) < 0
    return (behind and distance > s.passed) or distance > s.lost


def tower_spot(
    recording: Recording,
    t: float,
    ground,
    settings: TowerSettings | None = None,
    side: float = 1.0,
) -> np.ndarray:
    """Where to put a fixed camera that the plane will fly past shortly after ``t``:
    beside its path a little ahead, near the ground. ``side`` is +1 (right) or -1."""
    s = settings or TowerSettings()
    ahead = recording.at(t + s.lead)
    flat = np.array([ahead.forward[0], ahead.forward[1], 0.0])
    norm = float(np.linalg.norm(flat))
    flat = flat / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])
    right = np.array([-flat[1], flat[0], 0.0])
    spot = ahead.position + right * s.side * side
    floor = ground(float(spot[0]), float(spot[1])) + s.height
    spot[2] = max(floor, float(ahead.position[2]) - s.below)
    return spot
