"""Arcade flight model with a few physical traits.

The aircraft flies where its nose points, with three twists that make it feel
like a plane rather than a camera: it turns by banking, it trades speed for
height (and back), and below stall speed the wings stop lifting so it sinks
and drops its nose.

Units are world units and seconds. Nothing here knows about Pyxel; terrain
comes in as a ``ground(x, y) -> height`` function, so the model is easy to
test.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

Ground = Callable[[float, float], float]


@dataclass(frozen=True)
class AircraftSettings:
    gravity: float = 30.0

    # Engine and drag. Top speed in level flight at sea level is
    # sqrt(max_thrust / drag), here about 140.
    max_thrust: float = 40.0
    drag: float = 0.002
    # Thrust falls off with altitude: thrust * exp(-altitude / scale_height).
    scale_height: float = 1500.0
    throttle_rate: float = 0.5  # throttle change per second at full input

    stall_speed: float = 50.0
    takeoff_speed: float = 58.0
    # Controls reach full effect at this speed and weaken below it.
    control_speed: float = 75.0

    max_pitch_rate: float = 0.9  # radians per second
    max_roll_rate: float = 1.8
    max_yaw_rate: float = 0.25
    control_lag: float = 0.15  # seconds for rates to follow the stick
    roll_leveling: float = 0.8  # per second, when the stick is centred
    stall_nose_drop: float = 0.8  # radians per second at full stall
    # How quickly sink speed settles; terminal sink is gravity * deficit / this.
    sink_damping: float = 1.5

    max_pitch: float = math.radians(50)
    max_roll: float = math.radians(60)

    # Ground handling.
    gear_height: float = 1.5
    rolling_friction: float = 3.0
    brake_force: float = 20.0
    ground_steer_rate: float = 0.6
    max_ground_pitch: float = math.radians(15)

    # Touchdown limits: exceed any of them and it is a crash, not a landing.
    max_touchdown_sink: float = 8.0
    max_touchdown_roll: float = math.radians(12)
    min_touchdown_pitch: float = math.radians(-4)
    max_ground_slope: float = 0.3
    # Once rolling on the ground, only run into terrain steeper than this.
    max_taxi_slope: float = 0.6

    # Respawn.
    spawn_height: float = 120.0
    spawn_speed: float = 100.0
    spawn_throttle: float = 0.7

    sea_level: float = 0.0


class FlightState(Enum):
    FLYING = auto()
    LANDED = auto()
    CRASHED = auto()


@dataclass
class Controls:
    """Pilot input, each axis in [-1, 1]."""

    pitch: float = 0.0  # positive = pull back, nose up
    roll: float = 0.0  # positive = roll right
    yaw: float = 0.0  # positive = yaw right
    throttle: float = 0.0  # positive = open throttle
    brake: bool = False


@dataclass
class Aircraft:
    x: float = 0.0
    y: float = 0.0
    altitude: float = 0.0
    heading: float = 0.0  # radians clockwise from north (-y)
    pitch: float = 0.0  # radians, positive = nose up
    roll: float = 0.0  # radians, positive = right wing down
    speed: float = 0.0
    throttle: float = 0.0
    state: FlightState = FlightState.FLYING
    crash_reason: str = ""
    # Seconds since the last crash or respawn.
    state_time: float = 0.0
    vertical_speed: float = 0.0
    settings: AircraftSettings = AircraftSettings()

    pitch_rate: float = 0.0
    roll_rate: float = 0.0
    sink: float = 0.0  # extra descent speed from missing lift

    @property
    def lift(self) -> float:
        """Fraction of full lift the wings produce: 1 above stall speed."""
        return min(1.0, (self.speed / self.settings.stall_speed) ** 2)

    @property
    def stalling(self) -> bool:
        return (
            self.state is FlightState.FLYING and self.speed < self.settings.stall_speed
        )

    def respawn(self, x: float, y: float, heading: float, ground: Ground) -> None:
        """Start flying again above (x, y)."""
        s = self.settings
        self.x, self.y, self.heading = x, y, heading
        self.altitude = max(ground(x, y), s.sea_level) + s.spawn_height
        self.speed = s.spawn_speed
        self.throttle = s.spawn_throttle
        self.pitch = self.roll = self.pitch_rate = self.roll_rate = 0.0
        self.sink = self.vertical_speed = 0.0
        self.state = FlightState.FLYING
        self.crash_reason = ""
        self.state_time = 0.0

    def update(self, controls: Controls, dt: float, ground: Ground) -> None:
        self.state_time += dt
        if self.state is FlightState.CRASHED:
            return
        s = self.settings
        on_ground = self.state is FlightState.LANDED

        self.throttle = _clamp(
            self.throttle + controls.throttle * s.throttle_rate * dt, 0, 1
        )
        self._rotate(controls, dt, on_ground)
        self._accelerate(controls, dt, on_ground)

        # Banking tilts the lift away from vertical and a stall removes it;
        # either way the missing lift turns into sink.
        if on_ground:
            self.sink = 0.0
        else:
            deficit = 1 - self.lift * math.cos(self.roll)
            self.sink += (s.gravity * deficit - self.sink * s.sink_damping) * dt
            self.sink = max(self.sink, 0.0)

        horizontal = self.speed * math.cos(self.pitch)
        self.x += math.sin(self.heading) * horizontal * dt
        self.y -= math.cos(self.heading) * horizontal * dt
        climb = self.speed * math.sin(self.pitch) - self.sink
        if on_ground and self.speed < s.takeoff_speed:
            climb = min(climb, 0.0)
        self.vertical_speed = climb
        self.altitude += climb * dt

        self._touch_ground(ground)

    def _rotate(self, controls: Controls, dt: float, on_ground: bool) -> None:
        s = self.settings
        authority = _clamp(self.speed / s.control_speed, 0.25, 1)
        follow = min(1.0, dt / s.control_lag)
        self.pitch_rate += (
            controls.pitch * s.max_pitch_rate * authority - self.pitch_rate
        ) * follow
        self.pitch += self.pitch_rate * dt

        if on_ground:
            # Wings stay level; stick and rudder steer the nose wheel instead.
            self.roll = self.roll_rate = 0.0
            self.pitch = _clamp(self.pitch, 0, s.max_ground_pitch)
            steer = _clamp(controls.yaw + controls.roll, -1, 1)
            self.heading += steer * s.ground_steer_rate * min(1.0, self.speed / 10) * dt
        else:
            self.roll_rate += (
                controls.roll * s.max_roll_rate * authority - self.roll_rate
            ) * follow
            self.roll += self.roll_rate * dt
            if controls.roll == 0:
                self.roll -= self.roll * s.roll_leveling * dt
            self.pitch -= s.stall_nose_drop * (1 - self.lift) * dt
            self.pitch = _clamp(self.pitch, -s.max_pitch, s.max_pitch)
            self.roll = _clamp(self.roll, -s.max_roll, s.max_roll)

            # A banked wing pulls the nose round: the classic coordinated turn.
            turn = (
                s.gravity
                * math.tan(self.roll)
                / max(self.speed, s.stall_speed)
                * self.lift
            )
            self.heading += (turn + controls.yaw * s.max_yaw_rate * authority) * dt
        self.heading %= math.tau

    def _accelerate(self, controls: Controls, dt: float, on_ground: bool) -> None:
        s = self.settings
        density = math.exp(-max(self.altitude, 0.0) / s.scale_height)
        accel = (
            self.throttle * s.max_thrust * density
            - s.drag * self.speed**2
            - s.gravity * math.sin(self.pitch)
        )
        if on_ground:
            accel -= s.rolling_friction + (s.brake_force if controls.brake else 0.0)
        self.speed = max(0.0, self.speed + accel * dt)

    def _touch_ground(self, ground: Ground) -> None:
        s = self.settings
        height = ground(self.x, self.y)
        floor = height + s.gear_height
        if self.altitude > floor:
            if self.state is FlightState.LANDED:
                self.state = FlightState.FLYING  # took off or rolled over an edge
            return

        problem = self._ground_problem(height, ground)
        if problem:
            self._crash(problem)
            return
        if self.state is FlightState.FLYING:
            self.state = FlightState.LANDED
            self.roll = self.roll_rate = 0.0
            self.pitch = max(self.pitch, 0.0)
            self.pitch_rate = 0.0
        self.altitude = floor
        self.sink = 0.0
        self.vertical_speed = 0.0

    def _ground_problem(self, height: float, ground: Ground) -> str:
        """Why touching the ground here is a crash, or "" if it is fine."""
        s = self.settings
        landed = self.state is FlightState.LANDED
        if height <= s.sea_level:
            return "DITCHED IN THE WATER"
        max_slope = s.max_taxi_slope if landed else s.max_ground_slope
        if _slope(ground, self.x, self.y, height) > max_slope:
            return "HIT THE TERRAIN"
        if landed:
            return ""
        if self.vertical_speed < -s.max_touchdown_sink:
            return "HIT THE GROUND TOO HARD"
        if abs(self.roll) > s.max_touchdown_roll:
            return "WINGTIP HIT THE GROUND"
        if self.pitch < s.min_touchdown_pitch:
            return "NOSED INTO THE GROUND"
        return ""

    def _crash(self, reason: str) -> None:
        self.state = FlightState.CRASHED
        self.crash_reason = reason
        self.state_time = 0.0
        self.speed = 0.0
        self.vertical_speed = 0.0
        self.pitch_rate = self.roll_rate = 0.0


def _slope(ground: Ground, x: float, y: float, height: float, d: float = 3.0) -> float:
    """Steepest rise or fall (rise over run) from (x, y) to four nearby points."""
    return (
        max(
            abs(ground(x + dx, y + dy) - height)
            for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d))
        )
        / d
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
