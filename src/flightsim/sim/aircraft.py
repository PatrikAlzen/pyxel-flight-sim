"""Arcade flight model with a few physical traits.

The aircraft flies where its nose points, with three twists that make it feel
like a plane rather than a camera: it turns by banking, it trades speed for
height (and back), and below stall speed the wings stop lifting so it sinks
and drops its nose.

Orientation is kept as two unit vectors, ``forward`` (the nose) and ``up``
(the canopy), rotated about the aircraft's own axes. Unlike heading/pitch/roll
angles this has no special case pointing straight up, so it can roll all the
way round and fly loops. The angles are still available as properties.

Units are world units and seconds. Nothing here knows about Pyxel; terrain
comes in as a ``ground(x, y) -> height`` function, so the model is easy to
test.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from flightsim.sim.camera import basis_to_euler, euler_to_basis

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

    max_pitch_rate: float = 1.2  # radians per second; fast enough to loop
    max_roll_rate: float = 1.8
    max_yaw_rate: float = 0.25
    control_lag: float = 0.15  # seconds for rates to follow the stick
    roll_leveling: float = 0.8  # per second, when the stick is centred...
    roll_leveling_limit: float = math.radians(60)  # ...and the bank is below this
    stall_nose_drop: float = 0.8  # radians per second at full stall
    # How quickly sink speed settles; terminal sink is gravity * deficit / this.
    sink_damping: float = 1.5

    # Banking turns at gravity * this * sin(bank) / speed radians per second.
    turn_factor: float = 1.6

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

    # Fuel, as a share of a full tank. Burn rate at full throttle (a tank lasts
    # 8 minutes) times idle_burn + (1 - idle_burn) * throttle**1.5.
    full_burn: float = 1 / 480
    idle_burn: float = 0.3
    refuel_time: float = 5.0  # seconds to fill an empty tank
    cruise_throttle: float = 0.5  # for range estimates; a tank lasts ~15 minutes

    # Respawn.
    spawn_height: float = 120.0
    spawn_speed: float = 100.0
    spawn_throttle: float = 0.7

    sea_level: float = 0.0


class FlightState(Enum):
    FLYING = auto()
    LANDED = auto()
    CRASHED = auto()


class Alert(Enum):
    STALL = auto()
    PULL_UP = auto()


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
    speed: float = 0.0
    throttle: float = 0.0
    state: FlightState = FlightState.FLYING
    crash_reason: str = ""
    # Seconds since the last crash or respawn.
    state_time: float = 0.0
    vertical_speed: float = 0.0
    # Sink rate at the last touchdown (positive = descending), for grading landings.
    touchdown_sink: float = 0.0
    touchdown_engine_out: bool = False  # a deadstick landing
    fuel: float = 1.0  # share of a full tank
    uses_fuel: bool = False  # off: the tank never empties
    settings: AircraftSettings = AircraftSettings()

    # Orientation: unit vectors along the nose and out of the canopy.
    forward: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    pitch_rate: float = 0.0
    roll_rate: float = 0.0
    sink: float = 0.0  # extra descent speed from missing lift

    @property
    def right(self) -> np.ndarray:
        """Unit vector along the right wing."""
        return np.cross(self.up, self.forward)

    @property
    def heading(self) -> float:
        """Compass direction of the nose, radians clockwise from north (-y)."""
        return basis_to_euler(self.forward, self.up)[0]

    @heading.setter
    def heading(self, value: float) -> None:
        self._set_euler(heading=value)

    @property
    def pitch(self) -> float:
        """Nose angle above the horizon, in [-pi/2, pi/2]."""
        return basis_to_euler(self.forward, self.up)[1]

    @pitch.setter
    def pitch(self, value: float) -> None:
        self._set_euler(pitch=value)

    @property
    def roll(self) -> float:
        """Bank angle, positive = right wing down, +-pi = upside down."""
        return basis_to_euler(self.forward, self.up)[2]

    @roll.setter
    def roll(self, value: float) -> None:
        self._set_euler(roll=value)

    def _set_euler(self, **angles: float) -> None:
        current = dict(
            zip(("heading", "pitch", "roll"), basis_to_euler(self.forward, self.up))
        )
        current.update(angles)
        f, _, u = euler_to_basis(current["heading"], current["pitch"], current["roll"])
        self.forward, self.up = f, u

    @property
    def lift(self) -> float:
        """Fraction of full lift the wings produce: 1 above stall speed."""
        return min(1.0, (self.speed / self.settings.stall_speed) ** 2)

    @property
    def stalling(self) -> bool:
        return (
            self.state is FlightState.FLYING and self.speed < self.settings.stall_speed
        )

    @property
    def engine_running(self) -> bool:
        return not self.uses_fuel or self.fuel > 0

    def burn_rate(self, throttle: float | None = None) -> float:
        """Share of a full tank used per second at ``throttle`` (default: current)."""
        s = self.settings
        t = self.throttle if throttle is None else throttle
        return s.full_burn * (s.idle_burn + (1 - s.idle_burn) * t**1.5)

    def endurance(self) -> float:
        """Seconds until the tank is empty at the current throttle."""
        return self.fuel / self.burn_rate()

    def range_estimate(self) -> float:
        """How far the fuel left takes the plane in level flight at cruise
        throttle and the speed that throttle settles at at this altitude."""
        s = self.settings
        t = s.cruise_throttle
        density = math.exp(-max(self.altitude, 0.0) / s.scale_height)
        speed = math.sqrt(t * s.max_thrust * density / s.drag)
        return speed * self.fuel / self.burn_rate(t)

    def refuel(self, dt: float) -> None:
        """Pump fuel in; the caller checks the plane stands still at an airfield."""
        self.fuel = min(1.0, self.fuel + dt / self.settings.refuel_time)

    def alert(self, agl: float) -> Alert | None:
        """The most urgent cockpit alert at ``agl`` units above the ground."""
        if self.stalling:
            return Alert.STALL
        if self.state is FlightState.FLYING and agl < 40 and self.vertical_speed < -8:
            return Alert.PULL_UP
        return None

    def respawn(self, x: float, y: float, heading: float, ground: Ground) -> None:
        """Start flying again above (x, y), wings level."""
        s = self.settings
        self.x, self.y = x, y
        self.forward, _, self.up = euler_to_basis(heading, 0.0, 0.0)
        self.altitude = max(ground(x, y), s.sea_level) + s.spawn_height
        self.speed = s.spawn_speed
        self.throttle = s.spawn_throttle
        self.pitch_rate = self.roll_rate = 0.0
        self.sink = self.vertical_speed = 0.0
        self.state = FlightState.FLYING
        self.crash_reason = ""
        self.state_time = 0.0
        self.fuel = 1.0

    def park(self, x: float, y: float, heading: float, ground: Ground) -> None:
        """Stand still on the ground at (x, y), engine idle, ready to take off."""
        s = self.settings
        self.x, self.y = x, y
        self.forward, _, self.up = euler_to_basis(heading, 0.0, 0.0)
        self.altitude = ground(x, y) + s.gear_height
        self.speed = self.throttle = 0.0
        self.pitch_rate = self.roll_rate = 0.0
        self.sink = self.vertical_speed = 0.0
        self.state = FlightState.LANDED
        self.crash_reason = ""
        self.state_time = 0.0
        self.fuel = 1.0

    def update(
        self,
        controls: Controls,
        dt: float,
        ground: Ground,
        wind: tuple[float, float] = (0.0, 0.0),
        bump: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Advance one step. ``wind`` (x, y velocity) carries the plane along while
        it flies; on the ground the wheels hold it. ``bump`` is turbulence: extra
        pitch and roll rates (radians per second) on top of the stick's."""
        self.state_time += dt
        if self.state is FlightState.CRASHED:
            return
        s = self.settings
        on_ground = self.state is FlightState.LANDED

        self.throttle = _clamp(
            self.throttle + controls.throttle * s.throttle_rate * dt, 0, 1
        )
        if self.uses_fuel:
            self.fuel = max(0.0, self.fuel - self.burn_rate() * dt)
        if on_ground:
            self._steer_on_ground(controls, dt)
        else:
            self._steer_in_air(controls, dt, bump)
        self._accelerate(controls, dt, on_ground)

        # Lift has to hold up the part of the weight the flight path does not
        # carry (none in a vertical climb, all of it in level flight). Banking
        # tilts the lift away from vertical, upside down it pulls the wrong way
        # and a stall removes it; whatever is missing turns into sink.
        f = self.forward
        if on_ground:
            self.sink = 0.0
        else:
            deficit = math.hypot(f[0], f[1]) - self.lift * self.up[2]
            self.sink += (s.gravity * deficit - self.sink * s.sink_damping) * dt
            self.sink = max(self.sink, 0.0)

        drift = (0.0, 0.0) if on_ground else wind
        self.x += (f[0] * self.speed + drift[0]) * dt
        self.y += (f[1] * self.speed + drift[1]) * dt
        climb = self.speed * f[2] - self.sink
        if on_ground and self.speed < s.takeoff_speed:
            climb = min(climb, 0.0)
        self.vertical_speed = climb
        self.altitude += climb * dt

        self._touch_ground(ground)

    def _follow_stick(self, controls: Controls, dt: float, roll: bool) -> float:
        """Ease the pitch and roll rates towards the stick; returns control authority."""
        s = self.settings
        authority = _clamp(self.speed / s.control_speed, 0.25, 1)
        follow = min(1.0, dt / s.control_lag)
        self.pitch_rate += (
            controls.pitch * s.max_pitch_rate * authority - self.pitch_rate
        ) * follow
        target_roll = controls.roll * s.max_roll_rate * authority if roll else 0.0
        self.roll_rate += (target_roll - self.roll_rate) * follow
        return authority

    def _steer_on_ground(self, controls: Controls, dt: float) -> None:
        """Wings stay level; stick and rudder steer the nose wheel instead."""
        s = self.settings
        self._follow_stick(controls, dt, roll=False)
        self.roll_rate = 0.0
        steer = _clamp(controls.yaw + controls.roll, -1, 1)
        heading = (
            self.heading + steer * s.ground_steer_rate * min(1.0, self.speed / 10) * dt
        )
        pitch = _clamp(self.pitch + self.pitch_rate * dt, 0, s.max_ground_pitch)
        self.forward, _, self.up = euler_to_basis(heading, pitch, 0.0)

    def _steer_in_air(
        self, controls: Controls, dt: float, bump: tuple[float, float]
    ) -> None:
        s = self.settings
        authority = self._follow_stick(controls, dt, roll=True)
        f, u = self.forward, self.up

        # Pitch about the wings, roll about the nose, yaw about the canopy.
        f, u = _turn(f, u, (self.pitch_rate + bump[0]) * dt)
        right = np.cross(u, f)
        u, right = _turn(u, right, (self.roll_rate + bump[1]) * dt)
        f, right = _turn(f, right, controls.yaw * s.max_yaw_rate * authority * dt)

        # Help the wings back to level when the stick is let go, but only near
        # upright, so rolls and inverted flight are left alone.
        bank = math.atan2(-right[2], u[2])
        if controls.roll == 0 and abs(bank) < s.roll_leveling_limit:
            u, right = _turn(u, right, -bank * s.roll_leveling * dt)

        # Below stall speed the nose falls towards the ground.
        down = np.array([0.0, 0.0, -1.0])
        axis = np.cross(f, down)
        if np.linalg.norm(axis) > 1e-6:
            angle = s.stall_nose_drop * (1 - self.lift) * dt
            f, u = _rotate(f, axis, angle), _rotate(u, axis, angle)

        # The horizontal part of the lift swings the flight path round: a
        # banked plane turns without the rudder (the other way round when
        # upside down, since the lift then points the other way).
        horizontal = math.hypot(f[0], f[1])
        if horizontal > 0.05:
            level_right = np.array([-f[1], f[0], 0.0]) / horizontal
            turn = (
                s.gravity
                * s.turn_factor
                * float(u @ level_right)
                / max(self.speed, s.stall_speed)
                * self.lift
            )
            vertical = np.array([0.0, 0.0, 1.0])
            f, u = _rotate(f, vertical, turn * dt), _rotate(u, vertical, turn * dt)

        self.forward, self.up = _orthonormal(f, u)

    def _accelerate(self, controls: Controls, dt: float, on_ground: bool) -> None:
        s = self.settings
        density = math.exp(-max(self.altitude, 0.0) / s.scale_height)
        thrust = self.throttle if self.engine_running else 0.0
        accel = (
            thrust * s.max_thrust * density
            - s.drag * self.speed**2
            - s.gravity * self.forward[2]
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
            self.state_time = 0.0
            self.touchdown_sink = -self.vertical_speed
            self.touchdown_engine_out = not self.engine_running
            self.forward, _, self.up = euler_to_basis(
                self.heading, max(self.pitch, 0.0), 0.0
            )
            self.roll_rate = self.pitch_rate = 0.0
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
        roll = abs(self.roll)
        if roll > math.pi / 2:
            return "LANDED UPSIDE DOWN"
        if self.vertical_speed < -s.max_touchdown_sink:
            return "HIT THE GROUND TOO HARD"
        if roll > s.max_touchdown_roll:
            return "WINGTIP HIT THE GROUND"
        if self.pitch < s.min_touchdown_pitch:
            return "NOSED INTO THE GROUND"
        return ""

    def crash(self, reason: str) -> None:
        """End the flight here, e.g. on hitting something the model does not know about."""
        if self.state is not FlightState.CRASHED:
            self._crash(reason)

    def _crash(self, reason: str) -> None:
        self.state = FlightState.CRASHED
        self.crash_reason = reason
        self.state_time = 0.0
        self.speed = 0.0
        self.vertical_speed = 0.0
        self.pitch_rate = self.roll_rate = 0.0


def _turn(a: np.ndarray, b: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the perpendicular pair (a, b) by ``angle`` in their plane, a towards b."""
    c, s = math.cos(angle), math.sin(angle)
    return a * c + b * s, b * c - a * s


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``v`` about ``axis`` by ``angle`` (right-hand rule)."""
    k = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(k, v) * s + k * float(k @ v) * (1 - c)


def _orthonormal(f: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undo rounding drift: unit forward, and up made perpendicular to it."""
    f = f / np.linalg.norm(f)
    u = u - f * float(u @ f)
    return f, u / np.linalg.norm(u)


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
