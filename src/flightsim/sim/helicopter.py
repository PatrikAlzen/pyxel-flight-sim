"""A helicopter flight model with the help a keyboard pilot needs.

A real helicopter has to be flown every moment; this one helps:

- the stick sets how far the helicopter leans (and it levels itself when the
  stick is let go), instead of how fast it rolls and pitches;
- leaning tilts the rotor's lift, which pushes the helicopter that way, so it
  speeds up forwards, sideways or backwards, and air drag sets a top speed;
- the throttle keys ask for a climb or descent rate, and the collective (the
  rotor's pitch, shown in place of a throttle) is set to give it, so letting
  go hovers at the current height, and near the ground a descent slows down
  so holding it settles gently on the skids;
- the rudder keys turn the tail, and banking in forward flight also turns it
  a little, as a pilot would coordinate the turn.

It keeps the same outward shape as ``Aircraft`` (position, forward and up
vectors, speed, state, fuel, alerts, crashing), so the rest of the game, from
the chase camera to the replays, works with either. It lands on skids
anywhere flat enough; a steep slope, water, landing hard or tipped over, or
the rotor touching the ground ends the flight.

Units are world units and seconds; nothing here knows about Pyxel.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from flightsim.sim.aircraft import Aircraft, Controls, FlightState, Ground, _slope
from flightsim.sim.camera import euler_to_basis


@dataclass(frozen=True)
class HelicopterSettings:
    gravity: float = 30.0
    max_lift: float = 60.0  # at full collective and sea level: hover near 0.5
    scale_height: float = 1500.0  # lift falls off with altitude like thrust
    top_speed: float = 65.0
    max_tilt: float = math.radians(25)  # how far the stick leans it
    tilt_rate: float = 2.5  # per second the lean follows the stick
    yaw_rate: float = 1.1  # radians per second at full rudder
    turn_coupling: float = 0.6  # banking in forward flight turns it this much
    max_climb: float = 14.0  # climb or descent rate at full throttle input
    climb_follow: float = 2.0  # per second the collective chases the asked rate
    collective_rate: float = 1.5  # most the collective changes per second
    side_drag: float = 0.6  # per second: drift dies away when not leaning
    hover_brake: float = 0.5  # per second: with the stick centred it slows to a hover
    back_tilt: float = 0.6  # leaning back goes this much as far as leaning forward
    # Landing help: descent no faster than flare_sink + flare_slope * height.
    flare_sink: float = 1.5
    flare_slope: float = 0.35

    # Ground.
    gear_height: float = 1.5  # skids below the reference point
    max_touchdown_sink: float = 6.0
    max_touchdown_tilt: float = math.radians(15)
    max_ground_slope: float = 0.35
    rotor_radius: float = 4.5
    rotor_height: float = 2.0  # above the reference point
    spawn_height: float = 60.0

    # Fuel, as for aeroplanes (a tank lasts ~8 minutes at full collective).
    full_burn: float = 1 / 480
    idle_burn: float = 0.3
    refuel_time: float = 5.0
    cruise_throttle: float = 0.55

    # What the rest of the game asks of an aircraft's settings.
    stall_speed: float = 0.0  # it does not stall
    floats: bool = False
    sea_level: float = 0.0

    @property
    def drag(self) -> float:
        """Quadratic drag that stops the speed-up at full lean at ``top_speed``."""
        push = self.gravity * math.tan(self.max_tilt)
        return push / self.top_speed**2


@dataclass
class Helicopter(Aircraft):
    settings: HelicopterSettings = field(default_factory=HelicopterSettings)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lean: np.ndarray = field(default_factory=lambda: np.zeros(2))  # pitch, roll
    yaw: float = 0.0  # heading, radians clockwise from north

    # --- the shape the game knows ----------------------------------------

    @property
    def lift(self) -> float:
        return 1.0

    @property
    def stalling(self) -> bool:
        return False

    def range_estimate(self) -> float:
        """How far the fuel left takes it at cruise collective and top speed."""
        s = self.settings
        return s.top_speed * 0.8 * self.fuel / self.burn_rate(s.cruise_throttle)

    def _pose(self) -> None:
        self.forward, _, self.up = euler_to_basis(self.yaw, self.lean[0], self.lean[1])

    def respawn(self, x: float, y: float, heading: float, ground: Ground) -> None:
        """Hover above (x, y), level."""
        s = self.settings
        self.x, self.y = x, y
        self.yaw = heading
        self.lean = np.zeros(2)
        self._pose()
        self.altitude = max(ground(x, y), s.sea_level) + s.spawn_height
        self.velocity = np.zeros(3)
        self.speed = self.vertical_speed = 0.0
        self.throttle = self._hover_collective()
        self.state = FlightState.FLYING
        self.crash_reason = ""
        self.state_time = 0.0
        self.fuel = 1.0

    def park(self, x: float, y: float, heading: float, ground: Ground) -> None:
        """Stand on the skids at (x, y), rotor idling."""
        s = self.settings
        self.x, self.y = x, y
        self.yaw = heading
        self.lean = np.zeros(2)
        self._pose()
        self.altitude = ground(x, y) + s.gear_height
        self.velocity = np.zeros(3)
        self.speed = self.vertical_speed = 0.0
        self.throttle = 0.0
        self.state = FlightState.LANDED
        self.crash_reason = ""
        self.state_time = 0.0
        self.fuel = 1.0

    def _hover_collective(self) -> float:
        s = self.settings
        density = math.exp(-max(self.altitude, 0.0) / s.scale_height)
        return min(1.0, s.gravity / (s.max_lift * density))

    # --- flying ------------------------------------------------------------

    def update(
        self,
        controls: Controls,
        dt: float,
        ground: Ground,
        wind: tuple[float, float] = (0.0, 0.0),
        bump: tuple[float, float] = (0.0, 0.0),
        runway=None,
    ) -> None:
        """Advance one step; ``runway`` is unused (skids land anywhere)."""
        self.state_time += dt
        if self.state is FlightState.CRASHED:
            return
        s = self.settings
        on_ground = self.state is FlightState.LANDED
        if self.uses_fuel:
            self.fuel = max(0.0, self.fuel - self.burn_rate() * dt)

        self.yaw += controls.yaw * s.yaw_rate * dt
        if not on_ground:
            self._lean(controls, dt, bump)
        agl = self.altitude - s.gear_height - ground(self.x, self.y)
        self._collective(controls, dt, on_ground, agl)
        self._pose()

        # The rotor pushes along the helicopter's up; gravity and drag do
        # the rest. Wind carries it along (in the air) like the aeroplanes.
        density = math.exp(-max(self.altitude, 0.0) / s.scale_height)
        lift = self.throttle * s.max_lift * density if self.engine_running else 0.0
        accel = self.up * lift + np.array([0.0, 0.0, -s.gravity])
        air = self.velocity - np.array([wind[0], wind[1], 0.0])
        accel -= air * np.linalg.norm(air) * s.drag
        # The assist trims out drift over the ground (wind included):
        # sideways unless the stick asks for it, and all of it, holding the
        # spot, once the stick is let go.
        if not on_ground:
            ground_speed = np.array([self.velocity[0], self.velocity[1], 0.0])
            across = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
            sideways = across * float(ground_speed @ across)
            accel -= sideways * s.side_drag * (1.0 - min(1.0, abs(controls.roll)))
            if controls.pitch == 0 and controls.roll == 0:
                accel -= ground_speed * s.hover_brake
        self.velocity = self.velocity + accel * dt
        if on_ground:
            self.velocity[:2] = 0.0
            self.velocity[2] = max(self.velocity[2], 0.0)
        self.x += float(self.velocity[0]) * dt
        self.y += float(self.velocity[1]) * dt
        self.altitude += float(self.velocity[2]) * dt
        self.speed = float(np.linalg.norm(self.velocity))
        self.vertical_speed = float(self.velocity[2])
        self._touch_ground(ground)

    def _lean(self, controls: Controls, dt: float, bump: tuple[float, float]) -> None:
        """Ease the lean towards where the stick points; bank turns the tail."""
        s = self.settings
        # Pulling back (positive pitch input) raises the nose: it slows or backs.
        back = s.back_tilt if controls.pitch > 0 else 1.0
        target = np.array([controls.pitch * back, controls.roll]) * s.max_tilt
        follow = min(1.0, s.tilt_rate * dt)
        self.lean = self.lean + (target - self.lean) * follow
        self.lean = self.lean + np.array(bump) * dt
        forward_speed = float(self.velocity[:2] @ self.forward[:2])
        if forward_speed > 10.0:
            turn = s.gravity * math.tan(self.lean[1]) / forward_speed
            self.yaw += turn * s.turn_coupling * dt

    def _collective(
        self, controls: Controls, dt: float, on_ground: bool, agl: float
    ) -> None:
        """Set the collective for the climb rate the throttle keys ask for. Near
        the ground a descent is slowed so it settles gently on the skids."""
        s = self.settings
        wanted = controls.throttle * s.max_climb
        if wanted < 0:
            wanted = max(wanted, -(s.flare_sink + max(agl, 0.0) * s.flare_slope))
        if on_ground and wanted <= 0:
            self.throttle = max(0.0, self.throttle - s.collective_rate * dt)
            return
        density = math.exp(-max(self.altitude, 0.0) / s.scale_height)
        upright = max(float(self.up[2]), 0.2)
        accel = (wanted - float(self.velocity[2])) * s.climb_follow
        needed = (s.gravity + accel) / (s.max_lift * density * upright)
        step = s.collective_rate * dt
        self.throttle = min(
            1.0, max(0.0, self.throttle + max(-step, min(step, needed - self.throttle)))
        )

    # --- the ground ----------------------------------------------------------

    def _touch_ground(self, ground: Ground, runway=None) -> None:
        s = self.settings
        if self._rotor_strike(ground):
            self._crash("ROTOR STRIKE")
            return
        height = ground(self.x, self.y)
        floor = height + s.gear_height
        if self.altitude > floor:
            if self.state is FlightState.LANDED:
                self.state = FlightState.FLYING
                self.state_time = 0.0
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
            self.lean = np.zeros(2)
            self._pose()
        self.altitude = floor
        self.velocity = np.zeros(3)
        self.speed = self.vertical_speed = 0.0

    def _ground_problem(self, height: float, ground: Ground, runway=None) -> str:
        s = self.settings
        if height <= s.sea_level:
            return "DITCHED IN THE WATER"
        if self.state is FlightState.LANDED:
            return ""
        if _slope(ground, self.x, self.y, height) > s.max_ground_slope:
            return "SKIDS SLIPPED ON THE SLOPE"
        if float(self.up[2]) < 0:
            return "ROLLED OVER"
        if -self.vertical_speed > s.max_touchdown_sink:
            return "HIT THE GROUND TOO HARD"
        if math.acos(min(1.0, float(self.up[2]))) > s.max_touchdown_tilt:
            return "TIPPED OVER LANDING"
        return ""

    def _rotor_strike(self, ground: Ground) -> bool:
        """Whether the ground rises into the rotor disc around the mast."""
        s = self.settings
        hub = np.array([self.x, self.y, self.altitude]) + self.up * s.rotor_height
        right = np.cross(self.up, self.forward)
        for direction in (self.forward, -self.forward, right, -right):
            tip = hub + direction * s.rotor_radius
            if ground(float(tip[0]), float(tip[1])) > tip[2]:
                return True
        return False
