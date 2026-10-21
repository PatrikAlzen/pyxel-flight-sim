"""A camera the player moves freely: circling a point, or flying on its own.

Used by the photo mode. Orbiting keeps the target in the middle of the view
at any distance and angle; flying turns and moves the camera like the ghost
camera. Either way the camera stays above the ground and within reach of an
anchor point, so the terrain around it is already generated.
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from flightsim.sim.camera import Camera

MAX_PITCH = math.radians(85)


class FreeMode(Enum):
    ORBIT = "ORBIT"
    FLY = "FLY"


@dataclass(frozen=True)
class FreeCameraSettings:
    turn_rate: float = 1.2  # radians per second at full input
    zoom_rate: float = 1.0  # orbit distance changes by e**rate per second
    min_distance: float = 8.0
    start_distance: float = 25.0  # when starting from (nearly) the target itself
    max_distance: float = 300.0
    fly_speed: float = 40.0  # world units per second
    fast: float = 4.0  # speed multiplier while hurrying
    max_range: float = 400.0  # horizontal distance from the anchor
    clearance: float = 3.0  # minimum height above the ground


class FreeCamera:
    def __init__(
        self,
        camera: Camera,
        target: np.ndarray,
        settings: FreeCameraSettings | None = None,
    ) -> None:
        """Start from where ``camera`` is, circling ``target`` (also the anchor)."""
        self.settings = s = settings or FreeCameraSettings()
        self.camera = camera
        self.target = np.asarray(target, dtype=float)
        self.anchor = self.target.copy()
        self.mode = FreeMode.ORBIT
        offset = self.target - np.array(camera.position, dtype=float)
        horizontal = math.hypot(offset[0], offset[1])
        if np.linalg.norm(offset) < s.start_distance:
            # From the cockpit: step back behind and above, looking the same way.
            self.distance = s.start_distance
            self.yaw, self.pitch = camera.heading, -0.3
        else:
            self.distance = min(float(np.linalg.norm(offset)), 60.0)
            # Orbit angles: the heading the camera looks along and its pitch.
            self.yaw = math.atan2(offset[0], -offset[1])
            self.pitch = math.atan2(offset[2], max(horizontal, 1e-6))

    def toggle_mode(self) -> None:
        """Switch between orbiting and flying, keeping the current view."""
        cam = self.camera
        if self.mode is FreeMode.ORBIT:
            self.mode = FreeMode.FLY
            self.yaw, self.pitch = cam.heading, cam.pitch
        else:
            self.mode = FreeMode.ORBIT
            # Circle whatever is straight ahead at the orbit distance.
            self.target = np.array(cam.position, dtype=float) + self._look() * (
                self.distance
            )

    def orbit(self, turn: float, tilt: float, zoom: float, dt: float, ground) -> None:
        """``turn`` circles, ``tilt`` raises or lowers the view, ``zoom`` > 0 closes in."""
        s = self.settings
        self.yaw = (self.yaw + turn * s.turn_rate * dt) % math.tau
        self.pitch = _clamp(self.pitch - tilt * s.turn_rate * dt, -MAX_PITCH, 0.3)
        self.distance = _clamp(
            self.distance * math.exp(-zoom * s.zoom_rate * dt),
            s.min_distance,
            s.max_distance,
        )
        position = self.target - self._look() * self.distance
        self._place(position, ground)
        # Pushed up by a hill: look back down at the target.
        cam = self.camera
        offset = self.target - np.array(cam.position, dtype=float)
        cam.heading = self.yaw
        cam.pitch = math.atan2(offset[2], max(math.hypot(offset[0], offset[1]), 1e-6))
        cam.roll = 0.0

    def fly(
        self,
        turn: float,
        tilt: float,
        ahead: float,
        side: float,
        rise: float,
        dt: float,
        ground,
        fast: bool = False,
    ) -> None:
        """Turn and move like a drone: ``ahead`` along the view, ``side`` to the
        right and ``rise`` straight up, each -1..1."""
        s = self.settings
        self.yaw = (self.yaw + turn * s.turn_rate * dt) % math.tau
        self.pitch = _clamp(self.pitch - tilt * s.turn_rate * dt, -MAX_PITCH, MAX_PITCH)
        speed = s.fly_speed * (s.fast if fast else 1.0) * dt
        right = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        move = self._look() * ahead + right * side + np.array([0.0, 0.0, rise])
        self._place(np.array(self.camera.position, dtype=float) + move * speed, ground)
        cam = self.camera
        cam.heading, cam.pitch, cam.roll = self.yaw, self.pitch, 0.0

    def _look(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return np.array(
            [math.sin(self.yaw) * cp, -math.cos(self.yaw) * cp, math.sin(self.pitch)]
        )

    def _place(self, position: np.ndarray, ground) -> None:
        """Move the camera to ``position``, kept in range and above the ground."""
        s = self.settings
        away = position[:2] - self.anchor[:2]
        reach = float(np.hypot(*away))
        if reach > s.max_range:
            position[:2] = self.anchor[:2] + away * (s.max_range / reach)
        floor = ground(float(position[0]), float(position[1])) + s.clearance
        cam = self.camera
        cam.x, cam.y = float(position[0]), float(position[1])
        cam.altitude = max(float(position[2]), floor)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def look_at(camera: Camera, target: np.ndarray) -> None:
    """Turn ``camera`` (wings level) to look straight at ``target``."""
    offset = np.asarray(target, dtype=float) - np.array(camera.position, dtype=float)
    horizontal = math.hypot(offset[0], offset[1])
    if horizontal > 1e-6:
        camera.heading = math.atan2(offset[0], -offset[1]) % math.tau
    camera.pitch = _clamp(math.atan2(offset[2], horizontal), -MAX_PITCH, MAX_PITCH)
    camera.roll = 0.0
