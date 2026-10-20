"""Chase camera: follows the aircraft from behind with a little lag."""

import math
from dataclasses import dataclass

from flightsim.sim.aircraft import Aircraft
from flightsim.sim.camera import Camera


@dataclass(frozen=True)
class ChaseSettings:
    distance: float = 26.0  # behind the plane, along its flight path
    height: float = 6.0  # above it
    look_down: float = math.radians(8)
    heading_lag: float = 4.0  # per second; higher follows turns more tightly
    roll_share: float = 0.4  # how much of the plane's bank the camera copies
    pitch_share: float = 0.6
    clearance: float = 3.0  # minimum camera height above the ground


class ChaseCamera:
    def __init__(self, settings: ChaseSettings | None = None) -> None:
        self.settings = settings or ChaseSettings()
        self.heading: float | None = None

    def reset(self) -> None:
        """Snap straight behind the plane on the next update (e.g. after a respawn)."""
        self.heading = None

    def update(self, plane: Aircraft, camera: Camera, dt: float, ground) -> None:
        s = self.settings
        if self.heading is None:
            self.heading = plane.heading
        # Ease the heading towards the plane's along the shortest way round.
        diff = (plane.heading - self.heading + math.pi) % math.tau - math.pi
        self.heading = (self.heading + diff * min(1.0, s.heading_lag * dt)) % math.tau

        fx, fy = math.sin(self.heading), -math.cos(self.heading)
        back = s.distance * math.cos(plane.pitch)
        camera.x = plane.x - fx * back
        camera.y = plane.y - fy * back
        camera.altitude = plane.altitude + s.height - s.distance * math.sin(plane.pitch)
        camera.altitude = max(camera.altitude, ground(camera.x, camera.y) + s.clearance)
        camera.heading = self.heading
        camera.pitch = plane.pitch * s.pitch_share - s.look_down
        camera.roll = plane.roll * s.roll_share


def project(
    camera: Camera, focal: float, width: int, height: int, x: float, y: float, z: float
):
    """Screen position of world point (x, y, altitude z), or None if behind the camera.

    Uses the same model as the voxel renderer: depth along the view direction,
    horizon moved by pitch and sheared by roll.
    """
    dx, dy = x - camera.x, y - camera.y
    fx, fy = camera.forward
    rx, ry = camera.right
    depth = dx * fx + dy * fy
    if depth <= 1.0:
        return None
    lateral = dx * rx + dy * ry
    sx = width / 2 + lateral / depth * focal
    horizon = (
        height / 2
        + math.tan(camera.pitch) * focal
        - (sx - width / 2) * math.tan(camera.roll)
    )
    sy = horizon + (camera.altitude - z) / depth * focal
    return sx, sy, depth
