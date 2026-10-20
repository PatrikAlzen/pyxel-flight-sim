"""Chase camera: follows the aircraft from behind with a little lag.

The camera keeps its own forward and up vectors and eases them towards the
plane's, so it follows through loops and rolls without any special cases. It
sits behind and above the plane in its own frame, which keeps the plane at the
same spot on screen; the lag shows up as the plane rotating against the view.
"""

import math
from dataclasses import dataclass

import numpy as np

from flightsim.sim.aircraft import Aircraft
from flightsim.sim.camera import Camera


@dataclass(frozen=True)
class ChaseSettings:
    distance: float = 26.0  # behind the plane
    height: float = 6.0  # above it, in the camera's own up direction
    look_down: float = math.radians(8)
    lag: float = 4.0  # per second; higher follows the plane more tightly
    clearance: float = 3.0  # minimum camera height above the ground


class ChaseCamera:
    def __init__(self, settings: ChaseSettings | None = None) -> None:
        self.settings = settings or ChaseSettings()
        self.forward: np.ndarray | None = None
        self.up: np.ndarray | None = None

    def reset(self) -> None:
        """Snap straight behind the plane on the next update (e.g. after a respawn)."""
        self.forward = self.up = None

    def update(self, plane: Aircraft, camera: Camera, dt: float, ground) -> None:
        s = self.settings
        # Aim slightly below the plane's nose so the plane sits low in the view.
        c, si = math.cos(s.look_down), math.sin(s.look_down)
        target_f = plane.forward * c - plane.up * si
        target_u = plane.up * c + plane.forward * si
        if self.forward is None:
            self.forward, self.up = target_f, target_u
        else:
            k = min(1.0, s.lag * dt)
            f = self.forward + (target_f - self.forward) * k
            u = self.up + (target_u - self.up) * k
            f /= np.linalg.norm(f)
            u -= f * float(u @ f)
            self.forward, self.up = f, u / np.linalg.norm(u)

        position = (
            np.array([plane.x, plane.y, plane.altitude])
            - self.forward * s.distance
            + self.up * s.height
        )
        camera.x, camera.y = float(position[0]), float(position[1])
        camera.altitude = max(
            float(position[2]), ground(camera.x, camera.y) + s.clearance
        )
        camera.set_vectors(self.forward, self.up)


def sprite_angle(plane: Aircraft, camera: Camera) -> float:
    """Degrees to rotate the plane sprite (clockwise) to match its bank in the view."""
    _, right, up = camera.basis()
    return math.degrees(math.atan2(float(plane.up @ right), float(plane.up @ up)))
