"""Camera position and orientation in the world.

World axes: x grows east, y grows south, z (altitude) is up.

Orientation is stored as Euler angles applied in the order heading, pitch,
roll. Together they can describe any orientation, including upside down and
pointing straight up or down; ``basis`` turns them into the camera's
forward/right/up vectors and ``set_vectors`` goes back the other way.
"""

import math
from dataclasses import dataclass

import numpy as np

Vector = np.ndarray  # shape (3,)


@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    altitude: float = 150.0
    heading: float = 0.0  # radians clockwise from north (-y)
    pitch: float = 0.0  # radians in [-pi/2, pi/2], positive = looking up
    roll: float = 0.0  # radians, positive = right wing down; pi = upside down

    @property
    def forward(self) -> tuple[float, float]:
        """Unit vector along the ground in the compass direction of ``heading``."""
        return math.sin(self.heading), -math.cos(self.heading)

    @property
    def right(self) -> tuple[float, float]:
        """Unit vector along the ground to the right of ``heading``."""
        return math.cos(self.heading), math.sin(self.heading)

    @property
    def heading_degrees(self) -> float:
        """Compass heading in [0, 360)."""
        return math.degrees(self.heading) % 360

    @property
    def position(self) -> Vector:
        return np.array([self.x, self.y, self.altitude])

    def basis(self) -> tuple[Vector, Vector, Vector]:
        """Unit (forward, right, up) vectors of the view."""
        return euler_to_basis(self.heading, self.pitch, self.roll)

    def set_vectors(self, forward: Vector, up: Vector) -> None:
        """Point the camera along ``forward`` with ``up`` as the top of the view."""
        self.heading, self.pitch, self.roll = basis_to_euler(forward, up)

    def project(self, point: Vector, focal: float, width: int, height: int):
        """Screen (x, y, depth) of a world point, or None if it is behind the camera."""
        return self.project_direction(point - self.position, focal, width, height)

    def project_direction(
        self, direction: Vector, focal: float, width: int, height: int
    ):
        """Screen (x, y, depth) of a direction (or offset) from the camera, or None."""
        f, r, u = self.basis()
        depth = float(direction @ f)
        if depth <= 1e-6:
            return None
        sx = width / 2 + float(direction @ r) / depth * focal
        sy = height / 2 - float(direction @ u) / depth * focal
        return sx, sy, depth


def euler_to_basis(
    heading: float, pitch: float, roll: float
) -> tuple[Vector, Vector, Vector]:
    sh, ch = math.sin(heading), math.cos(heading)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    forward = np.array([sh * cp, -ch * cp, sp])
    level_right = np.array([ch, sh, 0.0])
    level_up = np.array([-sh * sp, ch * sp, cp])
    right = level_right * cr - level_up * sr
    up = level_up * cr + level_right * sr
    return forward, right, up


def basis_to_euler(forward: Vector, up: Vector) -> tuple[float, float, float]:
    """Heading, pitch and roll of an orientation given by its forward and up vectors."""
    f = forward / np.linalg.norm(forward)
    pitch = math.asin(max(-1.0, min(1.0, float(f[2]))))
    right = np.cross(up, f)
    level_right = np.array([-f[1], f[0], 0.0])  # world up x forward
    if np.linalg.norm(level_right) < 1e-9:
        # Nose straight up or down: heading is wherever the wings point, no roll.
        level_right = np.array([right[0], right[1], 0.0])
    level_right /= np.linalg.norm(level_right)
    heading = math.atan2(level_right[1], level_right[0]) % math.tau
    level_up = np.cross(f, level_right)
    roll = math.atan2(-float(right @ level_up), float(right @ level_right))
    return heading, pitch, roll
