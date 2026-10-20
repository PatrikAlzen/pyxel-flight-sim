"""Camera position and orientation in the world."""

import math
from dataclasses import dataclass


@dataclass
class Camera:
    """A viewpoint in world space.

    World axes: x grows east, y grows south, altitude is up. ``heading`` is in
    radians clockwise from north, so 0 looks towards -y and pi/2 towards +x.
    """

    x: float = 0.0
    y: float = 0.0
    altitude: float = 150.0
    heading: float = 0.0
    pitch: float = 0.0  # radians, positive = looking up
    roll: float = 0.0  # radians, positive = right wing down

    @property
    def forward(self) -> tuple[float, float]:
        """Unit vector along the ground in the viewing direction."""
        return math.sin(self.heading), -math.cos(self.heading)

    @property
    def right(self) -> tuple[float, float]:
        """Unit vector along the ground to the right of the viewing direction."""
        return math.cos(self.heading), math.sin(self.heading)

    @property
    def heading_degrees(self) -> float:
        """Compass heading in [0, 360)."""
        return math.degrees(self.heading) % 360
