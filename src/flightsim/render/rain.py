"""Rain as drops in the world around the camera, drawn as short streaks.

The drops live in a box that wraps around the camera, so there are always
drops nearby however far the plane flies. They fall and drift with the wind
in world space and are projected like everything else, so they keep falling
"down" in the world when the plane pitches, rolls or loops. Each streak is the
path the drop moves relative to the camera over a short time: drops hang
almost still to a parked plane and rush past a fast one.
"""

from dataclasses import dataclass

import numpy as np
import pyxel

from flightsim.render.effects import in_front_of_terrain, pixel_scale, project_points
from flightsim.render.voxel import VoxelRenderer
from flightsim.sim.camera import Camera

COLORS = (pyxel.COLOR_GRAY, pyxel.COLOR_LIGHT_BLUE)


@dataclass(frozen=True)
class RainSettings:
    drops: int = 500  # at full rain
    box: tuple[float, float, float] = (70.0, 70.0, 50.0)  # wraps around the camera
    fall: float = 40.0  # world units per second
    streak_time: float = 0.03  # seconds of relative motion shown per streak
    min_streak: float = 2.0  # pixels, so drops falling straight at you show
    max_streak: float = 16.0  # pixels
    near: float = 3.0  # drops closer than this are skipped (they would be huge)
    max_step: float = 60.0  # camera moves further than this in a frame: a jump


class Rain:
    def __init__(self, settings: RainSettings | None = None) -> None:
        self.settings = s = settings or RainSettings()
        rng = np.random.default_rng(7)
        self.box = np.array(s.box)
        self.position = rng.uniform(0, 1, size=(s.drops, 3)) * self.box
        self.fall = s.fall * rng.uniform(0.8, 1.2, size=s.drops)  # some fall faster
        self.wind = np.zeros(3)
        self.camera_velocity = np.zeros(3)
        self.last_camera: np.ndarray | None = None

    def update(self, dt: float, wind: tuple[float, float], camera: Camera) -> None:
        """Let the drops fall and note how fast the camera moves through them."""
        self.wind = np.array([wind[0], wind[1], 0.0])
        self.position += self.wind * dt
        self.position[:, 2] -= self.fall * dt
        here = np.array(camera.position, dtype=float)
        if self.last_camera is not None:
            step = here - self.last_camera
            moved = float(np.linalg.norm(step))
            self.camera_velocity = (
                step / dt if moved < self.settings.max_step else np.zeros(3)
            )
        self.last_camera = here

    def segments(
        self,
        amount: float,
        camera: Camera,
        focal: float,
        width: int,
        height: int,
        pixel: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Streaks (n, 4: x0, y0, x1, y1) for rain ``amount`` 0..1, and their
        drops' world positions (for hiding them behind terrain). Streak
        lengths are in pixels times ``pixel`` (the screen's scale)."""
        s = self.settings
        shortest, longest = s.min_streak * pixel, s.max_streak * pixel
        n = round(s.drops * min(max(amount, 0.0), 1.0))
        if n == 0:
            return np.zeros((0, 4)), np.zeros((0, 3))
        here = np.array(camera.position, dtype=float)
        half = self.box / 2
        drops = here + (self.position[:n] - here + half) % self.box - half
        velocity = self.wind - self.camera_velocity
        velocity = np.repeat(velocity[None, :], n, axis=0)
        velocity[:, 2] -= self.fall[:n]
        tails = drops - velocity * s.streak_time
        hx, hy, hd = project_points(camera, drops, focal, width, height)
        tx, ty, td = project_points(camera, tails, focal, width, height)
        keep = (hd > s.near) & (td > s.near)
        keep &= (hx > -longest) & (hx < width + longest)
        keep &= (hy > -longest) & (hy < height + longest)
        hx, hy, tx, ty, drops = hx[keep], hy[keep], tx[keep], ty[keep], drops[keep]

        # Keep the streak length within limits, measured back from the drop.
        dx, dy = tx - hx, ty - hy
        length = np.hypot(dx, dy)
        safe = np.where(length > 0, length, 1.0)
        target = np.clip(length, shortest, longest)
        ux = np.where(length > 0, dx / safe, 0.0)
        uy = np.where(length > 0, dy / safe, -1.0)  # no motion: a short dash
        tx, ty = hx + ux * target, hy + uy * target
        return np.column_stack([hx, hy, tx, ty]), drops

    def draw(self, amount: float, camera: Camera, renderer: VoxelRenderer) -> None:
        w, h = renderer.width, renderer.height
        segments, drops = self.segments(
            amount, camera, renderer.focal, w, h, pixel_scale(renderer)
        )
        if not len(segments):
            return
        seen = in_front_of_terrain(
            renderer, camera, drops, segments[:, 0], segments[:, 1]
        )
        pyxel.dither(0.6)
        for i, (x0, y0, x1, y1) in enumerate(segments[seen]):
            pyxel.line(x0, y0, x1, y1, COLORS[i % 2])
        pyxel.dither(1.0)
