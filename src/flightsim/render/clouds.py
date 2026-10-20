"""A layer of flat clouds at a fixed altitude, drawn over the terrain view.

Each screen pixel's ray is intersected with the cloud plane using the same
projection as the terrain renderer: a point at depth z and height h appears
``(camera altitude - h) / z * focal`` pixels below the horizon, so a pixel
``dy`` below the horizon meets the plane at ``z = (camera altitude - cloud
altitude) * focal / dy``. That works from below (rows above the horizon) and
from above (rows below it). Comparing z with the terrain depth in the same
pixel hides clouds behind mountains and lets high peaks poke through.
"""

from dataclasses import dataclass

import numpy as np

from flightsim.render import palette
from flightsim.sim.camera import Camera
from flightsim.terrain.biomes import Biome
from flightsim.terrain.noise import FractalNoise


@dataclass(frozen=True)
class CloudSettings:
    altitude: float = 380.0  # the highest peaks poke through
    cell: float = 24.0  # world units per texel of the cloud texture
    texture_size: int = 256  # texels per side; the pattern repeats after this
    coverage: float = 0.35  # fraction of the sky covered
    softness: float = 0.25  # density range over which edges fade (dithered)
    drift: tuple[float, float] = (6.0, 2.0)  # wind, world units per second
    # Clouds stay visible further than terrain: this many times z_far, with
    # the fog stretched to match.
    range_factor: float = 1.6


class CloudLayer:
    def __init__(self, seed: int, settings: CloudSettings | None = None) -> None:
        self.settings = s = settings or CloudSettings()
        self.density = _tileable_noise(seed, s.texture_size)
        self.mask = s.texture_size - 1
        # Density above which a texel is cloud, chosen to give the coverage.
        self.threshold = float(np.quantile(self.density, 1 - s.coverage))

    def coverage_at(self, x: np.ndarray, y: np.ndarray, time: float) -> np.ndarray:
        """Cloud opacity in [0, 1] at world positions, bilinearly filtered."""
        s = self.settings
        u = (x + s.drift[0] * time) / s.cell
        v = (y + s.drift[1] * time) / s.cell
        u0, v0 = np.floor(u), np.floor(v)
        fu, fv = u - u0, v - v0
        iu, iv = u0.astype(np.int64) & self.mask, v0.astype(np.int64) & self.mask
        ju, jv = (iu + 1) & self.mask, (iv + 1) & self.mask
        d = self.density
        top = d[iv, iu] + (d[iv, ju] - d[iv, iu]) * fu
        bottom = d[jv, iu] + (d[jv, ju] - d[jv, iu]) * fu
        density = top + (bottom - top) * fv
        return np.clip((density - self.threshold) / s.softness, 0.0, 1.0)

    def draw(self, image: np.ndarray, renderer, camera: Camera, time: float) -> None:
        """Composite clouds into ``image`` (palette indices) after terrain and sky."""
        s = self.settings
        height = camera.altitude - s.altitude  # > 0 when flying above the clouds
        if abs(height) < 1e-3:
            return
        dy = renderer._rows - renderer.horizon[None, :]
        with np.errstate(divide="ignore"):
            z = height * renderer.focal / dy
        z_far = renderer.settings.z_far * s.range_factor
        visible = (z > renderer.settings.z_near) & (z < z_far)
        visible &= z < renderer.pixel_depth
        rows, cols = np.nonzero(visible)
        if rows.size == 0:
            return

        z = z[rows, cols]
        fx, fy = camera.forward
        rx, ry = camera.right
        lateral = z * renderer.spread[cols]
        opacity = self.coverage_at(
            camera.x + z * fx + lateral * rx, camera.y + z * fy + lateral * ry, time
        )
        dither = renderer.dither[rows, cols]
        drawn = opacity > dither
        if not drawn.any():
            return
        rows, cols, z, opacity, dither = (
            a[drawn] for a in (rows, cols, z, opacity, dither)
        )

        # Sunlit tops seen from above; greyer, thicker-looking bellies from below.
        if height > 0:
            shade = np.where(opacity > 0.6, palette.SHADES - 1, palette.SHADES - 2)
        else:
            shade = np.where(opacity > 0.6, palette.SHADES - 3, palette.SHADES - 2)
        color = palette.biome_color(Biome.SNOW, shade).astype(np.int32)
        fog = np.floor(renderer.fog_for(z / s.range_factor) + dither).astype(np.int32)
        color = np.where(
            fog >= palette.FOG_LEVELS,
            palette.SKY_BASE,
            color + fog * palette.TERRAIN_COLORS,
        )
        image[rows, cols] = color


def _tileable_noise(seed: int, size: int) -> np.ndarray:
    """Fractal noise on a ``size`` x ``size`` grid that wraps seamlessly.

    Blends four shifted copies of the noise so the left edge matches the
    right one and the top matches the bottom.
    """
    noise = FractalNoise(seed + 911, octaves=5)
    scale = 8 / size  # about eight blobs across the tile at the lowest octave
    t = np.arange(size) / size
    x, y = t[None, :] * size, t[:, None] * size

    def at(dx: float, dy: float) -> np.ndarray:
        return noise.fbm((x - dx) * scale, (y - dy) * scale)

    wx, wy = t[None, :], t[:, None]
    return (
        at(0, 0) * (1 - wx) * (1 - wy)
        + at(size, 0) * wx * (1 - wy)
        + at(0, size) * (1 - wx) * wy
        + at(size, size) * wx * wy
    ).astype(np.float32)
