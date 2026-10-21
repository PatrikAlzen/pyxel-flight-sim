"""A layer of flat clouds at a fixed altitude, drawn over the terrain view.

Each screen pixel's ray is intersected with the cloud plane using the angles
the terrain renderer already computed: a ray with slope s (tangent of its
elevation angle) reaches the plane after the horizontal distance ``(cloud
altitude - camera altitude) / s``, if that is positive. That works from below, from above and upside
down. Comparing the distance with the terrain distance in the same pixel
hides clouds behind mountains and lets high peaks poke through.
"""

from dataclasses import dataclass

import numpy as np

from flightsim.render import palette
from flightsim.sim.camera import Camera
from flightsim.terrain.biomes import Biome
from flightsim.terrain.noise import FractalNoise

_UPSAMPLE = 4  # fine opacity texels per density texel along each side


@dataclass(frozen=True)
class CloudSettings:
    altitude: float = 380.0  # the highest peaks poke through
    cell: float = 24.0  # world units per texel of the cloud texture
    texture_size: int = 256  # texels per side; the pattern repeats after this
    coverage: float = 0.35  # fraction of the sky covered
    softness: float = 0.25  # density range over which edges fade (dithered)
    # Clouds stay visible further than terrain: this many times z_far, with
    # the fog stretched to match.
    range_factor: float = 1.6


class CloudLayer:
    """The cloud layer. ``coverage`` and ``altitude`` can change at any time (the
    weather sets them every frame), and ``advance`` moves the clouds with the wind.
    """

    def __init__(self, seed: int, settings: CloudSettings | None = None) -> None:
        self.settings = s = settings or CloudSettings()
        self.density = _tileable_noise(seed, s.texture_size)
        self.mask = s.texture_size - 1
        self.coverage = s.coverage
        self.altitude = s.altitude
        self.offset = np.zeros(2)  # how far the wind has carried the clouds
        # Sorted densities turn a coverage into a threshold without a search.
        self._sorted = np.sort(self.density.ravel())

        # Density bilinearly upsampled once to a finer grid, so drawing needs a
        # single lookup per pixel instead of four plus interpolation. The
        # threshold is applied when drawing, so coverage can change freely.
        fine = s.texture_size * _UPSAMPLE
        self.fine_cell = s.cell / _UPSAMPLE
        self.fine_mask = fine - 1
        coords = np.arange(fine) * self.fine_cell
        self.fine_density = self._filtered(coords[None, :], coords[:, None]).astype(
            np.float32
        )

    @property
    def threshold(self) -> float:
        """Density above which a texel is cloud, giving the current coverage."""
        coverage = min(max(self.coverage, 0.0), 1.0)
        return float(self._sorted[round((1 - coverage) * (len(self._sorted) - 1))])

    def advance(self, wind: tuple[float, float], dt: float) -> None:
        self.offset += np.asarray(wind) * dt

    def coverage_at(
        self, x: np.ndarray, y: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """Cloud opacity in [0, 1] at world positions, where the wind has taken them."""
        iu = np.floor((x - self.offset[0]) / self.fine_cell).astype(np.int64)
        iv = np.floor((y - self.offset[1]) / self.fine_cell).astype(np.int64)
        density = self.fine_density[iv & self.fine_mask, iu & self.fine_mask]
        return np.clip((density - self.threshold) / self.settings.softness, 0.0, 1.0)

    def inside(
        self, x: float, y: float, altitude: float, thickness: float = 15.0
    ) -> bool:
        """Whether a point is in the cloud layer, inside a cloud."""
        if abs(altitude - self.altitude) > thickness:
            return False
        return bool(self.coverage_at(np.array(x), np.array(y)) > 0.5)

    def _filtered(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Density at texture-space positions, bilinearly filtered."""
        s = self.settings
        u = x / s.cell
        v = y / s.cell
        u0, v0 = np.floor(u), np.floor(v)
        fu, fv = u - u0, v - v0
        iu, iv = u0.astype(np.int64) & self.mask, v0.astype(np.int64) & self.mask
        ju, jv = (iu + 1) & self.mask, (iv + 1) & self.mask
        d = self.density
        top = d[iv, iu] + (d[iv, ju] - d[iv, iu]) * fu
        bottom = d[jv, iu] + (d[jv, ju] - d[jv, iu]) * fu
        return top + (bottom - top) * fv

    def draw(self, image: np.ndarray, renderer, camera: Camera, time: float) -> None:
        """Composite clouds into ``image`` (palette indices) after terrain and sky."""
        s = self.settings
        height = camera.altitude - self.altitude  # > 0 when flying above the clouds
        if abs(height) < 1e-3:
            return
        with np.errstate(divide="ignore"):
            z = -height / renderer.pixel_slope
        z_far = renderer.settings.z_far * s.range_factor
        visible = (z > renderer.settings.z_near) & (z < z_far)
        visible &= z < renderer.pixel_distance
        rows, cols = np.nonzero(visible)
        if rows.size == 0:
            return

        z = z[rows, cols]
        azimuth = renderer.pixel_azimuth[rows, cols]
        opacity = self.coverage_at(
            camera.x + z * np.sin(azimuth), camera.y - z * np.cos(azimuth), time
        )
        # Thick cloud hides trees, smoke and the like behind it: overlays test
        # against pixel_distance. A fixed threshold rather than the dither, so
        # a sprite at a cloud's thin edge does not blink as the view moves.
        thick = opacity > 0.5
        renderer.pixel_distance[rows[thick], cols[thick]] = z[thick]
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
