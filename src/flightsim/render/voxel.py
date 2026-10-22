"""Voxel-style terrain renderer for any camera orientation, vectorised with NumPy.

The classic Voxel Space algorithm draws one vertical slice of the world per
screen column, which only works while the camera is upright: pitch and roll
have to be faked by shifting and shearing the image. This renderer works in
angles instead, so the camera can roll all the way round or fly upside down:

1. Every pixel's view ray is turned into two angles: its compass direction
   (azimuth) and how far above or below the horizontal it points (elevation).
2. Terrain is sampled outwards from the camera along a set of azimuths. For
   each sample, the elevation angle at which it is seen is the angle from the
   camera to the top of that terrain column.
3. Along each azimuth, a running maximum of that angle is the skyline as seen
   so far. A pixel shows the nearest sample whose skyline reaches the pixel's
   elevation, or sky if none does. The skyline is monotonic per azimuth, so a
   single global ``searchsorted`` finds this for every pixel at once.

Only the camera's rotation decides which angles each pixel gets; the terrain
side never needs to know about pitch or roll. Elevations are handled as their
tangent, the slope (height difference / horizontal distance): it sorts the
same way as the angle and costs a division instead of an arctangent.

Each azimuth is only traced as far as its pixels can possibly see: a ray
pointing down reaches sea level within ``altitude / tan(-elevation)``, and one
pointing up passes above the highest peak within ``(peak - altitude) /
tan(elevation)``. Azimuths are grouped into a few length classes so the work
stays in rectangular arrays. This keeps views straight up or down, where every
azimuth is on screen, about as cheap as looking at the horizon.

Distant samples read from coarser terrain levels, which both extends the view
distance and avoids shimmering from undersampling fine detail far away.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from flightsim.render import palette
from flightsim.sim.camera import Camera
from flightsim.terrain.biomes import Biome
from flightsim.terrain.noise import smoothstep
from flightsim.terrain.world import TerrainLevel

if TYPE_CHECKING:
    from flightsim.render.clouds import CloudLayer

# Water biomes come first in the palette, so water colours are a single range.
WATER_BIOMES = 2
assert Biome.DEEP_WATER == 0 and Biome.WATER == 1

# 4x4 ordered-dither thresholds in [0, 1).
_BAYER = (
    np.array(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32
    )
    + 0.5
) / 16

# Slopes are clamped to +-_MAX_SLOPE (about 89.9 degrees) and the keys of
# different azimuths are spaced further apart than that range, so they never
# overlap.
_MAX_SLOPE = 1000.0
_KEY_STRIDE = 4 * _MAX_SLOPE


@dataclass(frozen=True)
class RenderSettings:
    fov_degrees: float = 90.0
    # Horizontal distance of the first and last terrain samples.
    z_near: float = 1.0
    z_far: float = 2500.0
    # Sample spacing starts at dz_start and grows by dz_growth per step:
    # dense close to the camera, sparse far away.
    dz_start: float = 0.5
    dz_growth: float = 1.02
    # Fog starts at this fraction of z_far and is complete at z_far.
    fog_start: float = 0.25
    # The sky gradient spans this many degrees above the horizon.
    sky_span_degrees: float = 40.0
    # Azimuth resolution: directions sampled around the full circle. About two
    # per pixel at the screen centre, so the edges are not undersampled.
    azimuths: int = 1600
    # Pixels near the point straight above or below the camera share coarser
    # azimuths: a pixel rho pixels from it needs about 2 pi rho of them around
    # the circle, times this factor.
    azimuths_per_pixel: float = 1.0
    azimuth_tiers: int = 5  # full resolution and this many halvings minus one
    # No terrain surface is lower than this (sea level). Looking down, rays
    # cannot travel further than it allows, so fewer samples are needed.
    lowest_surface: float = 0.0
    # Added to the highest height found in the terrain levels, since peaks can
    # fall between the samples of the coarse level used to find it.
    peak_margin: float = 40.0
    # Water waves: shade shift amplitude (in shade steps) and wave speed.
    wave_strength: float = 0.7
    wave_speed: float = 1.5


class VoxelRenderer:
    def __init__(
        self, width: int, height: int, settings: RenderSettings | None = None
    ) -> None:
        self.width = width
        self.height = height
        self.settings = s = settings or RenderSettings()

        zs, z, dz = [], s.z_near, s.dz_start
        while z < s.z_far:
            zs.append(z)
            z += dz
            dz *= s.dz_growth
        self.z = np.array(zs, dtype=np.float64)  # horizontal sample distances

        half_fov = math.tan(math.radians(s.fov_degrees) / 2)
        self.focal = width / 2 / half_fov
        # Each pixel's offset from the view axis, in units of the focal length.
        self._px = ((np.arange(width) + 0.5 - width / 2) / self.focal)[None, :]
        self._py = (-(np.arange(height) + 0.5 - height / 2) / self.focal)[:, None]

        self.dither = np.tile(_BAYER, (height // 4 + 1, width // 4 + 1))[
            :height, :width
        ]
        # Share of the clear-weather view distance before everything is fog;
        # the weather lowers it in rain.
        self.visibility = 1.0
        self._sx = (np.arange(width) + 0.5)[None, :]
        self._sy = (np.arange(height) + 0.5)[:, None]

        # Azimuth bins of all tiers in one numbering: tier t has azimuths / 2^t
        # bins and starts at _tier_offset[t].
        sizes = [s.azimuths >> t for t in range(s.azimuth_tiers)]
        self._tier_size = np.array(sizes)
        self._tier_offset = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self._bin_angle = np.concatenate(
            [(np.arange(n) + 0.5) * math.tau / n for n in sizes]
        )

        # Sample counts azimuths are rounded up to, so each class is one array.
        # Steps of 20% waste little work on rounding up.
        n = len(self.z)
        self._lengths = np.unique(np.maximum(1, np.round(n * 0.8 ** np.arange(24))))
        self._lengths = self._lengths.astype(np.int64)

        # Filled in by render() for overlays that need per-pixel geometry.
        self.pixel_azimuth = np.zeros((height, width))
        self.pixel_slope = np.zeros((height, width))
        self.pixel_distance = np.full((height, width), np.inf)

    def fog_for(self, z: np.ndarray) -> np.ndarray:
        """Continuous fog level for distance ``z``: 0 (clear) .. FOG_LEVELS (sky)."""
        s = self.settings
        far = s.z_far * self.visibility
        return smoothstep(s.fog_start * far, far, z) * palette.FOG_LEVELS

    def depth_ranges(
        self, levels: Sequence[TerrainLevel], count: int | None = None
    ) -> list[tuple[TerrainLevel, slice]]:
        """Assign each sample distance to the finest level whose window covers it."""
        count = len(self.z) if count is None else count
        ranges, start = [], 0
        for i, level in enumerate(levels):
            last = i == len(levels) - 1
            end = (
                count
                if last
                else min(count, int(np.searchsorted(self.z, level.safe_radius)))
            )
            if end > start:
                ranges.append((level, slice(start, end)))
                start = end
        return ranges

    def render(
        self,
        out: np.ndarray,
        levels: Sequence[TerrainLevel],
        camera: Camera,
        time: float = 0.0,
        clouds: CloudLayer | None = None,
    ) -> None:
        """Draw terrain, sky and clouds into ``out`` (``(height, width)`` palette indices).

        ``levels`` go from finest to coarsest; each should be centred near the camera.
        ``time`` in seconds drives the water waves and cloud drift.

        Afterwards ``pixel_azimuth``, ``pixel_slope`` (tangent of the elevation
        angle) and ``pixel_distance``
        (horizontal distance to the terrain or a thick cloud, infinite for sky)
        describe what each pixel looks at, for overlays that must hide behind
        terrain and clouds.
        """
        s = self.settings
        azimuth, slope, east, south = self._pixel_angles(camera)

        # Azimuth bins present on screen, renumbered 0..k-1.
        tier = self._pixel_tiers(camera)
        size = self._tier_size[tier]
        bins = (
            self._tier_offset[tier]
            + (azimuth * size / math.tau).astype(np.int64) % size
        )
        present = np.bincount(bins.ravel(), minlength=len(self._bin_angle)) > 0
        used = np.flatnonzero(present)
        pixel_bin = (np.cumsum(present) - 1)[bins]

        # How many samples each azimuth needs, rounded up to a length class.
        # First bound the reach with the highest point anywhere in the coarsest
        # level, then tighten it with the highest point actually within reach.
        coarse = levels[-1]
        peak = float(coarse.heights.max()) + s.peak_margin
        reach = self._pixel_reach(slope, camera.altitude, peak)
        radius = float(np.max(reach, initial=0.0, where=np.isfinite(reach)))
        if np.isfinite(reach).all():
            peak = local_peak(levels, camera.x, camera.y, radius) + s.peak_margin
            reach = self._pixel_reach(slope, camera.altitude, peak)
        bin_reach = np.zeros(len(used))
        np.maximum.at(bin_reach, pixel_bin.ravel(), reach.ravel())
        end = np.minimum(np.searchsorted(self.z, bin_reach) + 1, len(self.z))

        # Flying above every nearby peak, terrain close by is seen at a steep
        # downward slope; anything shallower than an azimuth's steepest pixel
        # cannot be hit, so its samples can start further out.
        steepest = np.full(len(used), np.inf)
        np.minimum.at(steepest, pixel_bin.ravel(), slope.ravel())
        clearance = camera.altitude - peak
        with np.errstate(divide="ignore"):
            start_distance = np.where(
                (clearance > 0) & (steepest < 0), clearance / -steepest, 0.0
            )
        first = np.maximum(np.searchsorted(self.z, start_distance) - 1, 0)
        needed = np.maximum(end - first, 1)
        length_class = np.searchsorted(self._lengths, needed)

        # Lay the azimuths out class by class: rank is the position in that layout.
        order = np.argsort(length_class, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        lengths = self._lengths[length_class[order]]
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])

        keys, all_heights, samples = [], [], []
        for cls in np.unique(length_class):
            members = order[length_class[order] == cls]
            n_d = int(self._lengths[cls])
            index = np.minimum(
                first[members, None] + np.arange(n_d)[None, :], len(self.z) - 1
            )
            d = self.z[index]
            angle = self._bin_angle[used[members]]
            wx = camera.x + np.sin(angle)[:, None] * d
            wy = camera.y - np.cos(angle)[:, None] * d
            heights = np.empty(wx.shape, dtype=np.float32)
            for level, span in self.depth_ranges(levels):
                if span.start == 0 and index[:, -1].max() < span.stop:
                    mask = slice(None)  # the whole class fits in this level
                else:
                    mask = (index >= span.start) & (index < span.stop)
                    if not mask.any():
                        continue
                heights[mask] = level.smooth_heights(wx[mask], wy[mask])
            # Skyline per azimuth: the steepest slope seen so far outwards.
            skyline = np.maximum.accumulate(
                np.clip((heights - camera.altitude) / d, -_MAX_SLOPE, _MAX_SLOPE),
                axis=1,
            )
            keys.append((skyline + rank[members, None] * _KEY_STRIDE).ravel())
            all_heights.append(heights.ravel())
            samples.append(index.ravel())
        keys = np.concatenate(keys)
        heights = np.concatenate(all_heights)
        samples = np.concatenate(samples)

        pixel_rank = rank[pixel_bin]
        found = np.searchsorted(keys, (slope + pixel_rank * _KEY_STRIDE).ravel())
        found = found.reshape(self.height, self.width) - starts[pixel_rank]
        is_sky = found >= lengths[pixel_rank]
        found = np.minimum(found, lengths[pixel_rank] - 1)
        flat = starts[pixel_rank] + found

        # The ray crosses the terrain between the hit sample and the one before
        # it. Find where, assuming the ground is straight in between, and take
        # the colour at exactly that spot. Colours then belong to fixed points
        # in the world instead of to whichever sample is nearest, so edges
        # between e.g. snow and rock no longer jump by whole samples (several
        # pixels on steep faces) as the camera moves.
        d1, h1 = self.z[samples[flat]], heights[flat]
        before = np.maximum(flat - 1, 0)
        d0, h0 = self.z[samples[before]], heights[before]
        rise = (h1 - h0) - slope * (d1 - d0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (camera.altitude + slope * d0 - h0) / rise
        t = np.where(
            (found > 0) & (rise > 0), np.clip(np.nan_to_num(t, nan=1.0), 0, 1), 1
        )
        distance = d0 + t * (d1 - d0)
        hx = camera.x + distance * east
        hy = camera.y + distance * south
        colors = self._colors_at(levels, hx, hy, distance)
        self._animate_water(colors, hx, hy, time)

        fog_level = np.floor(self.fog_for(distance) + self.dither).astype(np.int32)
        ground = colors.astype(np.int32) + fog_level * palette.TERRAIN_COLORS
        ground = np.where(fog_level >= palette.FOG_LEVELS, palette.SKY_BASE, ground)

        span = math.radians(s.sky_span_degrees)
        sky_level = np.floor(
            np.arctan(slope) / span * (palette.SKY_COLORS - 1) + self.dither
        )
        sky = palette.SKY_BASE + np.clip(sky_level, 0, palette.SKY_COLORS - 1).astype(
            np.int32
        )

        image = np.where(is_sky, sky, ground)
        self.pixel_azimuth = azimuth
        self.pixel_slope = slope
        self.pixel_distance = np.where(is_sky, np.inf, distance)
        if clouds is not None:
            clouds.draw(image, self, camera, time)
        out[:] = image

    def _colors_at(
        self,
        levels: Sequence[TerrainLevel],
        x: np.ndarray,
        y: np.ndarray,
        distance: np.ndarray,
    ) -> np.ndarray:
        """Terrain colours at world points, each from the finest level covering it."""
        colors = np.empty(x.shape, dtype=np.uint8)
        radii = np.array([level.safe_radius for level in levels[:-1]])
        which = np.searchsorted(radii, distance)
        for i, level in enumerate(levels):
            mask = which == i
            if mask.any():
                iy, ix = level.indices(x[mask], y[mask])
                colors[mask] = level.colors[iy, ix]
        return colors

    def _pixel_angles(self, camera: Camera) -> tuple[np.ndarray, ...]:
        """Azimuth (clockwise from north, [0, 2pi)) and slope of every pixel's ray,
        plus the ray's horizontal direction as a unit (east, south) vector."""
        f, r, u = camera.basis()
        rx = f[0] + self._px * r[0] + self._py * u[0]
        ry = f[1] + self._px * r[1] + self._py * u[1]
        rz = f[2] + self._px * r[2] + self._py * u[2]
        azimuth = np.arctan2(rx, -ry)
        azimuth = np.where(azimuth < 0, azimuth + math.tau, azimuth)  # faster than %
        horizontal = np.sqrt(rx * rx + ry * ry)
        # Straight up or down there is no horizontal direction: 0 there.
        level = horizontal > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.clip(rz / horizontal, -_MAX_SLOPE, _MAX_SLOPE)
        east = np.divide(rx, horizontal, out=np.zeros_like(rx), where=level)
        south = np.divide(ry, horizontal, out=np.zeros_like(ry), where=level)
        return azimuth, np.nan_to_num(slope), east, south

    def _pixel_tiers(self, camera: Camera) -> np.ndarray:
        """Azimuth tier per pixel: 0 = full resolution, higher = coarser.

        Around the point on screen straight above or below the camera, the
        azimuth changes quickly from pixel to pixel, so pixels close to it
        need far fewer distinct azimuths than the full resolution.
        """
        s = self.settings
        f, r, u = camera.basis()
        vertical = np.array([0.0, 0.0, 1.0 if f[2] >= 0 else -1.0])
        depth = float(vertical @ f)
        if depth < 1e-3:
            return np.zeros((self.height, self.width), dtype=np.int64)
        vx = self.width / 2 + float(vertical @ r) / depth * self.focal
        vy = self.height / 2 - float(vertical @ u) / depth * self.focal
        rho = np.hypot(self._sx - vx, self._sy - vy)
        needed = np.maximum(math.tau * rho * s.azimuths_per_pixel, 1.0)
        tier = np.floor(np.log2(s.azimuths / needed)).astype(np.int64)
        return np.clip(tier, 0, s.azimuth_tiers - 1)

    def _pixel_reach(
        self, slope: np.ndarray, altitude: float, peak: float
    ) -> np.ndarray:
        """Furthest horizontal distance at which each pixel's ray can meet terrain."""
        above = altitude - self.settings.lowest_surface  # room to fall before sea level
        below_peak = peak - altitude  # room to climb before clearing every peak
        with np.errstate(divide="ignore"):
            down = above / -slope if above > 0 else np.inf
            up = below_peak / slope if below_peak > 0 else 0.0
        return np.where(slope < 0, down, np.where(slope > 0, up, np.inf))

    def _animate_water(
        self, colors: np.ndarray, wx: np.ndarray, wy: np.ndarray, time: float
    ) -> None:
        """Shift water shades up and down with waves travelling across the surface.

        The pattern is a function of world position and time, so the waves
        move over the water instead of sticking to the screen.
        """
        offset = colors.astype(np.int32) - palette.TERRAIN_BASE
        water = (offset >= 0) & (offset < WATER_BIOMES * palette.SHADES)
        if not water.any():
            return
        x, y, t = wx[water], wy[water], time * self.settings.wave_speed
        wave = (
            np.sin(x * 0.11 + y * 0.03 + t)
            + np.sin(y * 0.09 - x * 0.05 - t * 0.7)
            + 0.5 * np.sin(x * 0.31 - y * 0.27 + t * 2.3)
        ) / 2.5
        shift = np.round(wave * self.settings.wave_strength).astype(np.int32)
        offset = offset[water]
        shade = np.clip(offset % palette.SHADES + shift, 0, palette.SHADES - 1)
        colors[water] = palette.TERRAIN_BASE + offset - offset % palette.SHADES + shade


def local_peak(
    levels: Sequence[TerrainLevel], x: float, y: float, radius: float
) -> float:
    """Highest stored terrain within ``radius`` of (x, y), from a coarse level.

    Uses the finest level whose window holds the whole square, falling back to
    the entire coarsest level.
    """
    for level in levels[1:]:
        if radius <= level.safe_radius:
            cells = int(radius / level.step) + 2
            iy, ix = level.indices(x, y)
            rows = (iy + np.arange(-cells, cells + 1)) & level.mask
            cols = (ix + np.arange(-cells, cells + 1)) & level.mask
            return float(level.heights[np.ix_(rows, cols)].max())
    return float(levels[-1].heights.max())


class ScreenView:
    """A renderer's view at ``scale`` times its resolution, for what is drawn
    sharp on top of the scaled-up terrain: the same camera and field of view,
    so a point projects to the same place, but more pixels and a longer focal
    length. Its ``pixel_distance`` is the terrain's, each value repeated over
    the ``scale`` x ``scale`` screen pixels it covers (made once per rendered
    frame). Everything else (fog, settings) is the renderer's.
    """

    def __init__(self, renderer: VoxelRenderer, scale: int) -> None:
        self.renderer = renderer
        self.scale = scale
        self.width = renderer.width * scale
        self.height = renderer.height * scale
        self.focal = renderer.focal * scale
        self._source: np.ndarray | None = None
        self._distance: np.ndarray | None = None

    @property
    def pixel_distance(self) -> np.ndarray:
        source = self.renderer.pixel_distance
        if source is not self._source:
            s = self.scale
            self._distance = np.repeat(np.repeat(source, s, axis=0), s, axis=1)
            self._source = source
        return self._distance

    @property
    def settings(self) -> RenderSettings:
        return self.renderer.settings

    @property
    def visibility(self) -> float:
        return self.renderer.visibility

    def fog_for(self, z: np.ndarray) -> np.ndarray:
        return self.renderer.fog_for(z)
