"""Voxel Space terrain renderer, fully vectorised with NumPy.

The classic algorithm walks from the camera outwards one depth slice at a
time and, per screen column, draws the terrain from its projected height down
to the highest point drawn so far (the "ybuffer"). Here every depth slice is
handled at once instead:

1. Sample terrain for every (depth, column) pair and project it to a screen row.
2. A running minimum along depth gives the occlusion line after each slice.
3. Pixel (y, x) shows the nearest slice whose occlusion line reaches y. The
   line is monotonic per column, so one global ``searchsorted`` over all
   columns finds it for the whole screen.

Distant slices read from coarser terrain levels, which both extends the view
distance and avoids shimmering from undersampling fine detail far away.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from flightsim.render import palette
from flightsim.sim.camera import Camera
from flightsim.terrain.noise import smoothstep
from flightsim.terrain.world import TILE, TerrainLevel

# 4x4 ordered-dither thresholds in [0, 1).
_BAYER = (
    np.array(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32
    )
    + 0.5
) / 16


@dataclass(frozen=True)
class RenderSettings:
    fov_degrees: float = 90.0
    z_near: float = 1.0
    z_far: float = 700.0
    # Depth step starts at dz_start and grows by dz_growth per slice: dense
    # sampling close to the camera, sparse far away.
    dz_start: float = 0.5
    dz_growth: float = 1.02
    # Fog starts at this fraction of z_far and is complete at z_far.
    fog_start: float = 0.3
    # Sky gradient spans this many pixels above the horizon.
    sky_span: float = 110.0


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
        self.z = np.array(zs, dtype=np.float64)

        half_fov = math.tan(math.radians(s.fov_degrees) / 2)
        self.focal = width / 2 / half_fov
        # Horizontal offset of each column's ray, per unit of depth.
        self.spread = ((np.arange(width) + 0.5) / width * 2 - 1) * half_fov
        self.column_offset = np.arange(width) + 0.5 - width / 2

        self.dither = np.tile(_BAYER, (height // 4 + 1, width // 4 + 1))[
            :height, :width
        ]
        # Continuous fog level per depth slice, 0 (clear) .. FOG_LEVELS (sky).
        self.fog = (
            smoothstep(s.fog_start * s.z_far, s.z_far, self.z) * palette.FOG_LEVELS
        )

        self._cols = np.arange(width)
        self._rows = np.arange(height)[:, None]
        self._stride = height + 2
        self._key_base = self._cols * self._stride
        self._queries = (self._key_base[None, :] - self._rows).ravel()

    def depth_ranges(
        self, levels: Sequence[TerrainLevel]
    ) -> list[tuple[TerrainLevel, slice]]:
        """Assign each depth slice to the finest level whose window still covers it.

        A ray at depth z can reach z * sqrt(1 + spread^2) from the camera, and
        the camera can be up to one tile off the window centre.
        """
        reach = self.z * math.hypot(1.0, float(self.spread[-1]))
        ranges, start = [], 0
        for i, level in enumerate(levels):
            last = i == len(levels) - 1
            limit = level.extent / 2 - TILE * level.step
            end = len(self.z) if last else int(np.searchsorted(reach, limit))
            if end > start:
                ranges.append((level, slice(start, end)))
                start = end
        return ranges

    def render(
        self, out: np.ndarray, levels: Sequence[TerrainLevel], camera: Camera
    ) -> None:
        """Draw terrain and sky into ``out`` (``(height, width)`` palette indices).

        ``levels`` go from finest to coarsest; each should be centred near the camera.
        """
        n_z = len(self.z)
        z = self.z[:, None]
        fx, fy = camera.forward
        rx, ry = camera.right
        lateral = z * self.spread[None, :]
        wx = camera.x + z * fx + lateral * rx
        wy = camera.y + z * fy + lateral * ry

        heights = np.empty((n_z, self.width), dtype=np.float32)
        colors = np.empty((n_z, self.width), dtype=np.uint8)
        for level, rows in self.depth_ranges(levels):
            iy, ix = level.indices(wx[rows], wy[rows])
            heights[rows] = level.heights[iy, ix]
            colors[rows] = level.colors[iy, ix]

        # Pitch moves the horizon up and down; roll tilts it by shearing columns.
        horizon = (
            self.height / 2
            + math.tan(camera.pitch) * self.focal
            - self.column_offset * math.tan(camera.roll)
        )
        screen_y = horizon[None, :] + (camera.altitude - heights) / z * self.focal
        top = np.clip(np.ceil(screen_y), -1, self.height).astype(np.int32)
        occlusion = np.minimum.accumulate(top, axis=0)

        # Per column, -occlusion is ascending in depth; offsetting each column
        # by its own key range makes the whole array sorted for searchsorted.
        keys = (self._key_base[:, None] - occlusion.T).ravel()
        depth = np.searchsorted(keys, self._queries).reshape(self.height, self.width)
        depth -= self._cols * n_z
        is_sky = depth >= n_z
        depth = np.minimum(depth, n_z - 1)

        fog_level = np.floor(self.fog[depth] + self.dither).astype(np.int32)
        ground = (
            colors[depth, self._cols].astype(np.int32)
            + fog_level * palette.TERRAIN_COLORS
        )
        ground = np.where(fog_level >= palette.FOG_LEVELS, palette.SKY_BASE, ground)

        above = (horizon[None, :] - self._rows) / self.settings.sky_span
        sky_level = np.floor(above * (palette.SKY_COLORS - 1) + self.dither)
        sky = palette.SKY_BASE + np.clip(sky_level, 0, palette.SKY_COLORS - 1).astype(
            np.int32
        )

        out[:] = np.where(is_sky, sky, ground)
