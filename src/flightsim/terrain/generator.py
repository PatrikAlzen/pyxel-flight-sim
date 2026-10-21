"""Turns world coordinates into terrain: height, moisture, biome and colour.

Units: one world unit is one heightmap cell at full detail; heights use the
same unit. Sea level is 0.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from flightsim.render.palette import SHADES, biome_color
from flightsim.terrain import shading
from flightsim.terrain.airfields import Airfield, Airfields, AirfieldSettings
from flightsim.terrain.biomes import Biome, classify
from flightsim.terrain.bridges import Bridges
from flightsim.terrain.noise import FractalNoise, domain_warp, smoothstep
from flightsim.terrain.settlements import Settlements

# World spacing of the lattice the river noise is sampled on for tiles.
RIVER_LATTICE = 8.0


@dataclass(frozen=True)
class TerrainSettings:
    sea_level: float = 0.0

    # Continents: very low frequency, warped so coastlines are not blobby.
    continent_period: float = 2400.0
    continent_warp: float = 0.35  # in continent periods
    continent_bias: float = 0.06  # > 0 gives more land than sea
    continent_height: float = 110.0
    coast_step: float = 24.0

    # Mountain ranges: ridged noise, only inland and only where the range mask is high.
    mountain_period: float = 700.0
    range_period: float = 2500.0
    mountain_height: float = 320.0

    # Rolling hills everywhere on land.
    hill_period: float = 110.0
    hill_height: float = 14.0

    moisture_period: float = 1500.0

    # Rivers: the zero lines of a low-frequency noise, in the same warped
    # coordinates as the continents so they meander. A valley is cut towards a
    # low floor and the channel itself below sea level, so it fills with water.
    river_period: float = 1700.0
    river_width: tuple[float, float] = (3.0, 10.0)  # half-width of the water
    valley_width: float = 9.0  # valley half-width, in river half-widths
    valley_floor: float = 0.08  # of the height above sea level, left at the bottom
    channel_depth: float = 4.0  # below sea level in the middle of the river
    river_fade: tuple[float, float] = (150.0, 250.0)  # rivers end up in mountains
    valley_moisture: float = 0.3  # greener valleys

    # Biome thresholds (heights above sea level).
    deep_water: float = 18.0
    beach_height: float = 2.5
    tree_line: float = 135.0
    snow_line: float = 175.0
    rock_slope: float = 1.1
    biome_jitter: float = 10.0
    # Slope for choosing biomes (e.g. rock on steep faces) is measured over this
    # world distance at every level of detail. Measuring over the grid spacing
    # instead made coarse levels see smoother, snowier mountains than fine ones,
    # so slopes flipped between snow and rock where the levels meet on screen.
    biome_slope_span: float = 16.0

    airfields: AirfieldSettings = field(default_factory=AirfieldSettings)


@dataclass
class TerrainSample:
    """A rectangular grid of generated terrain.

    ``height`` is the visible surface (water flattened to sea level) and is what
    rendering and collision use. ``ground`` includes the sea floor.
    """

    x0: float
    y0: float
    step: float
    height: np.ndarray
    ground: np.ndarray
    moisture: np.ndarray
    biome: np.ndarray
    color: np.ndarray


class TerrainGenerator:
    def __init__(self, seed: int, settings: TerrainSettings | None = None) -> None:
        self.seed = seed
        self.settings = settings or TerrainSettings()
        self._continent = FractalNoise(seed + 1, octaves=5)
        self._warp_x = FractalNoise(seed + 2, octaves=3)
        self._warp_y = FractalNoise(seed + 3, octaves=3)
        self._mountains = FractalNoise(seed + 4, octaves=6)
        self._ranges = FractalNoise(seed + 5, octaves=2)
        self._hills = FractalNoise(seed + 6, octaves=4)
        self._moisture = FractalNoise(seed + 7, octaves=3)
        self._jitter = FractalNoise(seed + 8, octaves=2)
        self._rivers = FractalNoise(seed + 9, octaves=2)
        self.airfields = Airfields(
            seed,
            self.natural_elevation,
            self.settings.sea_level,
            self.settings.airfields,
        )
        self.settlements = Settlements(
            seed, self.elevation, self.moisture, self.settings.sea_level, self.airfields
        )
        self.bridges = Bridges(seed, self)

    def elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Terrain height at world coordinates, including the sea floor and
        river beds."""
        return self.airfields.flatten(x, y, self.natural_elevation(x, y))

    def natural_elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Terrain height before airfields level the ground."""
        return self._natural(x, y)[0]

    def river_valley(self, x: float, y: float) -> float:
        """How deep in a river valley (x, y) is: 0 outside, 1 over the river."""
        return float(self._natural(np.array([x]), np.array([y]))[1][0])

    def _natural(
        self, x: np.ndarray, y: np.ndarray, lattice: float = RIVER_LATTICE
    ) -> tuple[np.ndarray, np.ndarray]:
        """Natural terrain height, and how deep in a river valley each point is
        (0 outside, 1 at the river). ``lattice``: see ``_river_distance``."""
        s = self.settings
        cx, cy = domain_warp(
            np.asarray(x, dtype=np.float64) / s.continent_period,
            np.asarray(y, dtype=np.float64) / s.continent_period,
            self._warp_x,
            self._warp_y,
            s.continent_warp,
        )
        continent = self._continent.fbm(cx, cy) + s.continent_bias
        inland = smoothstep(0.02, 0.2, continent)
        on_land = smoothstep(0.0, 0.1, continent)
        # A step in height right at the coastline keeps shores narrow and crisp
        # instead of wide, speckled tidal flats.
        coast = (smoothstep(-0.03, 0.03, continent) - 0.5) * s.coast_step
        base = continent * s.continent_height + coast

        ranges = smoothstep(
            -0.25, 0.25, self._ranges.fbm(x / s.range_period, y / s.range_period)
        )
        # Mountains and hills are the most expensive layers and are masked out
        # over much of the world (sea, plains), so only evaluate them where needed.
        ridges = _where_needed(
            inland * ranges,
            lambda x, y: self._mountains.ridged(
                x / s.mountain_period, y / s.mountain_period
            ),
            x,
            y,
        )
        mountains = ridges**2 * s.mountain_height * inland * ranges

        hills = (
            _where_needed(
                on_land,
                lambda x, y: self._hills.fbm(x / s.hill_period, y / s.hill_period),
                x,
                y,
            )
            * s.hill_height
            * on_land
        )

        height = s.sea_level + base + mountains + hills
        return self._carve_rivers(x, y, cx, cy, height, lattice)

    def _carve_rivers(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        height: np.ndarray,
        lattice: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cut river valleys into ``height`` at world (x, y), whose warped
        continent coordinates are (cx, cy).

        A river runs where the river noise ``r`` is zero, and the distance to
        that line is about ``|r| / |grad r|`` (see ``_river_distance``).
        """
        s = self.settings
        above = height - s.sea_level
        fade = 1.0 - smoothstep(*s.river_fade, above)
        need = (above > 0) & (fade > 0)
        valley = np.zeros(np.shape(height), dtype=np.float32)
        if not need.any():
            return height, valley
        wx, wy = np.broadcast_arrays(np.asarray(x, dtype=np.float64), y)
        distance = self._river_distance(wx[need], wy[need], lattice)
        low, high = s.river_width
        # Only points within the widest valley of a river change at all.
        cells = np.flatnonzero(need)
        near = distance < high * s.valley_width
        cells, distance = cells[near], distance[near]
        result = np.broadcast_to(
            np.asarray(height, dtype=np.float64), need.shape
        ).copy()
        if not len(cells):
            return result, valley
        scale = s.continent_period / s.river_period
        u = np.broadcast_to(cx, need.shape).ravel()[cells] * scale
        v = np.broadcast_to(cy, need.shape).ravel()[cells] * scale
        # The river widens and narrows along its course.
        width = low + (high - low) * (0.5 + 0.5 * np.sin(u * 7.3 + v * 5.1))
        f = fade.ravel()[cells]
        here = above.ravel()[cells]
        floor = here * s.valley_floor
        valley_t = smoothstep(width * s.valley_width, width, distance) * f
        carved = here + (floor - here) * valley_t
        channel = smoothstep(width, width * 0.3, distance) * f
        carved = carved + (-s.channel_depth - carved) * channel
        result.ravel()[cells] = s.sea_level + np.minimum(carved, here)
        valley.ravel()[cells] = valley_t
        return result, valley

    def _river_distance(
        self, x: np.ndarray, y: np.ndarray, cell: float = RIVER_LATTICE
    ) -> np.ndarray:
        """Approximate world distance from each point to the nearest river line."""
        r, drdx, drdy = self.river_field(x, y, cell)
        return np.abs(r) / np.maximum(np.hypot(drdx, drdy), 1e-9)

    def river_field(
        self, x: np.ndarray, y: np.ndarray, cell: float = RIVER_LATTICE
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The river noise ``r`` (rivers run where it is 0) and its world
        gradient at points: the gradient points across the river.

        The river noise changes slowly, so it is evaluated on a fixed world
        lattice and interpolated bilinearly, which also gives its gradient from
        the cell corners. A tile needs only a few hundred corners instead of
        thousands of points. Every caller uses the same lattice, so heights
        depend on position alone: a village placed on a riverbank stands on the
        same ground the tile draws. Coarse levels of detail pass a coarser
        ``cell`` (four times their grid spacing); nothing stands on them, and the river
        line moves by well under a cell.
        """
        gx, gy = np.floor(x / cell), np.floor(y / cell)
        x0, y0 = gx.min(), gy.min()
        box = (gx.max() - x0 + 2) * (gy.max() - y0 + 2)
        if box <= 2 * x.size:
            # Points close together (a tile): all corners of their bounding box.
            lx = np.arange(x0, gx.max() + 2)[None, :]
            ly = np.arange(y0, gy.max() + 2)[:, None]
            grid = self._river_noise(lx * cell, ly * cell)
            width = grid.shape[1]
            k = ((gy - y0) * width + (gx - x0)).astype(np.int64)
            flat = grid.ravel()
            r00, r10 = flat[k], flat[k + 1]
            r01, r11 = flat[k + width], flat[k + width + 1]
        else:
            # Scattered points: each distinct corner once.
            ix = np.concatenate([gx, gx + 1, gx, gx + 1]).astype(np.int64)
            iy = np.concatenate([gy, gy, gy + 1, gy + 1]).astype(np.int64)
            keys, inverse = np.unique(
                (ix + 2**30) * 2**31 + (iy + 2**30), return_inverse=True
            )
            corners = self._river_noise(
                ((keys // 2**31) - 2**30) * cell, ((keys % 2**31) - 2**30) * cell
            )[inverse].reshape(4, -1)
            r00, r10, r01, r11 = corners
        fx = (x / cell - gx).astype(np.float32)
        fy = (y / cell - gy).astype(np.float32)
        top = r00 + (r10 - r00) * fx
        bottom = r01 + (r11 - r01) * fx
        r = top + (bottom - top) * fy
        drdx = ((r10 - r00) * (1 - fy) + (r11 - r01) * fy) / cell
        drdy = (bottom - top) / cell
        return r, drdx, drdy

    def river_crossing(self, x: float, y: float) -> tuple[float, float, float, float]:
        """The point on the river line nearest (x, y), and the unit direction
        straight across the river there: (cx, cy, ax, ay)."""
        px, py = float(x), float(y)
        for _ in range(3):  # Newton steps onto r = 0
            r, gx, gy = (
                float(v[0]) for v in self.river_field(np.array([px]), np.array([py]))
            )
            g2 = max(gx * gx + gy * gy, 1e-18)
            px, py = px - r * gx / g2, py - r * gy / g2
        _, gx, gy = (
            float(v[0]) for v in self.river_field(np.array([px]), np.array([py]))
        )
        norm = max(math.hypot(gx, gy), 1e-12)
        return px, py, gx / norm, gy / norm

    def _river_noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The river noise at world points, in warped continent coordinates."""
        s = self.settings
        cx, cy = domain_warp(
            np.asarray(x, dtype=np.float64) / s.continent_period,
            np.asarray(y, dtype=np.float64) / s.continent_period,
            self._warp_x,
            self._warp_y,
            s.continent_warp,
        )
        scale = s.continent_period / s.river_period
        return self._rivers.fbm(cx * scale, cy * scale)

    def _paint_runways(
        self, x: np.ndarray, y: np.ndarray, biome: np.ndarray, shade: np.ndarray
    ) -> None:
        """Paint asphalt and markings over the cells of any runway in the grid."""
        for airfield in self.airfields.overlapping(x.min(), y.min(), x.max(), y.max()):
            _paint(airfield, x, y, biome, shade)

    def moisture(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Moisture in [0, 1]; drives the choice between desert, grass and forest."""
        s = self.settings
        n = self._moisture.fbm(x / s.moisture_period, y / s.moisture_period)
        return np.clip(n * 1.4 + 0.5, 0.0, 1.0)

    def sample(
        self, x0: float, y0: float, width: int, height: int, step: float = 1.0
    ) -> TerrainSample:
        """Generate a ``height`` x ``width`` grid whose first cell is at world (x0, y0).

        ``step`` is the world distance between cells; values above 1 give a
        zoomed-out overview. Every value depends only on world position, so
        adjacent samples line up exactly.
        """
        s = self.settings
        # Extra cells on each side so slopes and lighting use real neighbours:
        # one for lighting, more for the wider span used to choose biomes.
        reach = max(1, math.ceil(s.biome_slope_span / 2 / step))
        gx = np.round(x0 / step) + np.arange(-reach, width + reach)
        gy = np.round(y0 / step) + np.arange(-reach, height + reach)
        xs = gx[None, :] * step
        ys = gy[:, None] * step
        natural, valley_b = self._natural(xs, ys, max(RIVER_LATTICE, 4 * step))
        ground_b = self.airfields.flatten(xs, ys, natural)
        surface_b = np.maximum(ground_b, s.sea_level)

        inner = slice(reach, reach + height), slice(reach, reach + width)
        lit = slice(reach - 1, reach + height + 1), slice(reach - 1, reach + width + 1)
        ground = ground_b[inner]
        surface = surface_b[inner]
        inner_x, inner_y = xs[:, inner[1]], ys[inner[0], :]
        moisture = np.clip(
            self.moisture(inner_x, inner_y) + valley_b[inner] * s.valley_moisture,
            0.0,
            1.0,
        )
        jitter = self._jitter.fbm(inner_x / 40, inner_y / 40) * s.biome_jitter
        biome = classify(
            ground,
            moisture,
            shading.slope(ground_b, step, reach),
            jitter,
            sea_level=s.sea_level,
            deep_water=s.deep_water,
            beach_height=s.beach_height,
            tree_line=s.tree_line,
            snow_line=s.snow_line,
            rock_slope=s.rock_slope,
        )

        # Land is lit by the sun; water gets lighter towards the shore.
        depth = s.sea_level - ground
        water_light = np.where(
            biome == Biome.DEEP_WATER,
            1.0 - np.clip((depth - s.deep_water) / 60, 0, 1),
            1.0 - np.clip(depth / s.deep_water, 0, 1),
        )
        light = np.where(depth > 0, water_light, shading.lambert(surface_b[lit], step))
        ix = gx[None, inner[1]].astype(np.int64)
        iy = gy[inner[0], None].astype(np.int64)
        shade = shading.quantize(light, SHADES, ix, iy)
        self._paint_runways(inner_x, inner_y, biome, shade)

        return TerrainSample(
            x0=float(gx[reach] * step),
            y0=float(gy[reach] * step),
            step=step,
            height=surface.astype(np.float32),
            ground=ground.astype(np.float32),
            moisture=moisture.astype(np.float32),
            biome=biome,
            color=biome_color(biome, shade),
        )


def _paint(
    airfield: Airfield,
    x: np.ndarray,
    y: np.ndarray,
    biome: np.ndarray,
    shade: np.ndarray,
) -> None:
    x, y = np.broadcast_arrays(x, y)
    runway = airfield.on_runway(x, y)
    if runway.any():
        biome[runway] = Biome.RUNWAY
        # Asphalt with a sparse lighter fleck (a full checkerboard looked coarse
        # close up), and the paint in the brightest shade.
        cells = np.floor(x[runway]) * 3 + np.floor(y[runway]) * 5
        shade[runway] = 1 + (cells % 4 == 0)
        shade[airfield.markings(x, y)] = SHADES - 1


def _where_needed(
    weight: np.ndarray,
    noise: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """``noise(x, y)`` where ``weight > 0`` and 0 elsewhere.

    Noise is evaluated point by point, so computing it on a subset gives
    exactly the same values as computing it everywhere.
    """
    need = weight > 0
    count = int(np.count_nonzero(need))
    if count == need.size:
        return np.broadcast_to(noise(x, y), need.shape)
    out = np.zeros(need.shape, dtype=np.float32)
    if count:
        xs, ys = np.broadcast_arrays(x, y)
        out[need] = noise(xs[need], ys[need])
    return out
