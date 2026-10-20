"""Phase 0 spike: can NumPy + ``data_ptr()`` render terrain fast enough?

Throwaway code. It answers one question before the real renderer is built: how
many milliseconds per frame does it cost to (1) generate noise for a full screen
and (2) render a Voxel Space view of a heightmap, when everything is vectorised
with NumPy and written straight into Pyxel's screen buffer?

Interactive:  uv run flightsim-spike
    1 / 2       noise view / voxel view
    arrows      move (up/down) and turn (left/right)
    W / S       altitude
    Q / ESC     quit

Headless:     uv run flightsim-spike --bench
"""

import argparse
import time

import numpy as np
import pyxel

from flightsim import config
from flightsim.render.framebuffer import image_array, screen_array

W, H = config.SCREEN_W, config.SCREEN_H

NOISE_GRID = 256  # lattice size of the value-noise grid (power of two)
MAP_SIZE = 512  # voxel heightmap size (power of two, so lookups wrap with a bitmask)
MAP_MASK = MAP_SIZE - 1
HEIGHT_SCALE = 120.0
WATER_LEVEL = 0.47 * HEIGHT_SCALE

# Height bands -> default Pyxel palette: deep water, water, shallows, sand,
# grass, forest, rock, snow.
BAND_LIMITS = np.array([0.38, 0.44, 0.47, 0.49, 0.56, 0.64, 0.71], dtype=np.float32)
BAND_COLORS = np.array([1, 5, 6, 15, 11, 3, 13, 7], dtype=np.uint8)

# Voxel camera.
PROJ_SCALE = 110.0
HORIZON = int(H * 0.4)
Z_FAR = 400.0


def value_noise(
    grid: np.ndarray, x: np.ndarray, y: np.ndarray, period: int, offset: int
) -> np.ndarray:
    """Smooth value noise that repeats every ``period`` lattice cells."""
    gmask = grid.shape[0] - 1
    pmask = period - 1
    x0 = np.floor(x)
    y0 = np.floor(y)
    tx = x - x0
    ty = y - y0
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    xi = x0.astype(np.int64) & pmask
    yi = y0.astype(np.int64) & pmask
    xj = (xi + 1) & pmask
    yj = (yi + 1) & pmask
    xi = (xi + offset) & gmask
    xj = (xj + offset) & gmask
    yi = (yi + offset) & gmask
    yj = (yj + offset) & gmask
    a = grid[yi, xi]
    b = grid[yi, xj]
    c = grid[yj, xi]
    d = grid[yj, xj]
    top = a + (b - a) * tx
    bottom = c + (d - c) * tx
    return top + (bottom - top) * ty


def fbm(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    octaves: int = 5,
    base_period: int = NOISE_GRID,
) -> np.ndarray:
    """Fractal sum of value noise, normalised to roughly [0, 1]."""
    total = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=np.float32)
    amp, freq, norm = 0.5, 1, 0.0
    for octave in range(octaves):
        total += amp * value_noise(
            grid, x * freq, y * freq, min(base_period * freq, NOISE_GRID), octave * 37
        )
        norm += amp
        amp *= 0.5
        freq *= 2
    return total / norm


def height_colors(h: np.ndarray) -> np.ndarray:
    return BAND_COLORS[np.digitize(h, BAND_LIMITS)]


def make_heightmap(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tileable heightmap and colour map for the voxel view."""
    cells = 8  # lattice cells across the map at the lowest octave
    coords = np.arange(MAP_SIZE, dtype=np.float32) * cells / MAP_SIZE
    h = fbm(grid, coords[None, :], coords[:, None], octaves=6, base_period=cells)
    colors = height_colors(h)
    heights = np.maximum(h * HEIGHT_SCALE, WATER_LEVEL).astype(np.float32)
    return heights, colors


def make_z_values() -> np.ndarray:
    """Sample distances, denser close to the camera."""
    zs, z, dz = [], 1.0, 0.5
    while z < Z_FAR:
        zs.append(z)
        z += dz
        dz *= 1.02
    return np.array(zs, dtype=np.float32)


def make_sky() -> np.ndarray:
    rows = np.arange(H)[:, None]
    return np.where(rows < HORIZON - 25, 12, 6).astype(np.uint8)


def render_noise(out: np.ndarray, grid: np.ndarray, ox: float, oy: float) -> None:
    xs = (np.arange(W, dtype=np.float32) + ox) / 48
    ys = (np.arange(H, dtype=np.float32) + oy) / 48
    out[:] = height_colors(fbm(grid, xs[None, :], ys[:, None]))


def render_voxel(
    out: np.ndarray,
    heights: np.ndarray,
    colors: np.ndarray,
    z_values: np.ndarray,
    sky: np.ndarray,
    cam_x: float,
    cam_y: float,
    altitude: float,
    heading: float,
) -> None:
    """Voxel Space render with no Python loop over depth.

    For every (depth, column) pair we project the terrain height to a screen row.
    A running minimum along depth gives the occlusion line ("ybuffer") after each
    step. Pixel (y, x) shows the nearest depth whose occlusion line reaches y, and
    because that line is monotonic per column it can be found with one global
    ``searchsorted`` over all columns at once.
    """
    n_z = len(z_values)
    z = z_values[:, None]
    # -1..1 across the screen, i.e. a 90 degree field of view.
    t = (np.arange(W, dtype=np.float32) + 0.5) / W * 2 - 1
    s, c = np.sin(heading), np.cos(heading)
    ix = (cam_x + z * (c - t * s)).astype(np.int64) & MAP_MASK
    iy = (cam_y + z * (s + t * c)).astype(np.int64) & MAP_MASK
    cols = colors[iy, ix]
    screen_y = (altitude - heights[iy, ix]) / z * PROJ_SCALE + HORIZON
    occlusion = np.minimum.accumulate(np.clip(screen_y, -1, H).astype(np.int32), axis=0)

    # Make each column's -occlusion (ascending) sortable as one array by giving
    # every column its own key range.
    stride = H + 2
    col_base = np.arange(W) * stride
    keys = (col_base[:, None] - occlusion.T).ravel()
    queries = col_base[None, :] - np.arange(H)[:, None]
    depth = np.searchsorted(keys, queries.ravel()).reshape(H, W) - np.arange(W) * n_z
    is_sky = depth >= n_z
    ground = cols[np.minimum(depth, n_z - 1), np.arange(W)]
    out[:] = np.where(is_sky, sky, ground)


class Timer:
    """Exponential moving average of a duration in milliseconds."""

    def __init__(self) -> None:
        self.ms = 0.0

    def add(self, seconds: float) -> None:
        self.ms += (seconds * 1000 - self.ms) * 0.1


class Spike:
    def __init__(self) -> None:
        rng = np.random.default_rng(config.SEED)
        self.grid = rng.random((NOISE_GRID, NOISE_GRID), dtype=np.float32)
        self.heights, self.colors = make_heightmap(self.grid)
        self.z_values = make_z_values()
        self.sky = make_sky()
        self.mode = 2
        self.x, self.y = 0.0, 0.0
        self.altitude = 100.0
        self.heading = 0.0
        self.render_time = Timer()
        self.frame_time = Timer()
        self.last_frame = time.perf_counter()

        pyxel.init(
            W,
            H,
            title=f"{config.TITLE} - spike",
            fps=config.FPS,
            display_scale=config.DISPLAY_SCALE,
        )
        pyxel.run(self.update, self.draw)

    def update(self) -> None:
        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()
        if pyxel.btnp(pyxel.KEY_1):
            self.mode = 1
        if pyxel.btnp(pyxel.KEY_2):
            self.mode = 2
        if pyxel.btn(pyxel.KEY_LEFT):
            self.heading -= 0.05
        if pyxel.btn(pyxel.KEY_RIGHT):
            self.heading += 0.05
        speed = 3.0 * (pyxel.btn(pyxel.KEY_UP) - pyxel.btn(pyxel.KEY_DOWN))
        self.x += np.cos(self.heading) * speed
        self.y += np.sin(self.heading) * speed
        self.altitude += 2.0 * (pyxel.btn(pyxel.KEY_W) - pyxel.btn(pyxel.KEY_S))
        ground = self.heights[int(self.y) & MAP_MASK, int(self.x) & MAP_MASK]
        self.altitude = max(self.altitude, ground + 5)

    def draw(self) -> None:
        now = time.perf_counter()
        self.frame_time.add(now - self.last_frame)
        self.last_frame = now

        start = time.perf_counter()
        out = screen_array()
        if self.mode == 1:
            render_noise(out, self.grid, self.x, self.y)
        else:
            render_voxel(
                out,
                self.heights,
                self.colors,
                self.z_values,
                self.sky,
                self.x,
                self.y,
                self.altitude,
                self.heading,
            )
        self.render_time.add(time.perf_counter() - start)

        fps = 1000 / self.frame_time.ms if self.frame_time.ms else 0
        label = "NOISE" if self.mode == 1 else f"VOXEL z={len(self.z_values)}"
        pyxel.rect(0, 0, W, 9, pyxel.COLOR_BLACK)
        pyxel.text(
            2,
            2,
            f"{label}  render {self.render_time.ms:5.1f} ms  {fps:4.1f} fps",
            pyxel.COLOR_WHITE,
        )


def bench(frames: int) -> None:
    """Time both render paths without opening a window."""
    rng = np.random.default_rng(config.SEED)
    grid = rng.random((NOISE_GRID, NOISE_GRID), dtype=np.float32)

    start = time.perf_counter()
    heights, colors = make_heightmap(grid)
    print(
        f"heightmap {MAP_SIZE}x{MAP_SIZE}: {(time.perf_counter() - start) * 1000:.1f} ms (one-off)"
    )

    z_values = make_z_values()
    sky = make_sky()
    img = pyxel.Image(W, H)  # keep a reference: the view does not keep the image alive
    out = image_array(img)

    def run(name: str, render) -> None:
        times = []
        for i in range(frames):
            start = time.perf_counter()
            render(i)
            times.append((time.perf_counter() - start) * 1000)
        times = np.array(times[3:])  # skip warm-up
        print(
            f"{name:<8} mean {times.mean():5.2f} ms   p95 {np.percentile(times, 95):5.2f} ms   "
            f"({1000 / config.FPS:.1f} ms budget at {config.FPS} fps)"
        )

    print(f"resolution {W}x{H}, voxel depth samples {len(z_values)}")
    run("noise", lambda i: render_noise(out, grid, i * 3.0, i * 1.0))
    run(
        "voxel",
        lambda i: render_voxel(
            out, heights, colors, z_values, sky, i * 3.0, i * 1.0, 100.0, i * 0.02
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bench", action="store_true", help="time the render paths headless and exit"
    )
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    if args.bench:
        bench(args.frames)
    else:
        Spike()


if __name__ == "__main__":
    main()
