"""Main Pyxel application.

For now this is a free "ghost" camera for exploring the terrain:

    up / down       move forward / back
    left / right    turn
    A / D           strafe
    W / S           rise / sink
    R / F           look up / down
    Z / X           roll left / right
    V               level the view
    SHIFT           move faster
    F1              toggle debug info
    Q / ESC         quit

``flightsim --bench`` times terrain generation and rendering without a window.
"""

import argparse
import math
import time
from collections.abc import Iterator

import numpy as np
import pyxel

from flightsim import config
from flightsim.render import palette
from flightsim.render.framebuffer import image_array, screen_array
from flightsim.render.voxel import VoxelRenderer
from flightsim.sim.camera import Camera
from flightsim.terrain.generator import TerrainGenerator
from flightsim.terrain.world import TerrainLevel

W, H = config.SCREEN_W, config.SCREEN_H
MAX_PITCH = math.radians(45)
MAX_ROLL = math.radians(45)
LOAD_BUDGET = 0.025  # seconds of terrain generation per frame while loading


def make_levels() -> list[TerrainLevel]:
    return [TerrainLevel.empty(size, step) for size, step in config.TERRAIN_LEVELS]


def fill_levels(
    levels: list[TerrainLevel], generator: TerrainGenerator, x: float, y: float
) -> Iterator[float]:
    """Fill every level around (x, y), yielding overall progress after each tile."""
    steps = [level.fill_tiles(generator, x, y) for level in levels]
    for i, step in enumerate(steps):
        for fraction in step:
            yield (i + fraction) / len(steps)


class Timer:
    """Exponential moving average of a duration in milliseconds."""

    def __init__(self) -> None:
        self.ms = 0.0

    def add(self, seconds: float) -> None:
        self.ms += (seconds * 1000 - self.ms) * 0.1


class App:
    def __init__(self) -> None:
        self.generator = TerrainGenerator(config.SEED)
        self.levels = make_levels()
        self.renderer = VoxelRenderer(W, H)
        self.camera = Camera()
        self.loading: Iterator[float] | None = fill_levels(
            self.levels, self.generator, self.camera.x, self.camera.y
        )
        self.progress = 0.0
        self.show_debug = True
        self.render_time = Timer()
        self.frame_time = Timer()
        self.last_frame = time.perf_counter()

        pyxel.init(
            W,
            H,
            title=config.TITLE,
            fps=config.FPS,
            display_scale=config.DISPLAY_SCALE,
        )
        palette.install()
        pyxel.run(self.update, self.draw)

    def update(self) -> None:
        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()
        if self.loading:
            self.load_step()
            return
        if pyxel.btnp(pyxel.KEY_F1):
            self.show_debug = not self.show_debug
        self.move_camera()

    def load_step(self) -> None:
        deadline = time.perf_counter() + LOAD_BUDGET
        try:
            while time.perf_counter() < deadline:
                self.progress = next(self.loading)
        except StopIteration:
            self.loading = None
            ground = self.levels[0].height_at(self.camera.x, self.camera.y)
            self.camera.altitude = ground + 60

    def move_camera(self) -> None:
        cam = self.camera
        speed = config.GHOST_SPEED * (
            config.GHOST_FAST if pyxel.btn(pyxel.KEY_SHIFT) else 1
        )

        cam.heading += config.GHOST_TURN * (
            pyxel.btn(pyxel.KEY_RIGHT) - pyxel.btn(pyxel.KEY_LEFT)
        )
        ahead = speed * (pyxel.btn(pyxel.KEY_UP) - pyxel.btn(pyxel.KEY_DOWN))
        side = speed * (pyxel.btn(pyxel.KEY_D) - pyxel.btn(pyxel.KEY_A))
        fx, fy = cam.forward
        rx, ry = cam.right
        cam.x += fx * ahead + rx * side
        cam.y += fy * ahead + ry * side
        cam.altitude += speed * (pyxel.btn(pyxel.KEY_W) - pyxel.btn(pyxel.KEY_S))

        tilt = config.GHOST_TURN
        cam.pitch += tilt * (pyxel.btn(pyxel.KEY_R) - pyxel.btn(pyxel.KEY_F))
        cam.roll += tilt * (pyxel.btn(pyxel.KEY_X) - pyxel.btn(pyxel.KEY_Z))
        if pyxel.btnp(pyxel.KEY_V):
            cam.pitch = cam.roll = 0.0
        cam.pitch = max(-MAX_PITCH, min(MAX_PITCH, cam.pitch))
        cam.roll = max(-MAX_ROLL, min(MAX_ROLL, cam.roll))

        ground = self.levels[0].height_at(cam.x, cam.y)
        cam.altitude = max(cam.altitude, ground + config.GHOST_CLEARANCE)

    def draw(self) -> None:
        if self.loading:
            self.draw_loading()
            return

        now = time.perf_counter()
        self.frame_time.add(now - self.last_frame)
        self.last_frame = now

        start = time.perf_counter()
        self.renderer.render(screen_array(), self.levels, self.camera)
        self.render_time.add(time.perf_counter() - start)

        if self.show_debug:
            self.draw_debug()

    def draw_loading(self) -> None:
        pyxel.cls(pyxel.COLOR_NAVY)
        msg = "GENERATING TERRAIN"
        pyxel.text((W - len(msg) * 4) // 2, H // 2 - 10, msg, pyxel.COLOR_WHITE)
        bar_w = 120
        x = (W - bar_w) // 2
        pyxel.rectb(x, H // 2, bar_w, 6, pyxel.COLOR_WHITE)
        pyxel.rect(
            x + 1, H // 2 + 1, int((bar_w - 2) * self.progress), 4, pyxel.COLOR_LIME
        )

    def draw_debug(self) -> None:
        cam = self.camera
        ground = self.levels[0].height_at(cam.x, cam.y)
        fps = 1000 / self.frame_time.ms if self.frame_time.ms else 0
        agl = cam.altitude - ground
        pitch = math.degrees(cam.pitch)
        roll = math.degrees(cam.roll)
        lines = [
            f"X {cam.x:.0f} Y {cam.y:.0f}  ALT {cam.altitude:.0f}  AGL {agl:.0f}",
            f"HDG {cam.heading_degrees:03.0f}  PITCH {pitch:+.0f}  ROLL {roll:+.0f}",
            f"RENDER {self.render_time.ms:4.1f}MS  {fps:4.1f}FPS",
        ]
        for i, line in enumerate(lines):
            pyxel.text(3, 3 + i * 7, line, pyxel.COLOR_BLACK)
            pyxel.text(2, 2 + i * 7, line, pyxel.COLOR_WHITE)


def bench(frames: int) -> None:
    """Time terrain generation and rendering without opening a window."""
    generator = TerrainGenerator(config.SEED)
    levels = make_levels()
    start = time.perf_counter()
    for _ in fill_levels(levels, generator, 0.0, 0.0):
        pass
    print(
        f"terrain levels {config.TERRAIN_LEVELS}: {time.perf_counter() - start:.2f} s"
    )

    renderer = VoxelRenderer(W, H)
    print(f"resolution {W}x{H}, depth slices {len(renderer.z)}")
    img = pyxel.Image(W, H)  # keep a reference: the view does not keep the image alive
    out = image_array(img)
    camera = Camera(altitude=levels[0].height_at(0, 0) + 60)
    times = []
    for i in range(frames):
        camera.heading = i * 0.02
        camera.x, camera.y = math.sin(i * 0.01) * 100, i * 1.5
        start = time.perf_counter()
        renderer.render(out, levels, camera)
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times[3:])  # skip warm-up
    print(
        f"render mean {times.mean():.2f} ms, p95 {np.percentile(times, 95):.2f} ms "
        f"({1000 / config.FPS:.1f} ms budget at {config.FPS} fps)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyxel flight simulator")
    parser.add_argument(
        "--bench", action="store_true", help="time rendering headless and exit"
    )
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    if args.bench:
        bench(args.frames)
    else:
        App()
