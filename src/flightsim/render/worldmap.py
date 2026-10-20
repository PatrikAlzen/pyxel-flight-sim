"""Full-screen top-down map, generated straight from the terrain generator.

Unlike the minimap it is not limited to the terrain kept around the camera,
so it can show the world far beyond the view distance. One screen takes a few
tens of milliseconds to generate, so the picture is cached and only rebuilt
when the view moves by a whole pixel or the zoom changes.
"""

import math
from collections.abc import Sequence

import numpy as np
import pyxel

from flightsim.persist import Place
from flightsim.render.framebuffer import screen_array
from flightsim.render.hud import CAUTION, HUD, TEXT, shadow_text
from flightsim.terrain.generator import TerrainGenerator

ZOOMS = [4, 8, 16, 32, 64]  # world units per pixel


class WorldMap:
    def __init__(self, generator: TerrainGenerator) -> None:
        self.generator = generator
        self._key: tuple | None = None
        self._image: np.ndarray | None = None
        self.origin = (0.0, 0.0)

    def image(
        self, cx: float, cy: float, step: float, width: int, height: int
    ) -> np.ndarray:
        """Colours of a ``height`` x ``width`` map centred near (cx, cy)."""
        gx, gy = round(cx / step), round(cy / step)
        key = (gx, gy, step, width, height)
        if key != self._key:
            x0, y0 = (gx - width // 2) * step, (gy - height // 2) * step
            self._image = self.generator.sample(x0, y0, width, height, step).color
            self._key = key
            self.origin = (x0, y0)
        return self._image

    def draw(self, cx: float, cy: float, step: float) -> None:
        """Fill the screen with the map centred near (cx, cy)."""
        screen_array()[:] = self.image(cx, cy, step, pyxel.width, pyxel.height)

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        step = self._key[2] if self._key else 1
        return (x - self.origin[0]) / step, (y - self.origin[1]) / step


def draw_plane_marker(x: float, y: float, heading: float, color: int = TEXT) -> None:
    """Arrow pointing along ``heading`` (radians clockwise from north)."""
    fx, fy = math.sin(heading), -math.cos(heading)
    rx, ry = -fy, fx
    tip = (x + fx * 5, y + fy * 5)
    left = (x - fx * 3 - rx * 3, y - fy * 3 - ry * 3)
    right = (x - fx * 3 + rx * 3, y - fy * 3 + ry * 3)
    pyxel.tri(*tip, *left, *right, pyxel.COLOR_BLACK)
    pyxel.trib(*tip, *left, *right, color)


def draw_wreck_marker(x: float, y: float) -> None:
    """A small cross where the plane once crashed."""
    for dx, dy in ((1, 1), (0, 0)):  # black outline, then red
        color = pyxel.COLOR_BLACK if dx else pyxel.COLOR_RED
        pyxel.line(x - 2 + dx, y - 2 + dy, x + 2 + dx, y + 2 + dy, color)
        pyxel.line(x - 2 + dx, y + 2 + dy, x + 2 + dx, y - 2 + dy, color)


def draw_place_marker(x: float, y: float, name: str) -> None:
    """A saved place: a small square with the place's number beside it."""
    pyxel.rect(x - 2, y - 2, 5, 5, pyxel.COLOR_BLACK)
    pyxel.rect(x - 1, y - 1, 3, 3, pyxel.COLOR_CYAN)
    shadow_text(x + 4, y - 2, name.removeprefix("PLACE "), pyxel.COLOR_CYAN)


def draw_map_overlay(
    world_map: WorldMap,
    x: float,
    y: float,
    heading: float,
    step: float,
    wrecks: np.ndarray | None = None,
    places: Sequence[Place] = (),
) -> None:
    """The in-flight map: terrain, crash sites, saved places, the plane, a scale
    bar and help."""
    world_map.draw(x, y, step)
    if wrecks is not None:
        for wx, wy, _ in wrecks:
            draw_wreck_marker(*world_map.to_screen(wx, wy))
    for place in places:
        draw_place_marker(*world_map.to_screen(place.x, place.y), place.name)
    px, py = world_map.to_screen(x, y)
    if pyxel.frame_count // 10 % 2 == 0:
        draw_plane_marker(px, py, heading, CAUTION)

    w, h = pyxel.width, pyxel.height
    pyxel.rect(0, 0, w, 9, pyxel.COLOR_BLACK)
    pyxel.text(2, 2, f"MAP  1:{step:g}   X {x:.0f}  Y {y:.0f}", TEXT)
    pyxel.text(w - 4 * 24 - 2, 2, "Z/X ZOOM   M/ESC CLOSE", HUD)

    # Scale bar: a round number of world units close to 40 pixels.
    units = 10 ** math.floor(math.log10(40 * step))
    for nice in (1, 2, 5):
        if units * nice / step <= 48:
            length = units * nice
    bar = length / step
    pyxel.rect(4, h - 10, bar, 3, pyxel.COLOR_BLACK)
    pyxel.rect(4, h - 9, bar, 1, TEXT)
    shadow_text(8 + bar, h - 12, f"{length:g}", TEXT)
    shadow_text(w - 8, 12, "N", CAUTION)
    pyxel.tri(w - 7, 19, w - 9, 24, w - 5, 24, CAUTION)
