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
from flightsim.render import canvas
from flightsim.render.hud import CAUTION, HUD, TEXT, shadow_text
from flightsim.sim.race import Ring
from flightsim.terrain.airfields import Airfield
from flightsim.terrain.bridges import Bridge
from flightsim.terrain.generator import TerrainGenerator
from flightsim.terrain.settlements import Lighthouse, Village

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
        canvas.array()[:] = self.image(cx, cy, step, canvas.width(), canvas.height())

    @property
    def step(self) -> float:
        """World units per pixel of the picture last drawn."""
        return self._key[2] if self._key else 1

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        step = self.step
        return (x - self.origin[0]) / step, (y - self.origin[1]) / step


def draw_plane_marker(x: float, y: float, heading: float, color: int = TEXT) -> None:
    """Arrow pointing along ``heading`` (radians clockwise from north)."""
    fx, fy = math.sin(heading), -math.cos(heading)
    rx, ry = -fy, fx
    tip = (x + fx * 5, y + fy * 5)
    left = (x - fx * 3 - rx * 3, y - fy * 3 - ry * 3)
    right = (x - fx * 3 + rx * 3, y - fy * 3 + ry * 3)
    canvas.tri(*tip, *left, *right, pyxel.COLOR_BLACK)
    canvas.trib(*tip, *left, *right, color)


def draw_wreck_marker(x: float, y: float) -> None:
    """A small cross where the plane once crashed."""
    for dx, dy in ((1, 1), (0, 0)):  # black outline, then red
        color = pyxel.COLOR_BLACK if dx else pyxel.COLOR_RED
        canvas.line(x - 2 + dx, y - 2 + dy, x + 2 + dx, y + 2 + dy, color)
        canvas.line(x - 2 + dx, y + 2 + dy, x + 2 + dx, y - 2 + dy, color)


def draw_runway_marker(
    world_map: WorldMap, field: Airfield, visited: bool, reachable: bool = True
) -> None:
    """A runway drawn to scale (at least a few pixels long), with its number:
    green once landed at, grey when the fuel left does not reach it."""
    half = max(field.length / 2, 3 * world_map.step)
    x0, y0 = world_map.to_screen(*field.point(-half))
    x1, y1 = world_map.to_screen(*field.point(half))
    color = pyxel.COLOR_LIME if visited else pyxel.COLOR_WHITE
    if not reachable:
        color = pyxel.COLOR_GRAY
    canvas.line(x0 + 1, y0 + 1, x1 + 1, y1 + 1, pyxel.COLOR_BLACK)
    canvas.line(x0, y0, x1, y1, color)
    shadow_text(max(x0, x1) + 3, (y0 + y1) / 2 - 2, str(field.number), color)


def draw_bridge_marker(world_map: WorldMap, bridge: Bridge, flown_under: bool) -> None:
    """A bridge across its river, at least a few pixels long; named once flown under."""
    half = max(bridge.half_span, 2 * world_map.step)
    x0, y0 = world_map.to_screen(
        bridge.x - bridge.ax * half, bridge.y - bridge.ay * half
    )
    x1, y1 = world_map.to_screen(
        bridge.x + bridge.ax * half, bridge.y + bridge.ay * half
    )
    color = pyxel.COLOR_YELLOW if flown_under else pyxel.COLOR_GRAY
    canvas.line(x0 + 1, y0 + 1, x1 + 1, y1 + 1, pyxel.COLOR_BLACK)
    canvas.line(x0, y0, x1, y1, color)
    if flown_under:
        shadow_text(max(x0, x1) + 3, (y0 + y1) / 2 - 2, bridge.name, color)


def draw_place_marker(x: float, y: float, name: str) -> None:
    """A saved place: a small square with the place's number beside it."""
    canvas.rect(x - 2, y - 2, 5, 5, pyxel.COLOR_BLACK)
    canvas.rect(x - 1, y - 1, 3, 3, pyxel.COLOR_CYAN)
    shadow_text(x + 4, y - 2, name.removeprefix("PLACE "), pyxel.COLOR_CYAN)


def draw_map_overlay(
    world_map: WorldMap,
    x: float,
    y: float,
    heading: float,
    step: float,
    wrecks: np.ndarray | None = None,
    places: Sequence[Place] = (),
    airfields: Sequence[Airfield] = (),
    visited: set[int] = frozenset(),
    rings: Sequence[Ring] = (),
    villages: Sequence[Village] = (),
    lighthouses: Sequence[Lighthouse] = (),
    discovered: set[int] = frozenset(),
    fuel_range: float | None = None,
    bridges: Sequence[Bridge] = (),
    flown_under: set[int] = frozenset(),
    helipads: Sequence = (),
) -> None:
    """The in-flight map: terrain, crash sites, saved places, the plane, a scale
    bar and help. With ``fuel_range``, a circle shows how far the fuel reaches
    and airfields beyond it are grey."""
    world_map.draw(x, y, step)
    if fuel_range is not None:
        px, py = world_map.to_screen(x, y)
        canvas.dither(0.5)
        canvas.circb(px, py, fuel_range / step, CAUTION)
        canvas.dither(1.0)
    for village in villages:
        vx, vy = world_map.to_screen(village.x, village.y)
        canvas.rect(vx - 1, vy - 1, 3, 3, pyxel.COLOR_ORANGE)
        shadow_text(vx - len(village.name) * 2, vy + 3, village.name, pyxel.COLOR_PEACH)
    for light in lighthouses:
        lx, ly = world_map.to_screen(light.x, light.y)
        color = pyxel.COLOR_YELLOW if light.number in discovered else pyxel.COLOR_GRAY
        canvas.tri(lx, ly - 4, lx - 2, ly + 1, lx + 2, ly + 1, color)
        if light.number in discovered:
            shadow_text(lx + 4, ly - 3, light.name, pyxel.COLOR_YELLOW)
    for bridge in bridges:
        draw_bridge_marker(world_map, bridge, bridge.number in flown_under)
    for pad in helipads:  # a small H in a box
        hx, hy = world_map.to_screen(pad.x, pad.y)
        canvas.rect(hx - 3, hy - 3, 7, 7, pyxel.COLOR_BLACK)
        canvas.text(hx - 1, hy - 2, "H", pyxel.COLOR_YELLOW)
    for field in airfields:
        reachable = fuel_range is None or math.hypot(field.x - x, field.y - y) <= (
            fuel_range
        )
        draw_runway_marker(world_map, field, field.number in visited, reachable)
    if wrecks is not None:
        for wx, wy, _ in wrecks:
            draw_wreck_marker(*world_map.to_screen(wx, wy))
    for place in places:
        draw_place_marker(*world_map.to_screen(place.x, place.y), place.name)
    # The rest of an active time-trial course, next ring in yellow.
    for i, ring in reversed(list(enumerate(rings))):
        rx, ry = world_map.to_screen(ring.centre[0], ring.centre[1])
        canvas.circb(rx, ry, 2, pyxel.COLOR_YELLOW if i == 0 else pyxel.COLOR_WHITE)
    px, py = world_map.to_screen(x, y)
    if pyxel.frame_count // 10 % 2 == 0:
        draw_plane_marker(px, py, heading, CAUTION)

    w, h = canvas.width(), canvas.height()
    canvas.rect(0, 0, w, 9, pyxel.COLOR_BLACK)
    canvas.text(2, 2, f"MAP  1:{step:g}   X {x:.0f}  Y {y:.0f}", TEXT)
    if fuel_range is not None:
        shadow_text(2, 12, f"FUEL RANGE {fuel_range / 1000:.1f}K", CAUTION)
    canvas.text(w - 4 * 24 - 2, 2, "Z/X ZOOM   M/ESC CLOSE", HUD)

    # Scale bar: a round number of world units close to 40 pixels.
    units = 10 ** math.floor(math.log10(40 * step))
    for nice in (1, 2, 5):
        if units * nice / step <= 48:
            length = units * nice
    bar = length / step
    canvas.rect(4, h - 10, bar, 3, pyxel.COLOR_BLACK)
    canvas.rect(4, h - 9, bar, 1, TEXT)
    shadow_text(8 + bar, h - 12, f"{length:g}", TEXT)
    shadow_text(w - 8, 12, "N", CAUTION)
    canvas.tri(w - 7, 19, w - 9, 24, w - 5, 24, CAUTION)
