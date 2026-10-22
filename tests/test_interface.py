"""The interface layer (384x216): text fits, and the HUD's parts keep apart."""

import itertools
import math

import numpy as np

from flightsim.app import (
    ACHIEVEMENT_ROWS,
    ACHIEVEMENTS_WIDTH,
    PAUSE_WIDTH,
    PLACE_ROWS,
    SHOWCASE,
    UIH,
    UIW,
)
from flightsim.render import canvas, palette
from flightsim.render.hud import MINIMAP_SCALE, MINIMAP_SIZE, Hud
from flightsim.sim.achievements import ACHIEVEMENTS
from flightsim.sim.planes import TRAINER, TYPES, UTILITY, make_vehicle
from flightsim.sim.weather import PRESETS
from flightsim.terrain.world import TerrainLevel

CHAR = 4  # Pyxel's font: pixels per character


def test_menu_and_title_text_fits():
    assert (UIW, UIH) == (384, 216)
    # The pause menu's longest item, selected (with arrows).
    longest = max(
        [f"WEATHER: {p.name}" for p in PRESETS] + [f"AIRCRAFT: {t.name}" for t in TYPES]
    )
    assert (len(longest) + 4) * CHAR <= PAUSE_WIDTH - 8
    # Achievements: the list rows and the description with its date.
    for item in ACHIEVEMENTS:
        assert (len(item.name) + 4) * CHAR <= ACHIEVEMENTS_WIDTH - 16
        assert (len(item.description) + len("  (2026-10-22)")) * CHAR <= (
            ACHIEVEMENTS_WIDTH - 8
        )
    assert 46 + ACHIEVEMENT_ROWS * 8 <= UIH
    assert 34 + PLACE_ROWS * 10 <= UIH
    # Aircraft descriptions left of the turning model on the title screen.
    x0, y0, w, h = SHOWCASE
    assert all(len(t.description) * CHAR <= x0 - 8 for t in TYPES)
    assert x0 + w <= UIW and y0 + h <= UIH - 40  # clear of the bottom band


class FlatWorld:
    """Just enough of a TerrainWorld for the HUD: level ground everywhere."""

    def __init__(self) -> None:
        level = TerrainLevel.empty(512, 16.0)
        level.colors[:] = palette.TERRAIN_BASE + 20
        self.levels = [level]

    def height_at(self, x: float, y: float) -> float:
        return 0.0


def drawn(part) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of what ``part`` draws on the layer."""
    canvas.cls(palette.CLEAR)
    part()
    ys, xs = np.nonzero(canvas.array() != palette.CLEAR)
    assert len(xs), part
    return xs.min(), ys.min(), xs.max(), ys.max()


def hud_boxes(hud: Hud, plane, world: FlatWorld) -> dict:
    """Where each part of the HUD draws, for this aircraft."""
    boxes = {
        "compass": drawn(lambda: hud.compass(plane.heading)),
        "speed": drawn(lambda: hud.speed_tape(plane)),
        "altitude": drawn(lambda: hud.altitude_tape(plane, 120.0)),
        "throttle": drawn(lambda: hud.throttle(plane.throttle)),
        "fuel": drawn(lambda: hud.fuel(plane)),
        "wind": drawn(lambda: hud.wind(plane, (6.0, 3.0))),
        "minimap": drawn(lambda: hud.minimap(plane, world)),
    }
    if plane.settings.stall_speed == 0:  # a helicopter
        boxes["drift"] = drawn(lambda: hud.drift(plane))
    return boxes


def test_hud_parts_keep_apart_on_the_interface_layer(screen):
    layer = screen.Image(UIW, UIH)
    hud = Hud(192.0, 4.0)
    world = FlatWorld()
    canvas.use(layer)
    try:
        for plane_type in (TRAINER, UTILITY):
            plane = make_vehicle(plane_type, 0.0)
            plane.respawn(0.0, 0.0, 0.0, world.height_at)
            plane.uses_fuel = True
            boxes = hud_boxes(hud, plane, world)
            for (a, ra), (b, rb) in itertools.combinations(boxes.items(), 2):
                apart = ra[2] < rb[0] or rb[2] < ra[0] or ra[3] < rb[1] or rb[3] < ra[1]
                assert apart, (plane_type.name, a, ra, b, rb)
            for name, (x0, y0, x1, y1) in boxes.items():
                assert 0 <= x0 and x1 < UIW and 0 <= y0 and y1 < UIH, name
    finally:
        canvas.use(None)


def test_minimap_shows_traffic_where_it_is(screen):
    """Nose up: an aircraft ahead is above the middle, one to the right is
    to the right of it, at MINIMAP_SCALE world units per pixel."""
    layer = screen.Image(UIW, UIH)
    hud = Hud(192.0, 4.0)
    world = FlatWorld()
    plane = make_vehicle(TRAINER, 0.0)
    plane.respawn(0.0, 0.0, math.pi / 2, world.height_at)  # nose east
    cx = UIW - MINIMAP_SIZE - 4 + MINIMAP_SIZE // 2
    cy = UIH - MINIMAP_SIZE - 4 + MINIMAP_SIZE // 2
    canvas.use(layer)
    try:
        for other, (dx, dy) in (
            (np.array([200.0, 0.0, plane.altitude]), (0, -20)),  # ahead
            (np.array([0.0, 150.0, plane.altitude]), (15, 0)),  # south: right
        ):
            canvas.cls(palette.CLEAR)
            hud.minimap(plane, world, [other])
            spot = canvas.array()[cy + dy : cy + dy + 2, cx + dx : cx + dx + 2]
            assert (spot == screen.COLOR_YELLOW).all(), (dx, dy)
    finally:
        canvas.use(None)
    assert MINIMAP_SCALE * MINIMAP_SIZE >= 600  # it still shows the area around
