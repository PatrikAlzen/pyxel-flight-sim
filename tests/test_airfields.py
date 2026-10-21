import math

import numpy as np
import pytest

from flightsim.sim.aircraft import Aircraft, Controls, FlightState
from flightsim.terrain.airfields import Airfields, landing_report
from flightsim.terrain.biomes import Biome
from flightsim.terrain.generator import TerrainGenerator

CELLS = [(cx, cy) for cx in range(-5, 5) for cy in range(-5, 5)]


@pytest.fixture(scope="module")
def generator():
    return TerrainGenerator(seed=1337)


@pytest.fixture(scope="module")
def fields(generator):
    found = [generator.airfields.in_cell(cx, cy) for cx, cy in CELLS]
    return [f for f in found if f is not None]


def test_placement_is_deterministic_and_order_independent(generator):
    fresh = TerrainGenerator(seed=1337).airfields
    backwards = [fresh.in_cell(cx, cy) for cx, cy in reversed(CELLS)]
    forwards = [generator.airfields.in_cell(cx, cy) for cx, cy in CELLS]
    assert forwards == backwards[::-1]


def test_some_but_not_all_cells_have_an_airfield(fields):
    assert 0.15 < len(fields) / len(CELLS) < 0.6


def test_runways_are_flat_dry_and_low(generator, fields):
    s = generator.settings
    u, v = np.meshgrid(np.linspace(-150, 150, 41), np.linspace(-8, 8, 5))
    for field in fields:
        x, y = field.point(u, v)
        heights = generator.elevation(np.asarray(x), np.asarray(y))
        assert np.allclose(heights, field.elevation, atol=1e-6)
        assert s.sea_level + 3 <= field.elevation <= s.sea_level + 80


def test_flattening_stays_inside_the_cell(generator, fields):
    a = generator.airfields
    cell = a.settings.cell
    for field in fields:
        cx, cy = math.floor(field.x / cell), math.floor(field.y / cell)
        assert cx * cell + a.reach <= field.x <= (cx + 1) * cell - a.reach
        assert cy * cell + a.reach <= field.y <= (cy + 1) * cell - a.reach


def test_terrain_is_seamless_across_a_runway(generator, fields):
    """Tiles cutting through a runway must line up exactly, flattening included."""
    field = fields[0]
    x0, y0 = math.floor(field.x) - 64, math.floor(field.y) - 32
    whole = generator.sample(x0, y0, 128, 64)
    left = generator.sample(x0, y0, 64, 64)
    right = generator.sample(x0 + 64, y0, 64, 64)
    for name in ("height", "biome", "color"):
        joined = np.hstack([getattr(left, name), getattr(right, name)])
        assert np.array_equal(joined, getattr(whole, name)), name


def test_runway_is_painted(generator, fields):
    field = fields[0]
    s = generator.sample(math.floor(field.x) - 100, math.floor(field.y) - 100, 200, 200)
    xs = s.x0 + np.arange(200)[None, :]
    ys = s.y0 + np.arange(200)[:, None]
    on = field.on_runway(*np.broadcast_arrays(xs, ys))
    assert on.sum() > 1000
    assert np.all(s.biome[on] == Biome.RUNWAY)
    assert not np.any(s.biome[~on] == Biome.RUNWAY)
    # Both asphalt and paint are there.
    assert len(np.unique(s.color[on])) >= 2


def test_runway_at(fields, generator):
    field = fields[0]
    assert generator.airfields.runway_at(*field.point(100, 3)) == field
    assert generator.airfields.runway_at(*field.point(100, 30)) is None
    assert generator.airfields.runway_at(*field.point(200, 0)) is None


def test_nearest(generator, fields):
    field = fields[0]
    near = generator.airfields.nearest(*field.point(10, 40))
    assert near == field


def test_no_airfields_where_nothing_is_flat():
    a = Airfields(1, lambda x, y: np.sin(np.asarray(x) * 0.3) * 50 + 40, sea_level=0)
    assert all(a.in_cell(cx, cy) is None for cx in range(-3, 3) for cy in range(-3, 3))


@pytest.mark.parametrize(
    ("sink", "grade"), [(0.5, "BUTTER"), (2.0, "SMOOTH"), (5.0, "FIRM"), (7.0, "HARD")]
)
def test_landing_report(fields, sink, grade):
    field = fields[0]
    report = landing_report(field, *field.point(0, 3), sink)
    assert report.startswith(f"FIELD {field.number}: {grade}")
    assert "3 OFF CENTRE" in report


def test_parked_plane_can_take_off(generator, fields):
    field = fields[0]
    ground = lambda x, y: float(generator.elevation(np.array(x), np.array(y)))
    plane = Aircraft()
    plane.park(*field.point(-130), field.heading, ground)
    assert plane.state is FlightState.LANDED and plane.speed == 0
    for _ in range(round(12 / (1 / 30))):
        throttle_and_rotate = Controls(throttle=1, pitch=0.4 if plane.speed > 62 else 0)
        plane.update(throttle_and_rotate, 1 / 30, ground)
        if plane.state is FlightState.FLYING and plane.altitude > field.elevation + 10:
            break
    assert plane.state is FlightState.FLYING
    assert plane.crash_reason == ""


def test_touchdown_sink_is_recorded(fields):
    def flat(x, y):
        return 50.0

    plane = Aircraft()
    plane.respawn(0, 0, 0, flat)
    plane.altitude, plane.speed, plane.throttle, plane.pitch = (
        53,
        65,
        0.25,
        math.radians(-2),
    )
    for _ in range(600):
        plane.update(Controls(), 1 / 30, flat)
        if plane.state is FlightState.LANDED:
            break
    assert plane.state is FlightState.LANDED
    assert 0 < plane.touchdown_sink < 8
