import math
import time

import numpy as np
import pytest

from flightsim.persist import SaveStore, WorldSave
from flightsim.sim.camera import Camera
from flightsim.terrain.bridges import Bridge, Bridges
from flightsim.terrain.generator import TerrainGenerator
from flightsim.terrain.objects import Kind, place_objects

SEED = 1337
AREA = (-4000.0, -4000.0, 4000.0, 4000.0)


@pytest.fixture(scope="module")
def generator():
    return TerrainGenerator(SEED)


@pytest.fixture(scope="module")
def bridges(generator):
    return generator.bridges.overlapping(*AREA)


def test_bridges_are_found_and_placement_does_not_depend_on_order(generator, bridges):
    assert len(bridges) >= 5
    fresh = Bridges(SEED, TerrainGenerator(SEED))
    later = fresh.overlapping(0.0, 0.0, 4000.0, 4000.0)  # a part first
    everything = fresh.overlapping(*AREA)
    assert set(everything) == set(bridges)
    assert set(later) <= set(bridges)
    assert len({b.number for b in bridges}) == len(bridges)


def test_every_bridge_spans_a_river_with_room_underneath(generator, bridges):
    s = generator.bridges.settings
    sea = generator.settings.sea_level
    for b in bridges:
        assert s.clearance[0] <= b.bottom - sea <= s.clearance[1]
        assert s.span[0] <= 2 * (b.half_span - s.embed) <= s.span[1]
        t = np.linspace(-b.half_span, b.half_span, 200)
        ground = generator.elevation(b.x + t * b.ax, b.y + t * b.ay)
        assert (ground < sea).any()  # water underneath
        ends = np.array([-1.0, 1.0]) * (b.half_span - s.embed)
        assert np.all(
            generator.elevation(b.x + ends * b.ax, b.y + ends * b.ay) >= b.deck - 1.0
        )
        for px, py, z in b.pillars:
            assert z < b.bottom - 4
            along, _ = b.local(px, py)
            assert abs(along) < b.half_span


def test_no_trees_on_the_bridge_ends(generator, bridges):
    b = bridges[0]
    x0 = math.floor((b.x - 128) / 128) * 128
    y0 = math.floor((b.y - 128) / 128) * 128
    objects = place_objects(generator, generator.sample(x0, y0, 384, 384))
    trees = objects.select(objects.kind == Kind.TREE)
    along, across = b.local(trees.position[:, 0], trees.position[:, 1])
    on_deck = (np.abs(along) < b.half_span) & (np.abs(across) < b.width / 2 + 2)
    assert not on_deck.any()


BRIDGE = Bridge(
    number=1,
    name="TEST BRIDGE",
    x=0.0,
    y=0.0,
    ax=1.0,  # deck runs east-west, the river north-south
    ay=0.0,
    half_span=40.0,
    deck=20.0,
    thickness=2.0,
    width=6.0,
    railing=1.2,
    pillars=((30.0, 0.0, 5.0),),
)


def test_deck_railings_and_pillars_are_solid():
    assert BRIDGE.hit(0.0, 0.0, 19.0)  # in the deck
    assert BRIDGE.hit(10.0, 2.5, 21.0)  # the railing
    assert BRIDGE.hit(30.0, 0.5, 10.0)  # a pillar
    assert not BRIDGE.hit(0.0, 0.0, 10.0)  # underneath: open
    assert not BRIDGE.hit(0.0, 0.0, 25.0)  # well above
    assert not BRIDGE.hit(0.0, 20.0, 19.0)  # beside


@pytest.mark.parametrize(
    ("start", "end", "under"),
    [
        ((0.0, -20.0, 10.0), (0.0, 20.0, 10.0), True),  # northwards, under
        ((5.0, 20.0, 12.0), (-5.0, -20.0, 12.0), True),  # southwards, at an angle
        ((0.0, -20.0, 30.0), (0.0, 20.0, 30.0), False),  # over the top
        ((60.0, -20.0, 10.0), (60.0, 20.0, 10.0), False),  # past the end
        ((0.0, -20.0, 10.0), (0.0, -5.0, 10.0), False),  # not through yet
    ],
)
def test_passing_under(start, end, under):
    assert BRIDGE.passed_under(start, end) is under


def draw(camera, bridges):
    from flightsim.render.bridges import draw_bridges
    from flightsim.render.framebuffer import screen_array
    from flightsim.render.voxel import VoxelRenderer

    renderer = VoxelRenderer(256, 144)
    renderer.pixel_distance[:] = np.inf  # nothing in the way
    screen_array()[:] = 0
    drawn = draw_bridges(bridges, camera, renderer, 1500.0)
    return drawn, screen_array().copy()


def test_a_deck_right_overhead_still_shows(screen):
    # Under the middle of the deck, looking north along the river.
    camera = Camera(x=0.0, y=0.0, altitude=12.0, heading=0.0, pitch=1.2)
    drawn, image = draw(camera, [BRIDGE])
    assert drawn > 0
    assert (image != 0).mean() > 0.5  # looking up, the underside fills the view


def test_bridges_hide_behind_terrain_and_far_away(screen):
    from flightsim.render.bridges import draw_bridges
    from flightsim.render.voxel import VoxelRenderer

    camera = Camera(x=0.0, y=200.0, altitude=15.0)  # south of it, looking north
    renderer = VoxelRenderer(256, 144)
    renderer.pixel_distance[:] = 50.0  # a hill 50 units ahead covers everything
    assert draw_bridges([BRIDGE], camera, renderer, 1500.0) == 0
    far = Camera(x=0.0, y=5000.0, altitude=15.0)
    assert draw(far, [BRIDGE])[0] == 0


def test_drawing_ten_bridges_is_quick(screen):
    many = [
        Bridge(**{**BRIDGE.__dict__, "number": i, "x": (i - 5) * 60.0, "y": -150.0})
        for i in range(10)
    ]
    camera = Camera(x=0.0, y=100.0, altitude=30.0, pitch=-0.1)
    draw(camera, many)
    start = time.perf_counter()
    for _ in range(5):
        drawn, _ = draw(camera, many)
    ms = (time.perf_counter() - start) / 5 * 1000
    assert drawn > 50
    assert ms < 15, ms  # ~4 ms measured (0.45 ms per bridge)


def test_bridges_flown_under_are_saved(tmp_path):
    store = SaveStore(tmp_path)
    store.save(WorldSave(3, bridges=[2147516415]))
    assert store.load(3).bridges == [2147516415]
