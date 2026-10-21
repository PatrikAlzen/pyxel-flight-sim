import math
import time

import numpy as np
import pytest

from flightsim.terrain.biomes import Biome
from flightsim.terrain.generator import TerrainGenerator
from flightsim.terrain.objects import Kind, Objects, Tree, hash01, hit, place_objects
from flightsim.terrain.world import TerrainLevel


@pytest.fixture(scope="module")
def generator():
    return TerrainGenerator(seed=1337)


def forest_tile(generator):
    """A tile with plenty of trees in it, found by scanning a row of tiles."""
    for tx in range(-20, 20):
        sample = generator.sample(tx * 128, 512, 128, 128)
        if np.isin(sample.biome, [Biome.FOREST, Biome.TAIGA]).mean() > 0.5:
            return sample
    pytest.skip("no forest found")


def test_hash01_is_deterministic_and_spread():
    ix, iy = np.meshgrid(np.arange(-50, 50), np.arange(-50, 50))
    a = hash01(7, 1, ix, iy)
    assert np.array_equal(a, hash01(7, 1, ix, iy))
    assert not np.array_equal(a, hash01(7, 2, ix, iy))
    assert not np.array_equal(a, hash01(8, 1, ix, iy))
    assert a.min() >= 0 and a.max() < 1
    assert 0.45 < a.mean() < 0.55


def test_trees_grow_in_forests(generator):
    sample = forest_tile(generator)
    objects = place_objects(generator, sample)
    trees = objects.select(objects.kind == Kind.TREE)
    assert len(trees) > 50
    col = (trees.position[:, 0] - sample.x0).astype(int)
    row = (trees.position[:, 1] - sample.y0).astype(int)
    assert not np.isin(
        sample.biome[row, col], [Biome.WATER, Biome.DEEP_WATER, Biome.ROCK]
    ).any()
    assert np.allclose(trees.position[:, 2], sample.height[row, col])
    assert set(trees.variant.tolist()) <= {
        Tree.BROADLEAF,
        Tree.SPRUCE,
        Tree.SNOWY_SPRUCE,
    }


def test_objects_are_seamless_across_tiles(generator):
    """Every object belongs to exactly one tile: two halves give the same as the whole."""
    whole = place_objects(generator, generator.sample(-128, 512, 256, 128))
    left = place_objects(generator, generator.sample(-128, 512, 128, 128))
    right = place_objects(generator, generator.sample(0, 512, 128, 128))
    joined = Objects.concat([left, right])

    def key(o):
        return sorted(map(tuple, np.round(o.position, 3).tolist()))

    assert key(joined) == key(whole)


def test_no_trees_on_runways(generator):
    field = generator.airfields.nearest(0, 0, rings=4)
    x0, y0 = math.floor(field.x) - 256, math.floor(field.y) - 256
    objects = place_objects(generator, generator.sample(x0, y0, 512, 512))
    u, v = field.local(objects.position[:, 0], objects.position[:, 1])
    near_runway = (np.abs(u) < field.length / 2 + 30) & (
        np.abs(v) < field.width / 2 + 60
    )
    assert not near_runway.any()


@pytest.fixture(scope="module")
def area(generator):
    s = generator.settlements
    return s.villages(-8000, -8000, 8000, 8000), s.lighthouses(-8000, -8000, 8000, 8000)


def test_villages_are_on_dry_flat_land_by_water(generator, area):
    villages, _ = area
    assert len(villages) >= 3
    sea = generator.settings.sea_level
    for village in villages:
        assert 4 <= len(village.houses) <= 12
        assert village.name.isalpha() and village.name.isupper()
        ground = np.array([h[2] for h in village.houses])
        assert np.all(ground > sea + 1.5)
        assert ground.max() - ground.min() < 12
        angle = np.linspace(0, math.tau, 32, endpoint=False)
        ring = generator.elevation(
            village.x + 220 * np.cos(angle), village.y + 220 * np.sin(angle)
        )
        assert (ring < sea).any()  # water within reach


def test_lighthouses_stand_on_capes(generator, area):
    _, lighthouses = area
    assert len(lighthouses) >= 3
    sea = generator.settings.sea_level
    for light in lighthouses:
        assert sea + 2 <= light.z <= sea + 30
        angle = np.linspace(0, math.tau, 16, endpoint=False)
        ring = generator.elevation(
            light.x + 60 * np.cos(angle), light.y + 60 * np.sin(angle)
        )
        assert (ring < sea).mean() >= 0.5
        assert light.name.endswith(" LIGHT")


def test_settlements_are_order_independent(area):
    fresh = TerrainGenerator(seed=1337).settlements
    lights = fresh.lighthouses(0, 0, 8000, 8000)  # other order, smaller area first
    villages = fresh.villages(-8000, -8000, 8000, 8000)
    assert set(villages) == set(area[0])
    assert set(lights) <= set(area[1])


def test_houses_and_lighthouses_end_up_in_tiles(generator, area):
    village = area[0][0]
    hx, hy = village.houses[0][:2]
    tx, ty = math.floor(hx / 128), math.floor(hy / 128)
    objects = place_objects(generator, generator.sample(tx * 128, ty * 128, 128, 128))
    houses = objects.select(objects.kind == Kind.HOUSE)
    assert any(np.allclose(p[:2], (hx, hy), atol=1e-3) for p in houses.position)


def test_level_keeps_objects_with_its_tiles(generator):
    level = TerrainLevel.empty(512, 1.0, keeps_objects=True)
    sample = forest_tile(generator)
    x, y = sample.x0 + 64, sample.y0 + 64
    level.fill(generator, x, y)
    near = level.objects_near(x, y, 100)
    assert len(near) > 10
    offset = near.position[:, :2] - (x, y)
    assert np.hypot(offset[:, 0], offset[:, 1]).max() <= 100


def test_hits():
    tree = Objects(
        np.array([[0.0, 0.0, 50.0]], np.float32),
        np.array([Kind.TREE], np.uint8),
        np.array([0], np.uint8),
        np.array([1.0], np.float32),
    )
    assert hit(tree, 0.5, 0.0, 55.0) is Kind.TREE  # into the crown
    assert hit(tree, 0.5, 0.0, 61.0) is None  # just above it
    assert hit(tree, 5.0, 0.0, 55.0) is None  # beside it
    assert hit(Objects.empty(), 0, 0, 0) is None


def test_drawing_800_objects_is_fast_enough():
    """The per-frame budget for objects: 800 sprites drawn through Pyxel."""
    import pyxel

    from flightsim.render.effects import draw_billboards
    from flightsim.render.sprites import BROADLEAF, WORLD_COLORS, make_image
    from flightsim.render.voxel import VoxelRenderer
    from flightsim.sim.camera import Camera

    pyxel.init(256, 144, headless=True)
    renderer = VoxelRenderer(256, 144)
    renderer.pixel_distance[:] = np.inf  # nothing in the way
    sprite = make_image(BROADLEAF, WORLD_COLORS)
    rng = np.random.default_rng(0)
    positions = np.column_stack(
        [rng.uniform(-120, 120, 800), rng.uniform(-250, -20, 800), np.zeros(800)]
    )
    camera = Camera(altitude=15.0)
    full = 10 * renderer.focal / sprite.height
    draw_billboards(sprite, 0, positions, camera, renderer, full)  # warm up
    start = time.perf_counter()
    for _ in range(5):
        drawn = draw_billboards(sprite, 0, positions, camera, renderer, full)
    ms = (time.perf_counter() - start) / 5 * 1000
    assert drawn > 600
    assert ms < 12, ms  # ~3 ms on a quiet machine; generous for slow ones
