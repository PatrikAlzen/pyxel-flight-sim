import time

import numpy as np
import pytest

from flightsim.terrain.generator import TerrainGenerator
from flightsim.terrain.world import TerrainLevel, TerrainWorld

LEVELS = [(512, 1.0), (512, 4.0)]  # 4 tiles per side, so safe_radius > 0


@pytest.fixture(scope="module")
def generator():
    return TerrainGenerator(seed=7)


def wait_until_ready(
    world: TerrainWorld, x: float, y: float, timeout: float = 20.0
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        world.update(x, y)
        if world.pending == 0:
            return
        assert time.monotonic() < deadline, "background generation timed out"
        time.sleep(0.005)


def assert_levels_equal(a: list[TerrainLevel], b: list[TerrainLevel]) -> None:
    for la, lb in zip(a, b):
        assert np.array_equal(la.heights, lb.heights)
        assert np.array_equal(la.colors, lb.colors)
        assert np.array_equal(la.slots, lb.slots)


@pytest.mark.parametrize("step", [1.0, 4.0])
def test_level_matches_generator(generator, step):
    level = TerrainLevel.empty(256, step)
    level.fill(generator, 1000.0, -500.0)
    # The camera is at most half a tile from the window centre, so everything
    # within half a tile of the camera is always inside.
    reach = level.tile_extent / 2 - step
    xs = np.array([1000.0, 1000.0 - reach, 1000.0 + reach])
    ys = np.array([-500.0, -500.0 + reach, -500.0 - reach / 3])
    iy, ix = level.indices(xs, ys)
    for x, y, h, c in zip(xs, ys, level.heights[iy, ix], level.colors[iy, ix]):
        sample = generator.sample(x, y, 1, 1, step)
        assert h == sample.height[0, 0]
        assert c == sample.color[0, 0]


def test_level_rejects_bad_size():
    with pytest.raises(ValueError):
        TerrainLevel.empty(100, 1.0)


def test_window_follows_camera_one_tile_at_a_time(generator):
    level = TerrainLevel.empty(512, 1.0)
    assert len(level.missing(0, 0)) == level.tiles**2
    level.fill(generator, 0, 0)
    assert level.missing(0, 0) == []
    # Still inside the same window until the nearest tile corner changes.
    assert level.missing(level.tile_extent * 0.49, 0) == []
    # Crossing it brings in exactly one new column of tiles.
    moved = level.missing(level.tile_extent * 0.51, 0)
    assert len(moved) == level.tiles
    assert {tx for tx, _ in moved} == {level.window(0, 0)[0] + level.tiles}


def test_safe_radius_is_inside_window():
    level = TerrainLevel.empty(512, 2.0)
    for x in np.linspace(-3 * level.tile_extent, 3 * level.tile_extent, 37):
        tx0, _ = level.window(x, 0)
        left, right = tx0 * level.tile_extent, (tx0 + level.tiles) * level.tile_extent
        assert x - left >= level.safe_radius + level.tile_extent / 2 - 1e-9
        assert right - x >= level.safe_radius + level.tile_extent / 2 - 1e-9


def test_incremental_updates_equal_fresh_fill(generator):
    """Flying across several tile borders must give exactly the terrain a fresh start would."""
    world = TerrainWorld(generator, LEVELS)
    try:
        for _ in world.fill(0, 0):
            pass
        path = [(x, x * 0.6) for x in np.linspace(0, 700, 15)]
        for x, y in path:
            wait_until_ready(world, x, y)
        assert world.stale_tiles(*path[-1]) == 0

        fresh = [TerrainLevel.empty(size, step) for size, step in LEVELS]
        for level in fresh:
            level.fill(generator, *path[-1])
        # Every slot now holds the tile the final window wants, either kept from
        # the start or written on the way, so the ring buffers must be
        # bit-for-bit identical to a fresh fill.
        for moved, clean in zip(world.levels, fresh):
            assert moved.missing(*path[-1]) == []
            assert_levels_equal([moved], [clean])
    finally:
        world.close()


def test_background_thread_fills_like_synchronous_fill(generator):
    world = TerrainWorld(generator, LEVELS)
    try:
        assert world.stale_tiles(300, 300) > 0
        wait_until_ready(world, 300, 300)
        assert world.stale_tiles(300, 300) == 0
    finally:
        world.close()
    reference = [TerrainLevel.empty(size, step) for size, step in LEVELS]
    for level in reference:
        level.fill(generator, 300, 300)
    assert_levels_equal(world.levels, reference)


def test_tiles_that_left_the_window_are_not_installed(generator):
    world = TerrainWorld(generator, [(256, 1.0)])
    try:
        world.update(0, 0)
        # Teleport far away before installing anything: old results must be dropped.
        time.sleep(0.2)
        wait_until_ready(world, 10_000, 10_000)
        level = world.levels[0]
        tx0, ty0 = level.window(10_000, 10_000)
        stored = {tuple(t) for t in level.slots.reshape(-1, 2)}
        assert stored == {
            (tx, ty)
            for tx in range(tx0, tx0 + level.tiles)
            for ty in range(ty0, ty0 + level.tiles)
        }
    finally:
        world.close()
