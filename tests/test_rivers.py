import math
import time
from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from flightsim.sim.wrecks import Wrecks
from flightsim.terrain.generator import TerrainGenerator, TerrainSettings

SEED = 1337
DRY = replace(TerrainSettings(), river_fade=(-2.0, -1.0))  # no rivers anywhere


@pytest.fixture(scope="module")
def generator():
    return TerrainGenerator(SEED)


@pytest.fixture(scope="module")
def area(generator):
    """An 8 x 8 km grid every 16 units: world x, y, height and valley depth."""
    step, n = 16.0, 512
    x = (np.arange(n) - n / 2) * step
    xs, ys = np.meshgrid(x, x)
    height, valley = generator._natural(xs, ys)
    return xs, ys, height, valley


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Groups of 8-connected True cells."""
    seen = np.zeros_like(mask)
    groups = []
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        seen[start] = True
        group, queue = [], deque([start])
        while queue:
            r, c = queue.popleft()
            group.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    inside = 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                    if inside and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        queue.append((rr, cc))
        groups.append(group)
    return groups


def test_rivers_run_a_long_way(area):
    xs, ys, _, valley = area
    river = valley > 0.5
    assert 0.003 < river.mean() < 0.1  # some rivers, not a swamp
    longest = max(components(river), key=len)
    rows, cols = np.array(longest).T
    extent = np.hypot(np.ptp(xs[rows, cols]), np.ptp(ys[rows, cols]))
    assert extent > 3000  # one river crosses more than 3 km of the map


def test_river_beds_hold_water(generator, area):
    xs, ys, _, valley = area
    centres = np.argwhere(valley > 0.97)
    assert len(centres) > 20
    rng = np.random.default_rng(0)
    wet = 0
    picks = centres[rng.choice(len(centres), 20, replace=False)]
    for r, c in picks:
        x, y = xs[r, c], ys[r, c]
        sample = generator.sample(x - 32, y - 32, 64, 64)
        wet += bool((sample.ground < generator.settings.sea_level).any())
    assert wet >= 16  # nearly every valley floor has its river


def test_rivers_only_cut_down_and_end_in_the_mountains(area):
    xs, ys, height, _ = area
    dry = TerrainGenerator(SEED, DRY).natural_elevation(xs, ys)
    assert np.all(height <= dry + 1e-6)
    high = dry - DRY.sea_level > TerrainSettings().river_fade[1]
    assert high.any()
    assert np.allclose(height[high], dry[high])


def test_rivers_cost_little(generator):
    """Tile generation with rivers against the same world without, timed in
    turns so a busy machine slows both alike."""
    dry = TerrainGenerator(SEED, DRY)
    tiles = [(x * 128, y * 128) for x in range(-12, 12, 4) for y in range(-12, 12, 4)]

    def run(g) -> float:
        start = time.perf_counter()
        for x, y in tiles:
            g.sample(x, y, 128, 128)
        return time.perf_counter() - start

    run(generator), run(dry)  # warm up (airfield cells, caches)
    wet_best = dry_best = math.inf
    for _ in range(4):
        wet_best = min(wet_best, run(generator))
        dry_best = min(dry_best, run(dry))
    assert wet_best / dry_best < 1.2, wet_best / dry_best  # ~1.12 measured


def test_tiles_and_single_points_agree(generator):
    """Villages and airfields look up single points; tiles are computed in
    bulk. Both must see the same riverbanks."""
    x = np.arange(0.0, 400.0) + 300
    xs, ys = np.meshgrid(x, x - 900)
    tile = generator.natural_elevation(xs, ys)
    rng = np.random.default_rng(0)
    for k in rng.choice(xs.size, 100, replace=False):
        point = generator.natural_elevation(
            np.array([xs.flat[k]]), np.array([ys.flat[k]])
        )
        assert point[0] == tile.flat[k]


def test_old_wrecks_settle_on_the_new_ground():
    wrecks = Wrecks()
    wrecks.add(np.array([10.0, 20.0, 55.0]), 1)  # saved before a valley was cut
    wrecks.add(np.array([-5.0, 0.0, 3.0]))
    wrecks.settle(lambda x, y: np.where(x > 0, 12.0, 4.0))
    assert wrecks.positions[:, 2].tolist() == [12.0, 4.0]
    assert wrecks.kinds.tolist() == [1, 0]
