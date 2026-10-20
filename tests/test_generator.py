import numpy as np
import pytest

from flightsim.render import palette
from flightsim.terrain.biomes import Biome
from flightsim.terrain.generator import TerrainGenerator

FIELDS = ["height", "ground", "moisture", "biome", "color"]


@pytest.fixture(scope="module")
def generator():
    return TerrainGenerator(seed=1337)


def test_same_seed_same_terrain(generator):
    a = generator.sample(100, -300, 32, 32)
    b = TerrainGenerator(seed=1337).sample(100, -300, 32, 32)
    for field in FIELDS:
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_different_seed_different_terrain(generator):
    a = generator.sample(0, 0, 32, 32, step=16)
    b = TerrainGenerator(seed=1338).sample(0, 0, 32, 32, step=16)
    assert not np.allclose(a.ground, b.ground)


@pytest.mark.parametrize("step", [1, 4])
def test_adjacent_samples_have_no_seams(generator, step):
    """Two neighbouring chunks must match one big sample exactly, including shading."""
    size = 64
    whole = generator.sample(-size * step, 0, 2 * size, size, step)
    left = generator.sample(-size * step, 0, size, size, step)
    right = generator.sample(0, 0, size, size, step)
    for field in FIELDS:
        joined = np.hstack([getattr(left, field), getattr(right, field)])
        assert np.array_equal(joined, getattr(whole, field)), field


def test_sample_shapes_and_types(generator):
    s = generator.sample(0, 0, 40, 24)
    for field in FIELDS:
        assert getattr(s, field).shape == (24, 40), field
    assert s.color.dtype == np.uint8
    assert s.biome.dtype == np.uint8


def test_surface_is_flat_water_at_sea_level(generator):
    s = generator.sample(0, 0, 128, 128, step=32)
    sea = generator.settings.sea_level
    assert np.all(s.height >= sea)
    under = s.ground < sea
    assert under.any()
    assert np.all(s.height[under] == sea)
    assert np.all(np.isin(s.biome[under], [Biome.WATER, Biome.DEEP_WATER]))


def test_overview_has_varied_world(generator):
    """A zoomed-out view should contain sea, lowland and mountains."""
    s = generator.sample(-4096, -4096, 128, 128, step=64)
    present = set(np.unique(s.biome).tolist())
    assert {Biome.DEEP_WATER, Biome.GRASS, Biome.ROCK} <= present
    assert len(present) >= 6
    assert s.ground.max() > generator.settings.snow_line


def test_colors_are_valid_palette_indices(generator):
    s = generator.sample(0, 0, 64, 64, step=8)
    assert s.color.min() >= palette.TERRAIN_BASE
    assert s.color.max() < len(palette.colors())


def test_palette_fits_pyxel():
    assert len(palette.colors()) <= 256
    assert (
        palette.biome_color(Biome.SNOW, palette.SHADES - 1) == len(palette.colors()) - 1
    )
