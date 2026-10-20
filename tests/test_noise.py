import numpy as np
import pytest

from flightsim.terrain.noise import FractalNoise, GradientNoise, domain_warp, smoothstep


@pytest.fixture
def grid():
    rng = np.random.default_rng(0)
    return rng.uniform(-2000, 2000, size=(2, 200, 200))


def test_gradient_noise_is_deterministic(grid):
    x, y = grid
    assert np.array_equal(GradientNoise(5)(x, y), GradientNoise(5)(x, y))


def test_different_seeds_give_different_noise(grid):
    x, y = grid
    assert not np.allclose(GradientNoise(5)(x, y), GradientNoise(6)(x, y))


def test_gradient_noise_is_zero_on_lattice_points():
    x, y = np.meshgrid(np.arange(-20, 20), np.arange(-20, 20))
    assert np.allclose(GradientNoise(1)(x, y), 0.0, atol=1e-6)


def test_gradient_noise_range(grid):
    x, y = grid
    n = GradientNoise(3)(x, y)
    assert n.min() >= -1.0001
    assert n.max() <= 1.0001
    assert n.std() > 0.1


def test_gradient_noise_is_continuous_across_cell_borders():
    noise = GradientNoise(2)
    x = np.arange(-50, 50, dtype=np.float64)
    y = np.full_like(x, 3.7)
    eps = 1e-6
    assert np.allclose(noise(x - eps, y), noise(x + eps, y), atol=1e-4)


def test_noise_broadcasts():
    noise = FractalNoise(1, octaves=3)
    out = noise.fbm(np.arange(10)[None, :] * 0.1, np.arange(4)[:, None] * 0.1)
    assert out.shape == (4, 10)


def test_fbm_and_ridged_ranges(grid):
    x, y = grid / 100
    noise = FractalNoise(9, octaves=5)
    fbm = noise.fbm(x, y)
    ridged = noise.ridged(x, y)
    assert -1.0 <= fbm.min() and fbm.max() <= 1.0
    assert 0.0 <= ridged.min() and ridged.max() <= 1.0
    assert ridged.std() > 0.05


def test_domain_warp_moves_points_by_at_most_strength(grid):
    x, y = grid
    wx, wy = domain_warp(x, y, FractalNoise(1, 3), FractalNoise(2, 3), strength=10.0)
    assert np.abs(wx - x).max() <= 10.0
    assert np.abs(wy - y).max() <= 10.0
    assert np.abs(wx - x).max() > 1.0


def test_smoothstep():
    assert np.allclose(
        smoothstep(0, 1, np.array([-1.0, 0.0, 0.5, 1.0, 2.0])), [0, 0, 0.5, 1, 1]
    )
