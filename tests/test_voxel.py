import math

import numpy as np

from flightsim.render import palette
from flightsim.render.voxel import RenderSettings, VoxelRenderer
from flightsim.sim.camera import Camera
from flightsim.terrain.world import TerrainLevel

W, H = 64, 48
GROUND = palette.biome_color(3, 2).item()
WALL = palette.biome_color(8, 4).item()
FOG_END = palette.TERRAIN_BASE + palette.TERRAIN_COLORS


def flat_level(size: int = 512) -> TerrainLevel:
    level = TerrainLevel.empty(size, 1.0)
    level.colors[:] = GROUND
    return level


def render(levels, camera, settings=None) -> np.ndarray:
    renderer = VoxelRenderer(W, H, settings or RenderSettings(z_far=150))
    out = np.zeros((H, W), dtype=np.uint8)
    renderer.render(out, levels, camera)
    return out


def is_sky(px) -> np.ndarray:
    return np.asarray(px) >= palette.SKY_BASE


def test_flat_ground_below_horizon_sky_above():
    out = render([flat_level()], Camera(altitude=20))
    assert is_sky(out[: H // 2 - 1]).all()
    assert np.all(out[-1] == GROUND)


def test_pitch_moves_horizon():
    level = [flat_level()]
    up = render(level, Camera(altitude=20, pitch=math.radians(10)))
    down = render(level, Camera(altitude=20, pitch=math.radians(-10)))
    # Looking up shows more sky, looking down more ground.
    assert is_sky(up).sum() > is_sky(down).sum()


def test_roll_tilts_horizon():
    out = render([flat_level()], Camera(altitude=20, roll=math.radians(20)))
    # Right wing down: the horizon is higher on the right side of the screen.
    ground_rows = (~is_sky(out)).sum(axis=0)
    assert ground_rows[-1] > ground_rows[0]


def test_near_wall_hides_terrain_behind_it():
    """A wall across the view occludes everything behind it."""
    level = flat_level()
    camera = Camera(x=0.5, y=0.5, altitude=20, heading=0.0)  # looking north (-y)
    wall_rows = level.indices(0, np.arange(-40, -30))[0]
    level.heights[wall_rows, :] = 40
    level.colors[wall_rows, :] = WALL
    out = render([level], camera)
    middle = out[:, W // 2]
    # From the bottom up: near ground, then the wall up to its top edge, then sky.
    assert middle[-1] == GROUND
    wall = np.flatnonzero(middle == WALL)
    assert len(wall) > 5
    assert is_sky(middle[: wall.min()]).all()
    assert np.all(middle[wall.min() :][middle[wall.min() :] != WALL] == GROUND)


def test_distant_terrain_is_fogged():
    settings = RenderSettings(z_far=150, fog_start=0.3)
    renderer = VoxelRenderer(W, H, settings)
    out = np.zeros((H, W), dtype=np.uint8)
    renderer.render(out, [flat_level()], Camera(altitude=20))
    # The ground row just below the horizon is far away and must be fogged.
    ground = out[~is_sky(out)]
    far_row = out[np.flatnonzero((~is_sky(out)).any(axis=1)).min()]
    assert np.any(far_row != GROUND)
    assert np.all((far_row == GROUND) | (far_row >= FOG_END - palette.TERRAIN_COLORS))
    assert np.any(ground == GROUND)


def test_coarse_level_used_for_far_slices():
    renderer = VoxelRenderer(W, H, RenderSettings(z_far=700))
    fine, coarse = TerrainLevel.empty(512, 1.0), TerrainLevel.empty(512, 8.0)
    ranges = renderer.depth_ranges([fine, coarse])
    assert [lvl for lvl, _ in ranges] == [fine, coarse]
    (_, near), (_, far) = ranges
    assert near.start == 0 and near.stop == far.start and far.stop == len(renderer.z)


def test_euler_and_vectors_round_trip():
    from flightsim.sim.camera import basis_to_euler, euler_to_basis

    rng = np.random.default_rng(3)
    for _ in range(200):
        heading = rng.uniform(0, 2 * math.pi)
        pitch = rng.uniform(-1.5, 1.5)
        roll = rng.uniform(-math.pi, math.pi)
        f, r, u = euler_to_basis(heading, pitch, roll)
        assert np.allclose(np.cross(u, f), r)
        h2, p2, r2 = basis_to_euler(f, u)
        f2, _, u2 = euler_to_basis(h2, p2, r2)
        assert np.allclose(f, f2) and np.allclose(u, u2)


def test_upside_down_shows_ground_on_top():
    out = render([flat_level()], Camera(altitude=20, roll=math.pi))
    assert np.all(out[0] == GROUND)
    assert is_sky(out[-H // 2 + 1 :]).all()


def test_roll_ninety_puts_horizon_vertical():
    out = render([flat_level()], Camera(altitude=20, roll=math.radians(90)))
    # Right wing down: the ground is on the right half of the screen.
    assert np.all(out[:, -1] == GROUND)
    assert is_sky(out[:, : W // 2 - 1]).all()


def test_looking_straight_down_sees_only_ground():
    out = render([flat_level()], Camera(altitude=20, pitch=-math.pi / 2))
    assert not is_sky(out).any()


def test_roll_does_not_stretch_the_edges():
    """Rolling is a rotation: a feature at the screen edge keeps its size."""
    level = flat_level()
    camera = Camera(x=0.5, y=0.5, altitude=20)
    stripes = level.indices(0, np.arange(-60, -40))[0]
    level.colors[stripes, :] = WALL
    level_view = render([level], camera)
    camera.roll = math.radians(60)
    rolled = render([level], camera)
    assert (
        abs(int((rolled == WALL).sum()) - int((level_view == WALL).sum()))
        < 0.15 * (level_view == WALL).sum()
    )
