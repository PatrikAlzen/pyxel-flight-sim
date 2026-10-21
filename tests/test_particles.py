import numpy as np
import pytest

from flightsim.render.effects import in_front_of_terrain, project_points
from flightsim.sim.camera import Camera
from flightsim.sim.particles import Kind, Particles, ParticleSettings

SITE = np.array([0.0, 0.0, 50.0])
DT = 1 / 30
CALM = (0.0, 0.0)


def test_explosion_makes_fire_smoke_and_sparks():
    p = Particles()
    p.explode(SITE)
    kinds = set(np.unique(p.kind).tolist())
    assert kinds == {Kind.SMOKE, Kind.FIRE, Kind.SPARK}
    assert np.all(p.position[:, 2] >= SITE[2] - 2)  # nothing starts underground


def test_burning_emits_at_a_steady_rate():
    p = Particles(ParticleSettings(smoke_rate=30, fire_rate=0))
    for _ in range(30):
        p.burn(SITE, DT)
    assert len(p) == pytest.approx(30, abs=1)
    assert np.all(p.kind == Kind.SMOKE)


def test_smoke_rises_and_drifts_with_the_wind():
    p = Particles(ParticleSettings(smoke_rate=60, fire_rate=0))
    p.burn(SITE, 0.5)
    for _ in range(45):
        p.update(DT, (10.0, 0.0))
    assert p.position[:, 2].mean() > SITE[2] + 10
    assert p.position[:, 0].mean() > SITE[0] + 5


def test_sparks_fall_back_down():
    p = Particles()
    p.explode(SITE)
    for _ in range(round(1.3 / DT)):
        p.update(DT, CALM)
    sparks = p.kind == Kind.SPARK
    assert sparks.any()
    assert p.velocity[sparks, 2].mean() < 0  # past the top of their arc


def test_particles_age_out():
    p = Particles()
    p.explode(SITE)
    for _ in range(round(8 / DT)):
        p.update(DT, CALM)
    assert len(p) == 0


def test_capacity_drops_the_oldest():
    p = Particles(ParticleSettings(capacity=50))
    p.explode(SITE)
    assert len(p) == 50
    p.update(0.3, CALM)
    p.explode(SITE)
    assert len(p) == 50
    assert np.all(p.age == 0)  # the old ones made room


def test_smoke_fades_and_grows():
    p = Particles(ParticleSettings(smoke_rate=30, fire_rate=0))
    p.burn(SITE, 0.2)
    r0, f0 = p.radius.copy(), p.fade.copy()
    p.update(1.0, CALM)
    assert np.all(p.radius > r0)
    assert np.all(p.fade < f0)


def test_points_behind_a_ridge_are_hidden():
    camera = Camera(x=0.0, y=0.0, altitude=20.0)  # looking north (-y)
    points = np.array([[0.0, -50.0, 20.0], [0.0, -200.0, 20.0]])
    sx, sy, depth = project_points(camera, points, 128, 256, 144)
    assert np.allclose(depth, [50, 200])
    assert np.allclose(sx, 128) and np.allclose(sy, 72)

    class Ridge:
        pixel_distance = np.full((144, 256), 100.0)  # terrain 100 units away everywhere

    visible = in_front_of_terrain(Ridge, camera, points, sx, sy)
    assert visible.tolist() == [True, False]


def test_old_wrecks_smoulder_thinly():
    p = Particles()
    wrecks = np.array([[0.0, 0.0, 10.0], [500.0, 0.0, 20.0]])
    for _ in range(round(4 / DT)):
        p.smoulder(wrecks, DT)
        p.update(DT, CALM)
    near_first = np.hypot(p.position[:, 0], p.position[:, 1]) < 50
    assert near_first.any() and (~near_first).any()  # both wrecks smoke
    assert 10 < len(p) < 40  # a wisp, not a fire: ~2.5 per second each
    assert np.all(p.kind == Kind.SMOKE)
    assert p.position[:, 2].max() > 30  # and it rises high


def test_billboard_on_the_ground_does_not_blink_at_a_shallow_angle():
    """A sprite whose foot is a little off the drawn ground must stay visible
    as the camera moves; it used to vanish every few frames."""
    from flightsim.render.effects import billboard_visible
    from flightsim.render.voxel import RenderSettings, VoxelRenderer
    from flightsim.terrain.world import TerrainLevel

    level = TerrainLevel.empty(1024, 1.0)
    level.heights[:] = 10.0
    renderer = VoxelRenderer(256, 144, RenderSettings(z_far=600))
    out = np.zeros((144, 256), dtype=np.uint8)
    sprite = np.array([[0.0, -300.0, 9.5]])  # foot half a unit below the ground
    seen = []
    for step in range(30):
        camera = Camera(x=0.0, y=-step * 1.7, altitude=40.0, pitch=-0.1)
        renderer.render(out, [level], camera)
        seen.append(bool(billboard_visible(renderer, camera, sprite, height=12.0)[0]))
    assert all(seen)

    # Behind a wall it is hidden.
    wall = level.indices(0, np.arange(-120, -110))[0]
    level.heights[wall, :] = 200.0
    camera = Camera(x=0.0, y=0.0, altitude=40.0)
    renderer.render(out, [level], camera)
    assert not billboard_visible(renderer, camera, sprite, height=12.0)[0]
