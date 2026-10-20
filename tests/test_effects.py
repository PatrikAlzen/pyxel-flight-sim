"""Phase 6 effects: sound selection, chase camera, sprites, water and clouds."""

import math

import numpy as np
import pytest

from flightsim.audio import ENGINE_LEVELS, WIND_LEVELS, SoundState, note_name
from flightsim.render import palette
from flightsim.render.clouds import CloudLayer, CloudSettings
from flightsim.render.sprites import _COLORS, PLANE_REAR
from flightsim.render.voxel import RenderSettings, VoxelRenderer
from flightsim.sim.aircraft import Aircraft, Alert, FlightState
from flightsim.sim.camera import Camera
from flightsim.sim.chase import ChaseCamera, project
from flightsim.terrain.biomes import Biome
from flightsim.terrain.world import TerrainLevel

W, H = 64, 48


def flat(x: float, y: float) -> float:
    return 50.0


def plane(**kwargs) -> Aircraft:
    p = Aircraft()
    p.respawn(0.0, 0.0, 0.0, flat)
    for key, value in kwargs.items():
        setattr(p, key, value)
    return p


# --- sound -------------------------------------------------------------------


def test_note_names():
    assert note_name(0) == "c0"
    assert note_name(33) == "a2"
    assert note_name(13) == "c#1"


def test_engine_follows_throttle_and_stops_after_crash():
    assert SoundState.choose(plane(throttle=0.0), 100).engine == 0
    assert SoundState.choose(plane(throttle=1.0), 100).engine == ENGINE_LEVELS - 1
    crashed = plane(state=FlightState.CRASHED)
    assert SoundState.choose(crashed, 0) == SoundState(None, 0, None)


def test_wind_grows_with_speed():
    levels = [SoundState.choose(plane(speed=v), 100).wind for v in (0, 60, 100, 200)]
    assert levels == sorted(levels)
    assert levels[0] == 0
    assert levels[-1] == WIND_LEVELS - 1


def test_alert_sounds_follow_the_cockpit_alert():
    assert SoundState.choose(plane(speed=30), 300).alert is Alert.STALL
    assert SoundState.choose(plane(vertical_speed=-20), 20).alert is Alert.PULL_UP
    assert SoundState.choose(plane(), 100).alert is None


# --- chase camera ------------------------------------------------------------


def test_chase_camera_sits_behind_and_above():
    p, cam = plane(heading=math.pi / 2), Camera()  # flying east
    ChaseCamera().update(p, cam, 1 / 30, flat)
    assert cam.x < p.x - 20
    assert cam.y == pytest.approx(p.y)
    assert cam.altitude > p.altitude


def test_chase_camera_lags_in_turns_and_catches_up():
    p, cam, chase = plane(), Camera(), ChaseCamera()
    chase.update(p, cam, 1 / 30, flat)
    p.heading = 0.5
    chase.update(p, cam, 1 / 30, flat)
    assert 0 < cam.heading < 0.5
    for _ in range(90):
        chase.update(p, cam, 1 / 30, flat)
    assert cam.heading == pytest.approx(0.5, abs=1e-3)


def test_chase_camera_stays_above_ground():
    p, cam = (
        plane(altitude=51.0, pitch=0.6),
        Camera(),
    )  # climbing steeply off the ground
    ChaseCamera().update(p, cam, 1 / 30, flat)
    assert cam.altitude >= flat(cam.x, cam.y) + 3


def test_plane_projects_ahead_of_chase_camera():
    p, cam = plane(), Camera()
    ChaseCamera().update(p, cam, 1 / 30, flat)
    x, y, depth = project(cam, 128, 256, 144, p.x, p.y, p.altitude)
    assert x == pytest.approx(128)
    assert 72 < y < 144  # below the horizon: the camera looks down at it
    assert depth == pytest.approx(26, abs=1)
    assert project(cam, 128, 256, 144, cam.x, cam.y + 50, 0) is None  # behind


def test_plane_sprite_is_rectangular_and_known_colors():
    assert len({len(row) for row in PLANE_REAR}) == 1
    assert set("".join(PLANE_REAR)) <= set(_COLORS)


# --- water and clouds --------------------------------------------------------


def water_level(size: int = 512) -> TerrainLevel:
    level = TerrainLevel.empty(size, 1.0)
    level.colors[:] = palette.biome_color(Biome.WATER, 3)
    return level


def test_waves_move_but_stay_water():
    renderer = VoxelRenderer(W, H, RenderSettings(z_far=150))
    frames = []
    for t in (0.0, 1.0):
        out = np.zeros((H, W), dtype=np.uint8)
        renderer.render(out, [water_level()], Camera(altitude=20), time=t)
        frames.append(out[H // 2 + 4 :])  # water below the horizon
    assert not np.array_equal(*frames)
    for frame in frames:
        offset = frame.astype(int) - palette.TERRAIN_BASE
        terrain = offset < palette.TERRAIN_COLORS * palette.FOG_LEVELS
        biome = (offset[terrain] % palette.TERRAIN_COLORS) // palette.SHADES
        assert np.all(biome == Biome.WATER)


def test_land_is_not_animated():
    level = TerrainLevel.empty(512, 1.0)
    level.colors[:] = palette.biome_color(Biome.GRASS, 3)
    renderer = VoxelRenderer(W, H, RenderSettings(z_far=150))
    a, b = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    renderer.render(a, [level], Camera(altitude=20), time=0.0)
    renderer.render(b, [level], Camera(altitude=20), time=5.0)
    assert np.array_equal(a, b)


def overcast() -> CloudLayer:
    return CloudLayer(1, CloudSettings(altitude=200, coverage=1.0))


def cloud_pixels(image: np.ndarray) -> np.ndarray:
    snow = palette.biome_color(Biome.SNOW, 0)
    offset = (image.astype(int) - palette.TERRAIN_BASE) % palette.TERRAIN_COLORS
    is_terrain = (image >= palette.TERRAIN_BASE) & (image < palette.SKY_BASE)
    return is_terrain & (offset >= snow - palette.TERRAIN_BASE)


def test_clouds_cover_the_sky_from_below():
    renderer = VoxelRenderer(W, H, RenderSettings(z_far=600))
    out = np.zeros((H, W), dtype=np.uint8)
    flat_ground = TerrainLevel.empty(512, 1.0)
    flat_ground.colors[:] = palette.biome_color(Biome.GRASS, 3)
    renderer.render(
        out, [flat_ground], Camera(altitude=20, pitch=0.5), clouds=overcast()
    )
    assert cloud_pixels(out[:5]).mean() > 0.9  # looking up: solid overcast
    assert not cloud_pixels(out[-3:]).any()  # ground below is unaffected


def test_clouds_seen_from_above_hide_the_ground():
    renderer = VoxelRenderer(W, H, RenderSettings(z_far=600))
    out = np.zeros((H, W), dtype=np.uint8)
    ground = TerrainLevel.empty(512, 1.0)
    ground.colors[:] = palette.biome_color(Biome.GRASS, 3)
    renderer.render(out, [ground], Camera(altitude=400, pitch=-0.6), clouds=overcast())
    assert cloud_pixels(out[-5:]).mean() > 0.9


def test_mountain_in_front_of_clouds_hides_them():
    renderer = VoxelRenderer(W, H, RenderSettings(z_far=600))
    out = np.zeros((H, W), dtype=np.uint8)
    level = TerrainLevel.empty(512, 1.0)
    level.colors[:] = palette.biome_color(Biome.ROCK, 3)
    rows = level.indices(0, np.arange(-40, -30))[0]
    level.heights[rows, :] = 1000  # a wall right in front, taller than the clouds
    renderer.render(out, [level], Camera(altitude=20, pitch=0.5), clouds=overcast())
    assert not cloud_pixels(out).any()
