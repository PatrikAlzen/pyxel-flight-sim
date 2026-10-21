import math

import numpy as np
import pytest

from flightsim.controls import Controls
from flightsim.render.clouds import CloudLayer
from flightsim.render.effects import project_points
from flightsim.render.rain import Rain
from flightsim.sim.aircraft import Aircraft, FlightState
from flightsim.sim.camera import Camera
from flightsim.sim.weather import (
    PRESETS,
    Weather,
    WeatherSettings,
    blend,
    turbulence_intensity,
)

DT = 1 / 30
CLEAR, _, _, RAIN, STORM = PRESETS


def flat(_x, _y):
    return 50.0  # dry land, above the sea


def test_wind_carries_a_flying_plane_sideways():
    calm, windy = Aircraft(), Aircraft()
    for plane in (calm, windy):
        plane.respawn(0.0, 0.0, 0.0, flat)  # heading north (-y)
    for _ in range(60):
        calm.update(Controls(), DT, flat)
        windy.update(Controls(), DT, flat, wind=(10.0, 0.0))  # blowing east
    assert windy.x - calm.x == pytest.approx(20.0, abs=0.5)
    assert windy.y == pytest.approx(calm.y, abs=0.5)


def test_wind_does_not_move_a_parked_plane():
    plane = Aircraft()
    plane.park(0.0, 0.0, 0.0, flat)
    plane.throttle = 0.0
    for _ in range(30):
        plane.update(Controls(), DT, flat, wind=(20.0, 0.0))
    assert plane.state is FlightState.LANDED
    assert plane.x == pytest.approx(0.0, abs=1e-6)


def test_bumps_turn_the_plane_only_in_the_air():
    level, bumped = Aircraft(), Aircraft()
    for plane in (level, bumped):
        plane.respawn(0.0, 0.0, 0.0, flat)
    for _ in range(15):
        level.update(Controls(), DT, flat)
        bumped.update(Controls(), DT, flat, bump=(0.3, 0.0))
    assert bumped.pitch > level.pitch + 0.1


def test_blend_is_continuous_between_presets():
    a, b = CLEAR, STORM
    assert blend(a, b, 0.0) == a
    assert blend(a, b, 1.0).wind == b.wind
    steps = [blend(a, b, t).cloud_cover for t in np.linspace(0, 1, 50)]
    assert np.all(np.diff(steps) > 0)
    assert max(np.diff(steps)) < 0.03


def test_weather_changes_smoothly_on_its_own():
    weather = Weather(3, WeatherSettings(hold=(5.0, 5.0), transition=4.0))
    names, covers = set(), []
    for _ in range(round(120 / DT)):
        weather.update(DT)
        names.add(weather.current.name)
        covers.append(weather.current.cloud_cover)
    assert len(names) > 1  # it did change
    assert max(abs(np.diff(covers))) < 0.02  # but never jumped


def test_same_seed_same_weather():
    a, b = Weather(9), Weather(9)
    for _ in range(round(600 / DT)):
        a.update(DT)
        b.update(DT)
    assert a.current == b.current and a.wind() == b.wind()


def test_lock_converges_to_the_preset_and_stays():
    weather = Weather(1)
    weather.lock(STORM)
    assert weather.mode == "STORM"
    for _ in range(round(600 / DT)):
        weather.update(DT)
    assert weather.current == STORM
    weather.lock(None)
    assert weather.mode == "AUTO"


def test_wind_speed_follows_the_conditions():
    weather = Weather(5)
    weather.lock(STORM)
    speeds = []
    for _ in range(round(60 / DT)):
        weather.update(DT)
        speeds.append(math.hypot(*weather.wind()))
    speeds = np.array(speeds[round(10 / DT) :])  # after the blend
    assert STORM.wind * 0.3 < speeds.mean() < STORM.wind * 1.3
    assert speeds.std() > 1.0  # gusty


def test_turbulence_is_bounded_and_off_at_zero():
    weather = Weather(2)
    worst = 0.0
    for _ in range(round(60 / DT)):
        weather.update(DT)
        assert weather.turbulence_kick(0.0) == (0.0, 0.0)
        pitch, roll = weather.turbulence_kick(1.0)
        worst = max(worst, abs(pitch), abs(roll))
    s = weather.settings
    assert 0.1 < worst <= max(s.pitch_kick, s.roll_kick) + 1e-9


def test_turbulence_is_worse_low_over_ridges_and_in_cloud():
    calm_high = turbulence_intensity(CLEAR, 500.0, 0.0, False)
    assert calm_high == 0.0
    assert turbulence_intensity(STORM, 20.0, 1.0, False) > turbulence_intensity(
        STORM, 400.0, 1.0, False
    )
    assert turbulence_intensity(CLEAR, 500.0, 0.0, True) > calm_high
    assert turbulence_intensity(STORM, 0.0, 5.0, True) == 1.0


@pytest.mark.parametrize("coverage", [0.1, 0.35, 0.8])
def test_cloud_coverage_matches_the_setting(coverage):
    clouds = CloudLayer(4)
    clouds.coverage = coverage
    x, y = np.meshgrid(np.arange(0, 6000, 7.0), np.arange(0, 6000, 7.0))
    covered = (clouds.coverage_at(x, y) > 0).mean()
    assert covered == pytest.approx(coverage, abs=0.05)


def test_clouds_move_with_the_wind():
    clouds = CloudLayer(4)
    x, y = np.arange(0, 3000, 5.0), np.zeros(600)
    before = clouds.coverage_at(x, y)
    clouds.advance((10.0, 0.0), 30.0)  # 300 units east
    after = clouds.coverage_at(x + 300, y)
    assert np.array_equal(before, after)


def test_rain_falls_down_in_the_world_however_the_camera_turns():
    """Pitching up must not make the rain look like it falls upwards: the drops
    are in the world, so the streaks point the way the world's down is seen."""
    rain = Rain()
    for pitch, roll in [(0.0, 0.0), (0.6, 0.0), (-0.6, 0.0), (0.0, math.pi / 2)]:
        camera = Camera(altitude=100.0, pitch=pitch, roll=roll)
        rain.update(DT, (0.0, 0.0), camera)
        rain.update(DT, (0.0, 0.0), camera)  # standing still
        segments, drops = rain.segments(1.0, camera, 128.0, 256, 144)
        assert len(segments) > 50
        # Each streak runs from the drop back up to where it was a moment ago.
        up_x, up_y, _ = project_points(camera, drops + [0, 0, 1.0], 128.0, 256, 144)
        towards_up = (segments[:, 2] - segments[:, 0]) * (up_x - segments[:, 0]) + (
            segments[:, 3] - segments[:, 1]
        ) * (up_y - segments[:, 1])
        assert np.all(towards_up > 0)


def test_rain_streaks_come_towards_a_moving_camera():
    rain = Rain()
    camera = Camera(altitude=100.0)
    rain.update(DT, (0.0, 0.0), camera)
    camera = Camera(y=-100.0 * DT, altitude=100.0)  # flying north at 100
    rain.update(DT, (0.0, 0.0), camera)
    assert rain.camera_velocity[1] == pytest.approx(-100.0)
    segments, _ = rain.segments(1.0, camera, 128.0, 256, 144)
    # Seen from the cockpit, streaks spread out from the middle of the screen.
    head = segments[:, :2] - (128, 72)
    tail = segments[:, 2:] - (128, 72)
    outward = np.hypot(*head.T) > np.hypot(*tail.T)
    assert outward.mean() > 0.8
    assert len(Rain().segments(0.0, camera, 128.0, 256, 144)[0]) == 0


def test_thick_cloud_hides_trees_below_it():
    """From above an overcast layer, trees on the ground must not show through
    the clouds when the ground itself is hidden."""
    from flightsim.render.effects import billboard_visible
    from flightsim.render.voxel import VoxelRenderer
    from flightsim.terrain.world import TerrainLevel

    level = TerrainLevel.empty(1024, 1.0)
    level.heights[:] = 10.0
    clouds = CloudLayer(4)
    clouds.coverage = 1.0  # solid deck
    clouds.altitude = 200.0
    renderer = VoxelRenderer(256, 144)
    out = np.zeros((144, 256), dtype=np.uint8)
    tree = np.array([[0.0, -150.0, 10.0]])

    above = Camera(altitude=300.0, pitch=-0.6)
    renderer.render(out, [level], above, 0.0, clouds)
    assert not billboard_visible(renderer, above, tree, height=10.0)[0]

    below = Camera(altitude=100.0, pitch=-0.6)
    renderer.render(out, [level], below, 0.0, clouds)
    assert billboard_visible(renderer, below, tree, height=10.0)[0]
