import math
from dataclasses import replace

import numpy as np
import pytest

from flightsim.controls import Controls
from flightsim.render.sprites import _COLORS, helicopter_sprites, plane_sprites
from flightsim.sim.aircraft import Aircraft, FlightState
from flightsim.sim.planes import CARGO, PLANES, SEAPLANE, STUNT, TRAINER, TYPES, by_name
from flightsim.sim.wrecks import Wrecks

DT = 1 / 30


def land(_x, _y):
    return 50.0


def sea(_x, _y):
    return 0.0  # over water the terrain gives the surface: sea level


def takeoff_distance(plane_type) -> float:
    """Ground roll from standstill at full throttle, pulling up at takeoff speed."""
    plane = Aircraft(settings=plane_type.settings)
    plane.park(0.0, 0.0, 0.0, land)
    runway = lambda _x, _y: True  # the whole world is a runway
    for _ in range(round(60 / DT)):
        pull = 1.0 if plane.speed >= plane_type.settings.takeoff_speed else 0.0
        plane.update(Controls(throttle=1.0, pitch=pull), DT, land, runway=runway)
        if plane.state is FlightState.FLYING:
            return math.hypot(plane.x, plane.y)
    raise AssertionError(f"{plane_type.name} never took off")


def roll_time(plane_type) -> float:
    """Seconds to roll 90 degrees from level flight at full stick."""
    plane = Aircraft(settings=plane_type.settings)
    plane.respawn(0.0, 0.0, 0.0, land)
    t = 0.0
    while plane.roll < math.radians(90):
        plane.update(Controls(roll=1.0), DT, land)
        t += DT
    return t


def test_types_have_their_character():
    speeds = {t.name: t.top_speed for t in PLANES}
    assert speeds["STUNT"] > speeds["TRAINER"] > speeds["SEAPLANE"] > speeds["CARGO"]
    assert speeds["CARGO"] == pytest.approx(110, abs=1)
    assert STUNT.settings.stall_speed < TRAINER.settings.stall_speed
    assert CARGO.settings.stall_speed > TRAINER.settings.stall_speed
    assert roll_time(STUNT) < roll_time(TRAINER) < roll_time(CARGO)
    distances = {t.name: takeoff_distance(t) for t in PLANES}
    assert distances["STUNT"] < distances["TRAINER"] < distances["CARGO"]
    assert distances["CARGO"] < 300 * 0.9  # still fits on a runway


def test_tank_sizes_differ():
    minutes = {t.name: 1 / t.settings.full_burn / 60 for t in PLANES}
    assert minutes["STUNT"] < minutes["TRAINER"] < minutes["CARGO"]


def test_seaplane_lands_on_water_and_takes_off_again():
    plane = Aircraft(settings=SEAPLANE.settings)
    plane.respawn(0.0, 0.0, 0.0, sea)
    plane.altitude = 0.0 + plane.settings.gear_height + 1.0
    plane.speed = 56.0
    plane.pitch = math.radians(-2)
    for _ in range(round(2 / DT)):
        plane.update(Controls(), DT, sea)
    assert plane.state is FlightState.LANDED and plane.on_water
    assert plane.altitude == pytest.approx(plane.settings.gear_height)
    for _ in range(round(30 / DT)):
        pull = 1.0 if plane.speed > plane.settings.takeoff_speed else 0.0
        plane.update(Controls(throttle=1.0, pitch=pull), DT, sea)
        if plane.altitude > 20:
            break
    assert plane.state is FlightState.FLYING and plane.altitude > 20


def touch_down_on_land(settings, runway=None) -> Aircraft:
    plane = Aircraft(settings=settings)
    plane.respawn(0.0, 0.0, 0.0, land)
    plane.altitude = 50.0 + settings.gear_height + 1.0
    plane.speed = 58.0
    plane.pitch = math.radians(-2)
    for _ in range(round(2 / DT)):
        plane.update(Controls(), DT, land, runway=runway)
    return plane


def test_seaplane_lands_on_runways_but_not_on_grass():
    grass = touch_down_on_land(SEAPLANE.settings)
    assert grass.state is FlightState.CRASHED
    assert grass.crash_reason == "FLOATS HIT THE GROUND"
    runway = touch_down_on_land(SEAPLANE.settings, runway=lambda _x, _y: True)
    assert runway.state is FlightState.LANDED


def test_wheels_still_ditch_in_water():
    plane = Aircraft(settings=TRAINER.settings)
    plane.respawn(0.0, 0.0, 0.0, sea)
    plane.altitude = plane.settings.gear_height + 1.0
    plane.speed = 56.0
    plane.pitch = math.radians(-2)
    for _ in range(round(2 / DT)):
        plane.update(Controls(), DT, sea)
    assert plane.crash_reason == "DITCHED IN THE WATER"


def test_sea_level_can_be_set_per_world():
    settings = replace(SEAPLANE.settings, sea_level=12.0)
    plane = Aircraft(settings=settings)
    plane.respawn(0.0, 0.0, 0.0, sea)
    plane.altitude = 12.0 + settings.gear_height + 0.5
    plane.speed = 55.0
    plane.pitch = math.radians(-2)
    for _ in range(round(2 / DT)):
        plane.update(Controls(), DT, sea)
    assert plane.on_water and plane.altitude == pytest.approx(12.0 + 1.5)


def test_every_type_has_complete_sprites():
    for t in TYPES:
        if t.helicopter:
            views = helicopter_sprites(t.body, t.trim)
        else:
            views = plane_sprites(t.body, t.trim, t.settings.floats)
        assert set(views) == {"rear", "front", "side", "top", "wreck"}
        for rows in views.values():
            assert len({len(row) for row in rows}) == 1
            assert set("".join(rows)) <= set(_COLORS)
    floats = plane_sprites("w", "l", floats=True)["side"]
    assert floats != plane_sprites("w", "l")["side"]


def test_unknown_type_names_fall_back_to_the_trainer():
    assert by_name("CARGO") is CARGO
    assert by_name("ZEPPELIN") is TRAINER


def test_wrecks_remember_their_type():
    wrecks = Wrecks(limit=2)
    wrecks.add(np.array([0.0, 0.0, 1.0]), 1)
    wrecks.add(np.array([10.0, 0.0, 1.0]), 3)
    wrecks.add(np.array([20.0, 0.0, 1.0]), 2)  # the first drops out
    assert wrecks.to_list() == [[10.0, 0.0, 1.0, 3], [20.0, 0.0, 1.0, 2]]
    positions, kinds = wrecks.near_with_kinds(18.0, 0.0, 5.0)
    assert kinds.tolist() == [2] and positions[0][0] == 20.0


def test_descriptions_fit_beside_the_title_screen_model():
    assert all(len(t.description) <= 40 for t in TYPES)
