import math

import pytest

from flightsim.audio import SoundState
from flightsim.controls import Controls
from flightsim.persist import Profile, SaveStore
from flightsim.sim.aircraft import Aircraft, FlightState
from flightsim.terrain.airfields import Airfield, landing_report

DT = 1 / 30


def ground(_x, _y):
    return 50.0


def fuelled_plane(throttle: float) -> Aircraft:
    plane = Aircraft(uses_fuel=True)
    plane.respawn(0.0, 0.0, 0.0, ground)
    plane.throttle = throttle
    return plane


def fly(plane: Aircraft, seconds: float, controls: Controls | None = None) -> None:
    for _ in range(round(seconds / DT)):
        plane.update(controls or Controls(), DT, ground)


def test_a_tank_lasts_8_minutes_at_full_throttle():
    plane = fuelled_plane(1.0)
    fly(plane, 60.0)
    assert 1.0 - plane.fuel == pytest.approx(1 / 8, rel=0.01)
    assert plane.endurance() == pytest.approx(7 * 60, rel=0.01)


def test_an_idling_engine_still_burns_a_little():
    plane = Aircraft(uses_fuel=True)
    plane.park(0.0, 0.0, 0.0, ground)  # standing on the ground at idle
    fly(plane, 60.0)
    assert plane.state is FlightState.LANDED
    assert 1.0 - plane.fuel == pytest.approx(0.3 / 8, rel=0.01)


def test_cruise_lasts_about_a_quarter_of_an_hour():
    plane = fuelled_plane(0.5)
    assert plane.endurance() / 60 == pytest.approx(15, abs=1.5)


def test_without_fuel_setting_the_tank_never_empties():
    plane = Aircraft()
    plane.respawn(0.0, 0.0, 0.0, ground)
    fly(plane, 30.0)
    assert plane.fuel == 1.0 and plane.engine_running


def test_empty_tank_stops_the_engine_but_not_the_controls():
    plane = fuelled_plane(1.0)
    plane.fuel = 0.0
    assert not plane.engine_running
    speed = plane.speed
    fly(plane, 3.0, Controls(throttle=1.0, roll=1.0))
    assert plane.speed < speed  # no thrust: slowing down, gliding
    assert abs(plane.roll) > 0.5  # still rolls
    assert SoundState.choose(plane, 100.0).engine is None


def test_range_estimate_matches_a_simulated_cruise():
    plane = fuelled_plane(plane_throttle := 0.5)
    plane.speed = math.sqrt(plane_throttle * 40.0 / 0.002) * 0.95  # near steady speed
    estimate = plane.range_estimate()
    start = (plane.x, plane.y)
    while plane.engine_running:
        plane.update(Controls(), DT, ground)
    flown = math.hypot(plane.x - start[0], plane.y - start[1])
    assert plane.state is FlightState.FLYING
    assert flown == pytest.approx(estimate, rel=0.1)


def test_refuel_fills_and_respawn_and_park_fill_up():
    plane = fuelled_plane(0.0)
    plane.fuel = 0.2
    plane.refuel(2.0)
    assert plane.fuel == pytest.approx(0.6)
    plane.refuel(10.0)
    assert plane.fuel == 1.0
    plane.fuel = 0.1
    plane.park(0.0, 0.0, 0.0, ground)
    assert plane.fuel == 1.0
    plane.fuel = 0.1
    plane.respawn(0.0, 0.0, 0.0, ground)
    assert plane.fuel == 1.0


def test_engine_out_landing_is_graded_deadstick():
    plane = fuelled_plane(0.0)
    plane.fuel = 0.0
    plane.altitude = 50.0 + plane.settings.gear_height + 1.0
    plane.speed = 55.0
    plane.pitch = math.radians(-2)  # gliding down onto the ground
    fly(plane, 2.0)
    assert plane.state is FlightState.LANDED
    assert plane.touchdown_engine_out
    field = Airfield(1, 0.0, 0.0, 0.0, 50.0, 300.0, 16.0)
    report = landing_report(field, 0.0, 0.0, 1.0, plane.touchdown_engine_out)
    assert "DEADSTICK BUTTER" in report


def test_profile_round_trip_and_defaults(tmp_path):
    store = SaveStore(tmp_path)
    assert store.load_profile() == Profile(fuel=False)
    store.save_profile(Profile(fuel=True))
    assert store.load_profile().fuel is True
    store.profile_path.write_text("{broken")
    assert store.load_profile() == Profile()
    assert store.profile_path.with_suffix(".json.corrupt").exists()
