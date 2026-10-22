import math

import numpy as np
import pytest

from flightsim.controls import Controls
from flightsim.sim.achievements import Sample, Tracker
from flightsim.sim.aircraft import Aircraft, FlightState
from flightsim.sim.helicopter import Helicopter
from flightsim.sim.planes import (
    HELICOPTERS,
    SCOUT,
    TANDEM,
    TRAINER,
    UTILITY,
    make_vehicle,
)

DT = 1 / 30


def flat(_x, _y):
    return 50.0


def fly(heli, seconds, controls=None, ground=flat, wind=(0.0, 0.0)):
    for _ in range(round(seconds / DT)):
        heli.update(controls or Controls(), DT, ground, wind=wind)
        if heli.state is FlightState.CRASHED:
            break


def hovering(plane_type=UTILITY) -> Helicopter:
    heli = make_vehicle(plane_type, 0.0)
    heli.respawn(0.0, 0.0, 0.0, flat)
    return heli


def test_types_make_the_right_flight_model():
    assert isinstance(make_vehicle(UTILITY, 0.0), Helicopter)
    assert not isinstance(make_vehicle(TRAINER, 0.0), Helicopter)
    assert all(t.helicopter for t in HELICOPTERS) and not TRAINER.helicopter


def test_it_hovers_still_when_left_alone():
    heli = hovering()
    start = heli.altitude
    fly(heli, 5.0)
    assert heli.state is FlightState.FLYING
    assert heli.altitude == pytest.approx(start, abs=0.2)
    assert heli.speed < 0.1
    assert 0.4 < heli.throttle < 0.7  # the collective it takes to hover


@pytest.mark.parametrize("plane_type", HELICOPTERS)
def test_leaning_flies_that_way_up_to_its_top_speed(plane_type):
    heli = hovering(plane_type)
    start = heli.altitude
    fly(heli, 30.0, Controls(pitch=-1.0))  # stick forward: nose down
    ahead = float(heli.velocity[:2] @ heli.forward[:2]) / np.linalg.norm(
        heli.forward[:2]
    )
    assert ahead == pytest.approx(plane_type.top_speed, rel=0.1)
    assert heli.altitude == pytest.approx(start, abs=2.0)  # holds its height


def test_sideways_and_backwards_too_and_it_levels_when_let_go():
    heli = hovering()
    fly(heli, 3.0, Controls(roll=1.0))
    right = np.cross(heli.up, heli.forward)
    assert float(heli.velocity @ right) > 10
    fly(heli, 6.0)  # let go
    assert heli.speed < 3.0
    assert np.allclose(heli.lean, 0.0, atol=0.01)
    fly(heli, 5.0, Controls(pitch=1.0))  # pull back: backwards
    assert float(heli.velocity[:2] @ heli.forward[:2]) < -10


def test_climb_rate_and_types_differ():
    rates = {}
    for plane_type in HELICOPTERS:
        heli = hovering(plane_type)
        fly(heli, 3.0, Controls(throttle=1.0))
        rates[plane_type.name] = heli.vertical_speed
        assert heli.vertical_speed == pytest.approx(
            plane_type.settings.max_climb, rel=0.1
        )
    assert rates["SCOUT"] > rates["UTILITY"] > rates["TANDEM"]
    assert (
        SCOUT.settings.yaw_rate > UTILITY.settings.yaw_rate > TANDEM.settings.yaw_rate
    )


def test_holding_down_settles_gently_and_up_lifts_off_again():
    heli = hovering()
    fly(heli, 30.0, Controls(throttle=-1.0))
    assert heli.state is FlightState.LANDED
    assert heli.touchdown_sink < heli.settings.max_touchdown_sink / 2
    assert heli.altitude == pytest.approx(50.0 + heli.settings.gear_height)
    fly(heli, 2.0, Controls(throttle=1.0))
    assert heli.state is FlightState.FLYING and heli.altitude > 60


def test_steep_ground_water_and_falling_are_crashes():
    def slope(x, _y):
        return 50.0 + x * 0.6

    heli = hovering()
    heli.respawn(0.0, 0.0, 0.0, slope)
    fly(heli, 30.0, Controls(throttle=-1.0), ground=slope)
    assert heli.crash_reason == "SKIDS SLIPPED ON THE SLOPE"

    heli = hovering()
    heli.respawn(0.0, 0.0, 0.0, lambda x, y: 0.0)  # over the sea
    fly(heli, 30.0, Controls(throttle=-1.0), ground=lambda x, y: 0.0)
    assert heli.crash_reason == "DITCHED IN THE WATER"

    heli = hovering()
    heli.uses_fuel, heli.fuel = True, 0.0  # engine out: it drops
    fly(heli, 10.0)
    assert heli.crash_reason == "HIT THE GROUND TOO HARD"


def test_rotor_strike_against_a_cliff():
    def cliff(x, _y):
        return 300.0 if x > 3.0 else 50.0

    heli = hovering()
    heli.respawn(0.0, 0.0, 0.0, cliff)
    heli.altitude = 60.0  # beside the cliff, rotor reaching over it
    fly(heli, 0.5, ground=cliff)
    assert heli.crash_reason == "ROTOR STRIKE"


def test_it_holds_its_spot_in_the_wind():
    heli = hovering()
    fly(heli, 20.0, wind=(12.0, 0.0))
    assert math.hypot(heli.x, heli.y) < 15


def test_fuel_follows_the_collective_and_respawn_fills_up():
    heli = hovering()
    heli.uses_fuel = True
    fly(heli, 60.0)  # hovering, about half collective
    used = 1.0 - heli.fuel
    assert 0.3 / 8 < used < 1 / 8
    assert heli.range_estimate() > 1000
    heli.respawn(0.0, 0.0, 0.0, flat)
    assert heli.fuel == 1.0


def test_it_behaves_like_an_aircraft_to_the_rest_of_the_game():
    heli = hovering()
    assert isinstance(heli, Aircraft)
    assert not heli.stalling and heli.lift == 1.0 and heli.alert(100.0) is None
    heli.park(10.0, 20.0, 1.0, flat)
    assert heli.state is FlightState.LANDED and heli.heading == pytest.approx(1.0)


def test_helicopter_achievements():
    t = Tracker({}, {}, seed=3)

    def still(k):
        return Sample(
            np.array([0.0, 0.0, 55.0]), np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]), 0.5, 5.0, True, helicopter=True,
        )  # fmt: skip

    earned = []
    for k in range(round(30.5 / DT)):
        earned += t.update(still(k), DT)
    assert "HOVER" in earned

    def back(k):
        return Sample(
            np.array([0.0, k * 0.4, 80.0]), np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]), 12.0, 30.0, True, helicopter=True, along=-12.0,
        )  # fmt: skip

    earned = []
    for k in range(round(5.5 / DT)):
        earned += t.update(back(k), DT)
    assert "BACKWARDS" in earned
    assert t.landed_off_field(250.0, None) == ["MOUNTAIN_TOP"]
    assert t.landed_off_field(30.0, 7) == ["HELIPAD"]
    for pad in range(8, 12):
        new = t.landed_off_field(30.0, pad)
    assert new == ["HELIPADS_5"]
    # An aeroplane on a mountain does not count, and planes do not hover.
    assert Tracker({}, {}, seed=3).landed_off_field(250.0, None, helicopter=False) == []


def test_helipads_are_placed_once_on_dry_flat_ground_clear_of_houses():
    from flightsim.terrain.generator import TerrainGenerator

    g = TerrainGenerator(1337)
    pads = g.helipads.overlapping(-4000, -4000, 4000, 4000)
    assert len(pads) >= 4
    again = TerrainGenerator(1337).helipads.overlapping(0, 0, 4000, 4000)
    assert set(again) <= set(pads)  # the same pads, asked for another way
    assert len({p.number for p in pads}) == len(pads)
    sea = g.settings.sea_level
    for pad in pads:
        corners = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0]]) * pad.half
        ground = g.elevation(pad.x + corners[:, 0], pad.y + corners[:, 1])
        assert ground.min() > sea + 1.5
        assert pad.top >= ground.max()
        assert pad.name.endswith(" HELIPAD")
        assert g.helipads.at(pad.x + 3, pad.y - 3) is pad
        assert g.helipads.at(pad.x + pad.half + 2, pad.y) is None
        for village in g.settlements.villages(
            pad.x - 300, pad.y - 300, pad.x + 300, pad.y + 300
        ):
            for hx, hy, _, _ in village.houses:
                assert math.hypot(hx - pad.x, hy - pad.y) > pad.half + 10
