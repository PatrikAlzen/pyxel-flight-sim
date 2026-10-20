import math

import pytest

from flightsim.sim.aircraft import Aircraft, Controls, FlightState

DT = 1 / 30
FIELD = 50.0  # height of the flat test ground, above sea level


def flat(x: float, y: float) -> float:
    return FIELD


def flying(**kwargs) -> Aircraft:
    plane = Aircraft()
    plane.respawn(0.0, 0.0, 0.0, flat)
    for key, value in kwargs.items():
        setattr(plane, key, value)
    return plane


def run(
    plane: Aircraft, seconds: float, controls: Controls | None = None, ground=flat
) -> None:
    controls = controls or Controls()
    for _ in range(round(seconds / DT)):
        plane.update(controls, DT, ground)


def test_level_flight_settles_at_steady_speed_and_height():
    plane = flying()
    start_altitude = plane.altitude
    run(plane, 30)
    assert plane.state is FlightState.FLYING
    speed = plane.speed
    run(plane, 5)
    assert plane.speed == pytest.approx(speed, abs=0.5)
    assert plane.speed > plane.settings.stall_speed
    assert plane.altitude == pytest.approx(start_altitude, abs=5)


def test_more_throttle_means_more_speed():
    slow, fast = flying(throttle=0.4), flying(throttle=1.0)
    run(slow, 30)
    run(fast, 30)
    assert fast.speed > slow.speed + 20


def test_banking_turns_the_plane():
    plane = flying()
    run(plane, 1.0, Controls(roll=1))
    assert plane.roll > math.radians(20)
    heading = plane.heading
    run(plane, 2.0)
    assert 0 < (plane.heading - heading) % math.tau < math.pi  # turned right

    straight = flying()
    run(straight, 3.0)
    assert straight.heading == pytest.approx(0.0)


def test_roll_levels_out_when_stick_is_released():
    plane = flying()
    run(plane, 0.5, Controls(roll=-1))
    assert plane.roll < math.radians(-15)
    run(plane, 6.0)
    assert abs(plane.roll) < math.radians(2)


def test_climbing_costs_speed_and_diving_gains_it():
    climber = flying(pitch=math.radians(20))
    diver = flying(pitch=math.radians(-20), altitude=600)
    start = climber.speed
    run(climber, 2.0)
    run(diver, 2.0)
    assert climber.speed < start - 5
    assert diver.speed > start + 5
    assert climber.altitude > FIELD + climber.settings.spawn_height


def test_stall_drops_the_nose_and_sinks():
    plane = flying(throttle=0.0, altitude=1000)
    run(plane, 6.0, Controls(pitch=1, throttle=-1))
    assert plane.speed < plane.settings.stall_speed
    assert plane.stalling
    assert plane.vertical_speed < -5
    # Letting go: the nose falls through and the plane picks up speed again.
    run(plane, 6.0, Controls(throttle=1))
    assert plane.pitch < 0 or plane.speed > plane.settings.stall_speed


def test_thin_air_lowers_top_speed():
    low, high = flying(throttle=1.0), flying(throttle=1.0, altitude=2500)
    run(low, 40)
    run(high, 40)
    assert high.speed < low.speed - 15


def test_controls_are_weaker_at_low_speed():
    fast, slow = flying(speed=110), flying(speed=30, altitude=800)
    run(fast, 0.3, Controls(roll=1))
    run(slow, 0.3, Controls(roll=1))
    assert slow.roll < fast.roll / 2


def descend_onto_field(
    plane: Aircraft, controls: Controls, seconds: float = 30
) -> None:
    for _ in range(round(seconds / DT)):
        plane.update(controls, DT, flat)
        if plane.state is not FlightState.FLYING:
            return


def test_gentle_approach_lands():
    plane = flying(altitude=FIELD + 20, speed=65, throttle=0.25, pitch=math.radians(-2))
    descend_onto_field(plane, Controls())
    assert plane.state is FlightState.LANDED, plane.crash_reason
    assert plane.altitude == pytest.approx(FIELD + plane.settings.gear_height)


def test_landed_plane_brakes_to_a_stop_and_can_take_off_again():
    plane = flying(altitude=FIELD + 20, speed=65, throttle=0.25, pitch=math.radians(-2))
    descend_onto_field(plane, Controls())
    assert plane.state is FlightState.LANDED
    run(plane, 10, Controls(throttle=-1, brake=True))
    assert plane.speed == 0
    assert plane.state is FlightState.LANDED

    run(plane, 8, Controls(throttle=1))
    assert plane.speed > plane.settings.takeoff_speed
    assert plane.state is FlightState.LANDED  # needs to rotate first
    run(plane, 4, Controls(pitch=0.5))
    assert plane.state is FlightState.FLYING
    assert plane.altitude > FIELD + 10


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        # Each sinks gently enough to land, but breaks one other limit.
        ({"pitch": math.radians(-6), "speed": 50}, "NOSED"),
        ({"roll": math.radians(40), "speed": 90, "sink": 2}, "WINGTIP"),
    ],
)
def test_bad_touchdowns_crash(setup, reason):
    plane = flying(altitude=FIELD + 2, **setup)
    descend_onto_field(plane, Controls())
    assert plane.state is FlightState.CRASHED
    assert reason in plane.crash_reason


def test_hard_landing_crashes():
    plane = flying(altitude=FIELD + 40, speed=40, throttle=0.0)
    plane.sink = 25  # falling out of a stall
    descend_onto_field(plane, Controls())
    assert plane.state is FlightState.CRASHED
    assert "HARD" in plane.crash_reason


def test_flying_into_a_mountain_crashes():
    def wall(x: float, y: float) -> float:
        return FIELD if y > -200 else FIELD + 500  # cliff ahead (north is -y)

    plane = flying()
    run(plane, 10, ground=wall)
    assert plane.state is FlightState.CRASHED
    assert plane.crash_reason == "HIT THE TERRAIN"


def test_touching_water_crashes():
    def sea(x: float, y: float) -> float:
        return 0.0

    plane = flying(altitude=20, speed=65, throttle=0.25, pitch=math.radians(-2))
    for _ in range(900):
        plane.update(Controls(), DT, sea)
        if plane.state is not FlightState.FLYING:
            break
    assert plane.state is FlightState.CRASHED
    assert "WATER" in plane.crash_reason


def test_crashed_plane_stays_put_until_respawn():
    plane = flying(pitch=math.radians(-40), altitude=FIELD + 5)
    run(plane, 1)
    assert plane.state is FlightState.CRASHED
    x, y, since_crash = plane.x, plane.y, plane.state_time
    run(plane, 2, Controls(throttle=1, pitch=1))
    assert (plane.x, plane.y) == (x, y)
    assert plane.state_time == pytest.approx(since_crash + 2)

    plane.respawn(x, y, 1.0, flat)
    assert plane.state is FlightState.FLYING
    assert plane.altitude == FIELD + plane.settings.spawn_height
    assert plane.heading == 1.0


def test_taxiing_over_a_gentle_bump_is_fine_but_a_hillside_is_not():
    def ground(slope: float):
        return lambda x, y: FIELD + max(0.0, -y - 300) * slope  # rises to the north

    for slope, survives in [(0.4, True), (1.0, False)]:
        plane = flying(
            altitude=FIELD + 20, speed=65, throttle=0.25, pitch=math.radians(-2)
        )
        descend_onto_field(plane, Controls())
        assert plane.state is FlightState.LANDED
        plane.x, plane.y, plane.heading, plane.speed = 0.0, -280.0, 0.0, 20.0
        run(plane, 5, Controls(throttle=0.2), ground=ground(slope))
        assert (plane.state is not FlightState.CRASHED) == survives, (
            slope,
            plane.crash_reason,
        )
