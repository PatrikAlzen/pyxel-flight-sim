import math

import numpy as np
import pytest

from flightsim.persist import Profile, SaveStore
from flightsim.sim.achievements import ACHIEVEMENTS, BY_KEY, Sample, Tracker
from flightsim.sim.events import Event, EventKind

DT = 1 / 30
NORTH = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def tracker() -> Tracker:
    return Tracker({}, {}, seed=7)


def level(position=(0.0, 0.0, 200.0), speed=90.0, agl=200.0, **extra) -> Sample:
    return Sample(
        np.array(position), NORTH.copy(), UP.copy(), speed, agl, True, **extra
    )


def run(t: Tracker, samples) -> list[str]:
    earned = []
    for sample in samples:
        earned += t.update(sample, DT)
    return earned


def banked(bank: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward and up for level flight north with the wings ``bank`` radians down."""
    right = np.cross(NORTH, UP)
    return NORTH.copy(), UP * math.cos(bank) + right * math.sin(bank)


def loop(
    seconds: float, lowest: float = 150.0, bank: float = 0.0, radius: float = 60.0
):
    """A loop flown in ``seconds``, its bottom ``lowest`` above the ground."""
    f0, u0 = banked(bank)
    steps = round(seconds / DT)
    for k in range(steps + 10):
        angle = math.tau * min(k / steps, 1.0) + (k - steps) * 0.01 * (k > steps)
        forward = math.cos(angle) * f0 + math.sin(angle) * u0
        up = -math.sin(angle) * f0 + math.cos(angle) * u0
        height = lowest + radius * (1 - math.cos(angle))
        position = np.array([0.0, -radius * math.sin(angle), height])
        yield Sample(position, forward, up, 90.0, height, True)


def test_a_loop_is_recognised_and_a_low_one_too():
    t = tracker()
    assert run(t, [level()]) == []
    assert run(t, loop(6.0, lowest=150)) == ["LOOP"]
    t = tracker()
    run(t, [level()])
    assert run(t, loop(6.0, lowest=40)) == ["LOOP", "LOOP_LOW"]


def test_low_flying_before_a_high_loop_does_not_make_it_low():
    t = tracker()
    run(t, held(5.0, lambda k: level((0.0, -k * 3.0, 30.0), agl=30.0)))  # low
    climb = [
        level((0.0, -500.0 - k * 3.0, 30.0 + k * 0.6), agl=30.0 + k * 0.6)
        for k in range(300)
    ]
    run(t, climb)
    assert run(t, loop(6.0, lowest=210)) == ["LOOP"]


def test_steeply_banked_or_slow_circles_are_not_loops():
    t = tracker()
    run(t, [level()])
    assert run(t, loop(6.0, bank=math.radians(45))) == []  # a steep turn
    t = tracker()
    run(t, [level()])
    assert run(t, loop(25.0)) == []  # longer than the 20 second window


def roll(seconds: float):
    f0, u0 = NORTH, UP
    right = np.cross(f0, u0)
    steps = round(seconds / DT)
    for k in range(steps + 1):
        angle = math.tau * k / steps
        up = math.cos(angle) * u0 + math.sin(angle) * right
        yield Sample(np.array([0.0, -k * 3.0, 200.0]), f0.copy(), up, 90.0, 200.0, True)


def test_barrel_roll_within_six_seconds():
    t = tracker()
    assert run(t, roll(4.0)) == ["BARREL_ROLL"]
    assert run(tracker(), roll(8.0)) == []


def held(seconds: float, sample_at) -> list[Sample]:
    return [sample_at(k) for k in range(round(seconds / DT) + 1)]


def inverted(k: int) -> Sample:
    return Sample(
        np.array([0.0, -k * 3.0, 200.0]), NORTH.copy(), -UP, 90.0, 200.0, True
    )


def test_inverted_for_ten_seconds():
    assert run(tracker(), held(10.1, inverted)) == ["INVERTED"]
    assert run(tracker(), held(9.5, inverted)) == []


def test_low_pass_must_be_fast_low_and_long_enough():
    def pass_at(speed, agl):
        return lambda k: level((0.0, -k * 4.0, agl), speed=speed, agl=agl)

    assert run(tracker(), held(3.1, pass_at(125, 8))) == ["LOW_PASS"]
    assert run(tracker(), held(2.5, pass_at(125, 8))) == []
    assert run(tracker(), held(5.0, pass_at(110, 8))) == []  # too slow
    assert run(tracker(), held(5.0, pass_at(125, 12))) == []  # too high


def test_clouds_storms_and_rivers():
    in_cloud = held(3.1, lambda k: level((0.0, -k * 3.0, 380.0), in_cloud=True))
    assert run(tracker(), in_cloud) == ["IN_THE_CLOUDS"]
    storm = held(60.1, lambda k: level((0.0, -k * 3.0, 200.0), storm=True))
    assert run(tracker(), storm) == ["STORM_CHASER"]

    def river(agl):
        # 3 units a frame: 90 per second, 5 km in about 56 seconds.
        return lambda k: level((0.0, -k * 3.0, 40.0), agl=agl, over_river=True)

    assert run(tracker(), held(56.0, river(30))) == ["RIVER_RUN"]
    assert run(tracker(), held(60.0, river(60))) == []  # too high above it


def test_a_respawn_is_not_distance_flown():
    t = tracker()
    run(t, held(1.0, lambda k: level((0.0, -k * 3.0, 200.0))))
    flown = t.progress["distance:7"]
    run(t, [level((90_000.0, 0.0, 200.0)), level((90_000.0, -3.0, 200.0))])
    assert t.progress["distance:7"] == pytest.approx(flown + 3.0)


def test_first_flight_is_a_take_off():
    t = tracker()
    on_ground = level(agl=0.0)
    on_ground.flying = False
    assert run(t, [on_ground, level()]) == ["FIRST_FLIGHT"]


def test_each_achievement_only_once():
    t = tracker()
    assert run(t, roll(4.0)) == ["BARREL_ROLL"]
    assert run(t, roll(4.0)) == []
    assert t.earned["BARREL_ROLL"]["seed"] == 7


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            Event(EventKind.LANDING, UP, "FIELD 3: BUTTER (1.0 SINK, 2 OFF CENTRE)"),
            ["BUTTER"],
        ),
        (
            Event(EventKind.LANDING, UP, "NEW FIELD 3: DEADSTICK FIRM (4.0 SINK)"),
            ["DEADSTICK"],
        ),
        (
            Event(EventKind.LANDING, UP, "SPLASHDOWN: DEADSTICK SMOOTH (2.0 SINK)"),
            ["SPLASHDOWN"],
        ),
        (Event(EventKind.CRASH, UP, "HIT THE LIGHTHOUSE"), ["HARD_LESSONS"]),
        (Event(EventKind.CRASH, UP, "HIT A TREE"), []),
        (Event(EventKind.BRIDGE, UP, "UNDER NORBERG BRIDGE!"), ["UNDER_BRIDGE"]),
        (Event(EventKind.NEAR_MISS, UP, "NEAR MISS!"), ["NEAR_MISS"]),
        (Event(EventKind.TOWER, UP, "BUZZED THE TOWER!"), ["BUZZ_TOWER"]),
    ],
)
def test_events(event, expected):
    assert tracker().on_event(event) == expected


def test_notes_from_the_game():
    t = tracker()
    assert t.landed_at(1, crosswind=13.0) == ["CROSSWIND"]
    assert t.landed_at(1, crosswind=3.0) == []  # the same field twice counts once
    for number in range(2, 6):
        new = t.landed_at(number, crosswind=0.0)
    assert new == ["VISIT_5"]
    assert t.finished_trial(55.0, record=None) == ["FINISH_TRIAL", "FAST_TRIAL"]
    assert t.finished_trial(70.0, record=80.0) == ["BEAT_RECORD"]
    for k in range(9):
        assert t.took_photo() == []
    assert t.took_photo() == ["PHOTOGRAPHER"]


def test_progress_carries_over_between_sessions_and_worlds(tmp_path):
    store = SaveStore(tmp_path)
    profile = Profile()
    t = Tracker(profile.earned, profile.progress, seed=1)
    for number in range(3):
        t.landed_at(number, 0.0)
    for number in range(4):
        t.flew_under_bridge(number)
    store.save_profile(profile)

    again = store.load_profile()
    t = Tracker(again.earned, again.progress, seed=2)  # another world
    assert t.landed_at(1, 0.0) == []  # field 1 in world 2 is a different field
    assert t.landed_at(2, 0.0) == ["VISIT_5"]
    assert t.flew_under_bridge(9) == ["BRIDGES_5"]
    assert "VISIT_5" in again.earned


def test_every_achievement_is_defined_once():
    keys = [a.key for a in ACHIEVEMENTS]
    assert len(keys) == len(set(keys)) == len(BY_KEY)
    assert all(len(a.name) <= 22 and a.description for a in ACHIEVEMENTS)
