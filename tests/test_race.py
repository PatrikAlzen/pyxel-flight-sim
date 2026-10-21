import itertools
import math

import numpy as np
import pytest

from flightsim.sim.race import (
    Course,
    Phase,
    Race,
    Ring,
    build_course,
    format_time,
    ghost_position,
    passes,
)
from flightsim.terrain.generator import TerrainGenerator

RING = Ring(centre=(0.0, 0.0, 100.0), normal=(0.0, -1.0, 0.0), radius=10.0)  # fly north


def a(*v):
    return np.array(v, dtype=float)


def test_flying_through_counts():
    assert passes(RING, a(0, 5, 100), a(0, -5, 100))
    assert passes(RING, a(6, 5, 104), a(6, -5, 104))  # off centre but inside


def test_beside_above_and_short_do_not_count():
    assert not passes(RING, a(15, 5, 100), a(15, -5, 100))  # beside
    assert not passes(RING, a(0, 5, 112), a(0, -5, 112))  # too high
    assert not passes(RING, a(0, 5, 100), a(0, 1, 100))  # not reached yet


def test_backwards_does_not_count():
    assert not passes(RING, a(0, -5, 100), a(0, 5, 100))


def test_crossing_exactly_between_frames_counts_once():
    path = [a(0, y, 100) for y in (12, 4, -4, -12)]
    hits = [passes(RING, p, q) for p, q in itertools.pairwise(path)]
    assert hits == [False, True, False]


def test_ring_points_lie_on_the_ring():
    points = RING.points(24)
    assert points.shape == (24, 3)
    assert np.allclose(np.linalg.norm(points - RING.centre, axis=1), RING.radius)
    assert np.allclose((points - RING.centre) @ RING.normal, 0)


@pytest.fixture(scope="module")
def generator():
    return TerrainGenerator(seed=1337)


@pytest.fixture(scope="module")
def field(generator):
    return generator.airfields.nearest(0, 0, rings=4)


@pytest.fixture(scope="module")
def course(generator, field):
    return build_course(generator.elevation, field, 1337)


def test_course_is_deterministic(generator, field, course):
    assert build_course(generator.elevation, field, 1337) == course
    assert build_course(generator.elevation, field, 1338) != course


def test_course_shape(field, course):
    assert 8 <= len(course.rings) <= 15
    assert course.field_number == field.number
    first = np.asarray(course.rings[0].centre)
    off_the_end = np.asarray([*field.point(field.length / 2 + 350), first[2]])
    assert np.allclose(first, off_the_end)
    for r0, r1 in itertools.pairwise(course.rings):
        step = math.hypot(r1.centre[0] - r0.centre[0], r1.centre[1] - r0.centre[1])
        assert 290 <= step <= 610
        # Each ring faces the way from the previous one.
        direction = np.subtract(r1.centre, r0.centre)
        assert np.dot(
            direction / np.linalg.norm(direction), r1.normal
        ) == pytest.approx(1)


def test_no_ring_is_in_the_terrain(generator, course):
    for ring in course.rings:
        points = ring.points(24)
        ground = generator.elevation(points[:, 0], points[:, 1])
        assert np.all(points[:, 2] > ground + 3)


def fly_course(course, speed=40.0, dt=1 / 30, record=None):
    """Fly straight from ring to ring and return the finished race."""
    race = Race(course, record=record)
    position = np.array(course.rings[0].centre) - np.array(course.rings[0].normal) * 50
    for ring in course.rings:
        target = np.array(ring.centre) + np.array(ring.normal) * 5
        while np.linalg.norm(target - position) > speed * dt:
            step = (target - position) / np.linalg.norm(target - position) * speed * dt
            race.update(dt, position, position + step, moving=True)
            position = position + step
    return race


def test_race_runs_to_the_finish(course):
    race = fly_course(course)
    assert race.phase is Phase.FINISHED
    assert len(race.splits) == len(course.rings)
    assert race.splits == sorted(race.splits)
    assert race.time == race.splits[-1]
    assert race.is_record  # no record yet
    assert len(race.ghost) >= race.time / race.ghost_interval - 1


def test_clock_waits_until_moving(course):
    race = Race(course)
    race.update(1.0, a(0, 0, 0), a(0, 0, 0), moving=False)
    assert race.phase is Phase.READY and race.elapsed == 0
    race.update(1.0, a(0, 0, 0), a(0, 0, 0), moving=True)
    assert race.phase is Phase.RUNNING and race.elapsed == 1.0


def test_deltas_against_the_record(course):
    fast = fly_course(course, speed=60)
    slow = fly_course(course, speed=40, record=fast.splits)
    assert slow.delta(0) > 0  # behind the record
    assert not slow.is_record
    faster = fly_course(course, speed=80, record=fast.splits)
    assert faster.delta(len(course.rings) - 1) < 0
    assert faster.is_record


def test_rings_must_be_taken_in_order():
    rings = (RING, Ring((0.0, -100.0, 100.0), (0.0, -1.0, 0.0), 10.0))
    race = Race(Course(1, rings))
    assert (
        race.update(0.1, a(0, -95, 100), a(0, -105, 100), moving=True) is None
    )  # second first
    assert race.update(0.1, a(0, 5, 100), a(0, -5, 100), moving=True) == 0


def test_ghost_replay_interpolates():
    ghost = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    assert np.allclose(ghost_position(ghost, 0.5, 1.0), [1.5, 0, 0])
    assert ghost_position(ghost, 5.0, 1.0) is None


def test_format_time():
    assert format_time(83.42) == "1:23.4"
    assert format_time(9.0) == "0:09.0"
