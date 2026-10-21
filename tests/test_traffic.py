import math
import time

import numpy as np
import pytest

from flightsim.sim.traffic import Traffic, closest_approach, midair
from flightsim.terrain.generator import TerrainGenerator

SEED = 1337
DT = 1 / 30
AREAS = [(0.0, 0.0), (8000.0, 3000.0)]


@pytest.fixture(scope="module")
def generator():
    return TerrainGenerator(SEED)


@pytest.fixture(scope="module")
def routes(generator):
    """Every route around the two areas, laid out at once."""
    traffic = Traffic(SEED, generator, 4)
    found = {}
    for x, y in AREAS:
        for a, b in traffic.pairs_near(x, y):
            for start, end in ((a, b), (b, a)):
                found[(start.number, end.number)] = (
                    start,
                    end,
                    traffic.route(start, end, build=False),
                )
    return traffic, found


def test_routes_exist_and_do_not_depend_on_order(generator, routes):
    traffic, found = routes
    laid_out = [r for _, _, r in found.values() if r is not None]
    assert len(laid_out) >= 0.7 * len(found) and len(laid_out) > 20
    other = Traffic(SEED, generator, 4)
    for x, y in reversed(AREAS):  # the other way round
        for a, b in other.pairs_near(x, y):
            assert other.schedule(a, b) == traffic.schedule(a, b)
            mine = found[(a.number, b.number)][2]
            theirs = other.route(a, b, build=False)
            assert (mine is None) == (theirs is None)
            if mine is not None:
                assert np.array_equal(mine.points, theirs.points)


def test_routes_go_from_runway_to_runway(routes):
    _, found = routes
    for start, end, route in found.values():
        if route is None:
            continue
        first, last = route.points[0], route.points[-1]
        assert start.on_runway(first[0], first[1])
        assert end.on_runway(last[0], last[1])
        assert first[2] == pytest.approx(start.elevation + 1.5)
        assert last[2] == pytest.approx(end.elevation + 1.5)


def test_routes_clear_the_terrain_and_stay_within_the_slope(generator, routes):
    traffic, found = routes
    s = traffic.settings
    for _, _, route in found.values():
        if route is None:
            continue
        points = route.points
        steps = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        assert np.all(np.abs(np.diff(points[:, 2])) <= s.slope * steps + 1e-6)
        dist = np.concatenate([[0.0], np.cumsum(steps)])
        from_ends = np.minimum(dist - s.takeoff_roll, dist[-1] - s.landing_roll - dist)
        terrain = np.maximum(
            generator.elevation(points[:, 0], points[:, 1]),
            generator.settings.sea_level,
        )
        clearance = points[:, 2] - terrain
        assert clearance[from_ends > s.runway_reach].min() >= s.min_clearance - 1e-6
        far = from_ends > 1400  # the full cruise margin is reached by here
        if far.any():
            assert clearance[far].min() >= s.cruise_margin - 1e-6


def test_flights_move_smoothly(routes):
    _, found = routes
    for _, _, route in found.values():
        if route is None:
            continue
        times = np.arange(-1.0, route.duration + 1.0, DT)
        positions = np.array([route.position(t) for t in times])
        jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        assert jumps.max() < 95.0 * DT * 1.5  # never faster than cruise allows
        for t in np.linspace(0, route.duration, 25):
            _, forward, up, _ = route.pose(t)
            assert np.linalg.norm(forward) == pytest.approx(1.0)
            assert abs(float(forward @ up)) < 1e-6


def test_timetable_is_fixed_in_game_time(generator):
    """The same flights at the same game time, however the player got there."""
    a = Traffic(SEED, generator, 4)
    b = Traffic(SEED, generator, 4)
    for t in (0.0, 123.4, 900.0):
        first = a.flights(0.0, 0.0, t, build=False)
        second = b.flights(0.0, 0.0, t, build=False)
        assert [f.ident for f in first] == [f.ident for f in second]
        for f, g in zip(first, second):
            assert np.allclose(f.position, g.position)
            assert f.kind == g.kind
    # A flight keeps its identity while it moves.
    flights = a.flights(0.0, 0.0, 900.0, build=False)
    if flights:
        later = {f.ident: f for f in a.flights(0.0, 0.0, 901.0, build=False)}
        f = flights[0]
        if f.ident in later:
            assert np.linalg.norm(later[f.ident].position - f.position) < 120


def test_routes_are_built_a_few_per_frame(generator):
    traffic = Traffic(SEED, generator, 4)
    traffic.flights(0.0, 0.0, 100.0)  # first frame: nothing laid out yet
    assert len(traffic._routes) <= traffic.settings.builds_per_frame
    for _ in range(60):
        traffic.flights(0.0, 0.0, 100.0)
    assert not traffic._pending


def test_midair_collisions():
    assert midair(2.0, 5.5, 5.5, 0.7)
    assert not midair(5.0, 5.5, 5.5, 0.7)
    assert midair(5.0, 8.5, 8.5, 0.7)  # cargo planes are bigger


def test_head_on_planes_cannot_slip_through_each_other():
    """At 95 units/s each they close 6 units a frame, more than they need."""
    me_before, me = np.array([0.0, 0.0, 100.0]), np.array([3.2, 0.0, 100.0])
    them_before, them = np.array([6.0, 0.5, 100.0]), np.array([2.8, 0.5, 100.0])
    assert (
        np.linalg.norm(them - me) > 0.3 and np.linalg.norm(them_before - me_before) > 5
    )
    assert closest_approach(me, them, me_before, them_before) == pytest.approx(0.5)
    still = np.array([50.0, 0.0, 100.0])
    assert closest_approach(me, still, me_before, still) == pytest.approx(46.8)


def test_finding_flights_is_quick(generator):
    traffic = Traffic(SEED, generator, 4)
    traffic.flights(0.0, 0.0, 0.0, build=False)  # lay out the routes
    start = time.perf_counter()
    for k in range(60):
        traffic.flights(0.0, 0.0, 300.0 + k * DT)
    ms = (time.perf_counter() - start) / 60 * 1000
    assert ms < 2.0, ms  # ~0.5 ms measured
    assert math.isfinite(ms)
