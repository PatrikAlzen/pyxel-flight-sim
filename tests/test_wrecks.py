import numpy as np

from flightsim.sim.wrecks import Wrecks


def test_wrecks_accumulate_in_order():
    w = Wrecks()
    assert w.latest is None
    w.add(np.array([1.0, 2.0, 3.0]))
    w.add(np.array([4.0, 5.0, 6.0]))
    assert len(w) == 2
    assert w.latest.tolist() == [4.0, 5.0, 6.0]


def test_oldest_wrecks_go_first():
    w = Wrecks(limit=3)
    for i in range(5):
        w.add(np.array([float(i), 0.0, 0.0]))
    assert w.positions[:, 0].tolist() == [2.0, 3.0, 4.0]


def test_near_finds_wrecks_within_radius():
    w = Wrecks()
    for x in (0.0, 100.0, 1000.0):
        w.add(np.array([x, 0.0, 10.0]))
    assert w.near(50.0, 0.0, 60.0)[:, 0].tolist() == [0.0, 100.0]
    assert len(w.near(5000.0, 0.0, 60.0)) == 0


def test_clear_forgets_everything():
    w = Wrecks()
    w.add(np.array([1.0, 2.0, 3.0]))
    w.clear()
    assert len(w) == 0
    assert w.latest is None
