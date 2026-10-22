import math

import numpy as np
import pytest

from flightsim.controls import Controls
from flightsim.sim.aircraft import Aircraft, FlightState
from flightsim.sim.events import EventKind, EventLog
from flightsim.sim.replay import Playback, Recorder, tower_spot

DT = 1 / 30


def ground(_x, _y):
    return 50.0


def fly(recorder: Recorder, plane: Aircraft, seconds: float, controls=None) -> None:
    for _ in range(round(seconds / DT)):
        plane.update(controls or Controls(), DT, ground)
        recorder.record(plane, DT)


def flying_plane() -> Aircraft:
    plane = Aircraft()
    plane.respawn(0.0, 0.0, 0.0, ground)
    return plane


def test_the_buffer_keeps_the_last_minute():
    recorder, plane = Recorder(), flying_plane()
    fly(recorder, plane, 75.0)
    assert len(recorder.samples) == 600
    assert recorder.duration == pytest.approx(59.9, abs=0.05)
    recording = recorder.recording()
    end = recording.at(recording.duration)
    # The newest sample is at most one interval old.
    assert np.allclose(
        end.position, (plane.x, plane.y, plane.altitude), atol=plane.speed * 0.1
    )


def test_interpolation_hits_the_samples_and_is_continuous():
    recorder, plane = Recorder(), flying_plane()
    fly(recorder, plane, 3.0, Controls(pitch=-0.5, roll=0.6))  # climbing turn
    recording = recorder.recording()
    for i in (0, 7, len(recording.times) - 1):
        pose = recording.at(recording.times[i])
        assert np.allclose(pose.position, recording.positions[i])
        assert np.allclose(pose.forward, recording.forwards[i])
    times = np.linspace(0, recording.duration, 400)
    positions = np.array([recording.at(t).position for t in times])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert steps.max() < 1.0  # no jumps between samples
    for t in times[::37]:
        pose = recording.at(t)
        assert np.linalg.norm(pose.forward) == pytest.approx(1.0)
        assert pose.forward @ pose.up == pytest.approx(0.0, abs=1e-9)


def test_a_jump_starts_a_new_recording():
    recorder, plane = Recorder(), flying_plane()
    fly(recorder, plane, 5.0)
    plane.respawn(5000.0, 0.0, 0.0, ground)  # a saved place far away
    fly(recorder, plane, 2.0)
    recording = recorder.recording()
    assert recording.duration < 2.1
    assert recording.positions[:, 0].min() > 4000


def test_crash_is_kept_and_respawn_starts_over():
    recorder, plane = Recorder(), flying_plane()
    events = EventLog()
    events.subscribe(recorder.mark)
    fly(recorder, plane, 3.0)
    plane.crash("HIT THE TERRAIN")
    events.emit(EventKind.CRASH, (plane.x, plane.y, 50.0), plane.crash_reason)
    crash_time = recorder.time
    fly(recorder, plane, 20.0)  # the wreck lies there; recording stops after a while
    recording = recorder.recording()
    assert recording.duration == pytest.approx(3.0 + 4.0, abs=0.2)
    t, event = recording.crash()
    assert event.text == "HIT THE TERRAIN"
    assert t == pytest.approx(crash_time - recorder.samples[0][0], abs=1e-6)
    assert recording.at(recording.duration).state is FlightState.CRASHED

    plane.respawn(plane.x, plane.y, plane.heading, ground)
    fly(recorder, plane, 1.0)
    assert recorder.recording().crash() is None  # a fresh start


def test_playback_reports_events_once_at_their_time():
    recorder, plane = Recorder(), flying_plane()
    events = EventLog()
    events.subscribe(recorder.mark)
    fly(recorder, plane, 2.0)
    events.emit(EventKind.RING, (plane.x, plane.y, plane.altitude), "RING 1/8")
    fly(recorder, plane, 3.0)
    playback = Playback(recorder.recording())
    passed = []
    while not playback.finished:
        for event in playback.step(DT):
            passed.append((round(playback.time, 1), event.text))
    assert passed == [(2.0, "RING 1/8")]

    playback = Playback(recorder.recording())
    playback.seek(3.0)  # past the ring: skipped
    assert playback.step(DT) == []
    playback.faster(1)
    assert playback.speed == 2.0
    playback.step(1.0)
    assert playback.time == pytest.approx(5.0, abs=0.01) or playback.finished


def test_playback_is_deterministic_and_paused_stands_still():
    recorder, plane = Recorder(), flying_plane()
    fly(recorder, plane, 4.0, Controls(roll=0.4))
    recording = recorder.recording()
    a, b = Playback(recording), Playback(recording)
    for _ in range(50):
        a.step(DT)
        b.step(DT)
    assert np.array_equal(a.pose().position, b.pose().position)
    a.paused = True
    before = a.time
    a.step(DT)
    assert a.time == before


def test_events_reach_every_listener():
    heard = []
    log = EventLog()
    log.subscribe(heard.append)
    log.subscribe(lambda e: heard.append(e.kind))
    event = log.emit(EventKind.LANDING, (1, 2, 3), "BUTTER")
    assert heard == [event, EventKind.LANDING]
    assert np.allclose(event.position, (1, 2, 3))


def test_tower_stands_beside_the_path_ahead_near_the_ground():
    recorder, plane = Recorder(), flying_plane()
    plane.altitude = 70.0  # low pass
    fly(recorder, plane, 6.0)
    recording = recorder.recording()
    spot = tower_spot(recording, 1.0, ground)
    ahead = recording.at(2.5).position
    sideways = math.hypot(spot[0] - ahead[0], spot[1] - ahead[1])
    assert sideways == pytest.approx(45.0, abs=1.0)
    assert spot[2] >= ground(0, 0) + 6.0 - 1e-9
    other = tower_spot(recording, 1.0, ground, side=-1.0)
    assert np.sign(other[0] - ahead[0]) != np.sign(spot[0] - ahead[0])


def test_tower_camera_holds_its_spot_until_the_plane_has_gone_by():
    """It used to move every frame: a spot 2.5 s ahead is already further away
    than the old limit, so the view jumped from side to side."""
    from flightsim.sim.replay import tower_due

    recorder, plane = Recorder(), flying_plane()
    plane.altitude = 70.0
    fly(recorder, plane, 30.0)  # half a minute straight and level
    recording = recorder.recording()
    playback = Playback(recording)
    tower, side, moves, frame = None, 1.0, [], 0
    while not playback.finished:
        playback.step(DT)
        pose = playback.pose()
        if tower_due(tower, pose.position, pose.forward):
            tower = tower_spot(recording, playback.time, ground, side=side)
            side = -side
            moves.append(frame)
        frame += 1
    gaps = np.diff(moves)
    assert len(moves) >= 3  # it does move on
    assert gaps.min() > 60  # but holds each spot for seconds, not frames
