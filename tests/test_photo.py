import math

import numpy as np
import pytest

from flightsim.persist import SaveStore
from flightsim.render.effects import aircraft_pose
from flightsim.sim.camera import Camera
from flightsim.sim.freecam import FreeCamera, FreeCameraSettings, FreeMode

DT = 1 / 30
TARGET = np.array([0.0, 0.0, 100.0])


def flat(_x, _y):
    return 10.0


def hill(x, _y):
    return 10.0 + max(0.0, 80.0 - abs(x))  # a ridge along y, 90 high at x = 0


def start_behind():
    return FreeCamera(Camera(x=0.0, y=30.0, altitude=110.0), TARGET)


def test_starting_in_the_cockpit_steps_back_behind_the_plane():
    cockpit = Camera(x=0.0, y=0.0, altitude=101.0, heading=0.5)
    free = FreeCamera(cockpit, TARGET)
    free.orbit(0, 0, 0, DT, flat)
    offset = np.array(cockpit.position) - TARGET
    assert np.linalg.norm(offset) == pytest.approx(free.settings.start_distance)
    assert offset[2] > 0  # above
    assert offset[:2] @ [math.sin(0.5), -math.cos(0.5)] < 0  # behind


def test_orbit_keeps_the_target_in_the_middle():
    free = start_behind()
    cam = free.camera
    for turn, tilt, zoom in [(1, 0, 0), (0, 1, 0), (-1, -1, 1), (0, 0, -1)]:
        for _ in range(20):
            free.orbit(turn, tilt, zoom, DT, flat)
        f, _, _ = cam.basis()
        to_target = TARGET - np.array(cam.position)
        to_target /= np.linalg.norm(to_target)
        assert float(f @ to_target) > 0.999


def test_orbit_circles_at_a_fixed_distance_and_zoom_changes_it():
    free = start_behind()
    free.orbit(0, 0, 0, DT, flat)
    d0 = np.linalg.norm(TARGET - np.array(free.camera.position))
    for _ in range(45):  # 1.5 s of circling: 1.8 rad
        free.orbit(1, 0, 0, DT, flat)
    cam = free.camera
    assert np.linalg.norm(TARGET - np.array(cam.position)) == pytest.approx(d0)
    assert cam.heading == pytest.approx((free.yaw) % math.tau)
    for _ in range(30):
        free.orbit(0, 0, 1, DT, flat)  # closer
    assert np.linalg.norm(TARGET - np.array(cam.position)) < d0 * 0.5


def test_camera_stays_above_the_ground_and_looks_back_at_the_target():
    free = FreeCamera(Camera(x=0.0, y=30.0, altitude=40.0), np.array([0, 0, 20.0]))
    for _ in range(60):
        free.orbit(0, -1, 0, DT, hill)  # tilt up: the camera sinks into the hill
    cam = free.camera
    assert cam.altitude >= hill(cam.x, cam.y) + free.settings.clearance - 1e-9
    assert cam.pitch < 0  # still looking down at the target


def test_flying_moves_along_the_view_and_stays_in_range():
    settings = FreeCameraSettings(max_range=100.0)
    free = FreeCamera(Camera(x=0.0, y=0.0, altitude=50.0), TARGET, settings)
    free.toggle_mode()
    assert free.mode is FreeMode.FLY
    cam = free.camera
    heading = cam.heading
    for _ in range(30):
        free.fly(0, 0, 1, 0, 0, DT, flat)
    moved = np.array([cam.x, cam.y])
    assert np.linalg.norm(moved) == pytest.approx(settings.fly_speed, rel=0.05)
    assert moved @ [math.sin(heading), -math.cos(heading)] > 0
    for _ in range(300):
        free.fly(0, 0, 1, 1, 0, DT, flat, fast=True)
    assert math.hypot(cam.x, cam.y) <= settings.max_range + 1e-6


def test_switching_modes_keeps_the_view():
    free = start_behind()
    for _ in range(10):
        free.orbit(1, 0.5, 0, DT, flat)
    cam = free.camera
    before = (cam.x, cam.y, cam.altitude, cam.heading, cam.pitch)
    free.toggle_mode()
    free.fly(0, 0, 0, 0, 0, DT, flat)
    free.toggle_mode()
    free.orbit(0, 0, 0, DT, flat)
    after = (cam.x, cam.y, cam.altitude, cam.heading, cam.pitch)
    assert after == pytest.approx(before, abs=1e-6)


NORTH = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("camera", "expected"),
    [
        (Camera(x=0.0, y=40.0, altitude=100.0), ("rear", False, False)),
        (
            Camera(x=0.0, y=-40.0, altitude=100.0, heading=math.pi),
            ("front", False, False),
        ),
        # From the west looking east the plane flies to the left on screen.
        (
            Camera(x=-40.0, y=0.0, altitude=100.0, heading=math.pi / 2),
            ("side", True, False),
        ),
        (
            Camera(x=40.0, y=0.0, altitude=100.0, heading=-math.pi / 2),
            ("side", False, False),
        ),
    ],
)
def test_aircraft_view_depends_on_where_the_camera_is(camera, expected):
    view, mirrored, angle = aircraft_pose(NORTH, UP, TARGET, camera)
    assert (view, mirrored) == expected[:2]
    assert abs(angle) < 1.0


def test_side_view_pitches_and_turns_upside_down():
    camera = Camera(x=40.0, y=0.0, altitude=100.0, heading=-math.pi / 2)
    climbing = np.array([0.0, -math.cos(0.5), math.sin(0.5)])
    climbing_up = np.array([0.0, math.sin(0.5), math.cos(0.5)])
    _, mirrored, angle = aircraft_pose(climbing, climbing_up, TARGET, camera)
    assert not mirrored
    assert angle == pytest.approx(-math.degrees(0.5), abs=1.0)  # nose up on screen
    # Upside down, still flying right: the mirrored sprite turned half a turn.
    _, mirrored, angle = aircraft_pose(NORTH, -UP, TARGET, camera)
    assert mirrored and abs(angle) == pytest.approx(180.0)
    # Straight up: nose up on screen, the plane's top towards the left.
    _, mirrored, angle = aircraft_pose(UP, -NORTH, TARGET, camera)
    assert not mirrored and angle == pytest.approx(-90.0)


def test_seen_obliquely_from_above_the_side_view_stays_level():
    """From in front and above, a level plane's nose projects downwards on
    screen; the sprite must not tilt as if it were diving."""
    _, _, angle = aircraft_pose(
        NORTH, UP, TARGET, orbit_camera(math.radians(120), -0.5)
    )
    assert abs(angle) < 1.0


def test_photo_names_say_where_and_never_overwrite(tmp_path):
    store = SaveStore(tmp_path)
    first = store.photo_path(42, 1203.4, -880.2, "DUSK")
    assert first.name == "seed42_x1203_y-880_dusk"
    first.with_name(first.name + ".png").write_bytes(b"png")
    second = store.photo_path(42, 1203.0, -880.0, "DUSK")
    assert second.name == "seed42_x1203_y-880_dusk_2"
    assert second.parent == tmp_path / "photos"


def orbit_camera(yaw: float, pitch: float, distance: float = 40.0) -> Camera:
    """A camera on a circle round TARGET looking at it (as the photo mode does)."""
    look = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            -math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ]
    )
    x, y, z = TARGET - look * distance
    return Camera(x=x, y=y, altitude=z, heading=yaw, pitch=pitch)


def test_a_full_circle_shows_rear_side_front_side():
    """Circling a plane flying north: behind it the rear view, then one side,
    the front when facing it, then the other side mirrored."""
    seen = []
    for degrees in range(0, 360, 10):
        view, mirrored, _ = aircraft_pose(
            NORTH, UP, TARGET, orbit_camera(math.radians(degrees), -0.3)
        )
        if not seen or seen[-1] != (view, mirrored):
            seen.append((view, mirrored))
    if seen[-1] == seen[0]:
        seen.pop()
    # Turning right round the plane: start behind, then its left side (nose to
    # the left on screen, so mirrored), front, its right side.
    assert seen == [
        ("rear", False),
        ("side", True),
        ("front", False),
        ("side", False),
    ]


def test_front_and_rear_are_used_on_the_right_sides():
    behind = orbit_camera(0.0, -0.3)  # camera south of the plane, looking north
    ahead = orbit_camera(math.pi, -0.3)
    assert aircraft_pose(NORTH, UP, TARGET, behind)[0] == "rear"
    assert aircraft_pose(NORTH, UP, TARGET, ahead)[0] == "front"


def test_top_view_from_above_and_mirrored_from_below():
    above = orbit_camera(0.0, -1.4)
    view, mirrored, angle = aircraft_pose(NORTH, UP, TARGET, above)
    assert (view, mirrored) == ("top", False)
    assert abs(angle) < 1.0  # flying away from the camera: nose up on screen
    turned = orbit_camera(math.pi / 2, -1.4)  # same spot, camera heading east
    _, _, angle = aircraft_pose(NORTH, UP, TARGET, turned)
    assert angle == pytest.approx(-90.0, abs=8.0)  # nose points left on screen
    below = orbit_camera(0.0, 1.4)
    assert aircraft_pose(NORTH, UP, TARGET, below)[:2] == ("top", True)
    # Rolled onto its left wing, its top faces east; seen from the east that
    # is the top view.
    wings_up = np.array([1.0, 0.0, 0.0])
    view, _, _ = aircraft_pose(NORTH, wings_up, TARGET, orbit_camera(-math.pi / 2, 0.0))
    assert view == "top"
