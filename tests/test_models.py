import math
import time

import numpy as np
import pytest

from flightsim.render.mesh import OUTLINE, box, draw_mesh, place
from flightsim.render.models import (
    LOD_PIXELS,
    MODELS,
    WRECKS,
    plane_mesh,
    plane_size,
    wreck_pose,
)
from flightsim.sim.camera import Camera
from flightsim.sim.planes import TYPES

NORTH = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
HERE = np.array([0.0, 0.0, 100.0])
SKY = 1  # a background colour no model uses


def test_models_are_placed_like_the_flight_model_orients_planes():
    """Forward is the nose, x the right wing (east when heading north)."""
    mesh = box((0, 0, 0), (1, 1, 1), 0)
    tip = mesh.vertices[np.argmax(mesh.vertices @ [10.0, 1.0, 0.1])]  # right, front
    world = place(mesh, HERE, NORTH, UP)
    placed = world[np.argmax(mesh.vertices @ [10.0, 1.0, 0.1])]
    assert placed[0] - HERE[0] == pytest.approx(tip[0])  # right wing: east
    assert placed[1] - HERE[1] == pytest.approx(-tip[1])  # nose: north (-y)
    assert placed[2] - HERE[2] == pytest.approx(tip[2])


@pytest.mark.parametrize("kind", range(len(TYPES)))
def test_every_model_has_its_size_and_sits_on_its_wheels(kind):
    body, props = MODELS[kind]
    v = body.vertices
    span = TYPES[kind].span
    if TYPES[kind].helicopter:  # the rotors set its size
        whole = plane_mesh(kind).vertices
        assert np.ptp(whole[:, :2], axis=0).max() == pytest.approx(span, rel=0.1)
    else:
        assert np.ptp(v[:, 0]) == pytest.approx(span, rel=0.1)
    assert v[:, 2].min() == pytest.approx(-1.5, abs=0.06)  # gear height
    assert v[:, 1].max() > 1.5 and v[:, 1].min() < -1.5  # nose ahead, tail behind
    assert props  # every type has at least one propeller
    assert len(body.edges)  # closed boxes: every edge between two faces


def draw(mesh, camera, forward=NORTH, up=UP):
    from flightsim.render.framebuffer import screen_array
    from flightsim.render.voxel import VoxelRenderer

    renderer = VoxelRenderer(256, 144)
    renderer.pixel_distance[:] = np.inf
    screen_array()[:] = SKY
    filled = draw_mesh(mesh, HERE, forward, up, camera, renderer)
    return filled, screen_array().copy()


def test_right_wing_shows_on_the_right_from_behind(screen):
    tipped = MODELS[0][0].turned("y", 0.0)
    camera = Camera(x=0.0, y=30.0, altitude=100.0)  # behind, looking north
    world = place(tipped, HERE, NORTH, UP)
    right_tip = world[np.argmax(tipped.vertices[:, 0])]
    x, _, _ = camera.project(right_tip, 128.0, 256, 144)
    assert x > 128


def test_only_faces_towards_the_camera_are_filled(screen):
    mesh = box((0, 0, 0), (2, 2, 2), 3)
    behind = Camera(x=0.0, y=30.0, altitude=100.0)
    above = Camera(x=0.0, y=0.0, altitude=130.0, pitch=-math.pi / 2 + 0.01)
    filled, _ = draw(mesh, behind)
    assert filled == 1  # only the back face, straight on
    filled, _ = draw(mesh, Camera(x=20.0, y=30.0, altitude=110.0))
    assert filled == 3  # back, side and top from up and to one side
    filled, _ = draw(mesh, above)
    assert filled == 1


def test_outline_is_only_a_rim(screen):
    camera = Camera(x=8.0, y=24.0, altitude=106.0, pitch=-0.15)
    filled, image = draw(plane_mesh(0, 0.3), camera)
    assert filled > 10
    black = np.argwhere(image == OUTLINE)
    assert len(black) > 20
    padded = np.pad(image, 1, constant_values=SKY)
    rim = sum((padded[row : row + 3, col : col + 3] == SKY).any() for row, col in black)
    # Nearly all of it borders the sky; a few pixels may show where two parts
    # of the model meet, like the lines between parts on the sprites.
    assert rim / len(black) > 0.8


def test_a_model_half_behind_the_camera_is_cut_not_lost(screen):
    """Under the cargo plane's wing, beside the fuselage, looking forward: the
    wing reaches behind the camera and must be cut, not dropped."""
    camera = Camera(x=2.5, y=0.0, altitude=100.0, pitch=0.6)
    filled, image = draw(plane_mesh(2), camera)
    assert filled > 0
    assert (image[:40] != SKY).mean() > 0.5  # the wing's underside overhead


def test_close_planes_are_models_far_ones_sprites():
    from flightsim.render.voxel import VoxelRenderer

    renderer = VoxelRenderer(256, 144)
    camera = Camera(x=0.0, y=0.0, altitude=100.0)
    assert plane_size(0, np.array([0.0, -26.0, 100.0]), camera, renderer) > LOD_PIXELS
    assert plane_size(0, np.array([0.0, -100.0, 100.0]), camera, renderer) < LOD_PIXELS
    assert plane_size(0, np.array([0.0, 5.0, 100.0]), camera, renderer) == math.inf


def test_the_switch_to_a_model_is_as_far_away_drawn_sharp():
    """Models cost more than sprites, so drawing sharp over the scaled-up
    terrain must not make them switch on further away."""
    from flightsim.render.models import as_model
    from flightsim.render.voxel import ScreenView, VoxelRenderer

    renderer = VoxelRenderer(256, 144)
    view = ScreenView(renderer, 3)
    camera = Camera(x=0.0, y=0.0, altitude=100.0)
    for distance in (20.0, 40.0, 60.0, 80.0, 120.0):
        at = np.array([0.0, -distance, 100.0])
        assert as_model(plane_size(0, at, camera, renderer), renderer) == as_model(
            plane_size(0, at, camera, view), view
        )


def test_wrecks_have_lost_the_right_wing():
    for kind, wreck in enumerate(WRECKS):
        if TYPES[kind].helicopter:
            continue  # no wings; its rotor is gone instead (not part of the body)
        span = TYPES[kind].span
        assert wreck.vertices[:, 0].max() < span * 0.3
        assert wreck.vertices[:, 0].min() < -span * 0.4
    where, forward, up = wreck_pose(np.array([10.0, 20.0, 5.0]))
    again = wreck_pose(np.array([10.0, 20.0, 5.0]))
    assert np.array_equal(forward, again[1])  # lies the same way every time
    assert where[2] > 5.0 and forward[2] < 0 and up[2] > 0.8


def test_drawing_planes_is_quick(screen):
    camera = Camera(x=0.0, y=26.0, altitude=106.0, pitch=-0.13)
    mesh = plane_mesh(2, 0.3)  # the biggest model
    draw(mesh, camera)
    start = time.perf_counter()
    for _ in range(20):
        draw(mesh, camera)
    ms = (time.perf_counter() - start) / 20 * 1000
    assert ms < 1.5, ms  # ~0.5 ms measured


def test_prism_faces_point_outwards():
    from flightsim.render.buildings import hangar_mesh

    shell, _ = hangar_mesh(36.0)
    corners = shell.vertices[shell.faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    outward = corners[:, :3].mean(axis=1) - shell.vertices.mean(axis=0)
    assert np.all(np.einsum("fk,fk->f", normals, outward) > 0)


def test_hangar_door_faces_the_runway_and_only_shows_from_the_front(screen):
    from flightsim.render.buildings import door_facing, hangar_mesh
    from flightsim.render.voxel import VoxelRenderer

    across = (1.0, 0.0)  # runway across east; a hangar east of it (+1) faces west
    assert np.allclose(door_facing(across, 1.0), [-1.0, 0.0, 0.0])
    assert np.allclose(door_facing(across, -2.0), [1.0, 0.0, 0.0])

    shell, door = hangar_mesh(36.0)
    renderer = VoxelRenderer(256, 144)
    renderer.pixel_distance[:] = np.inf
    facing = np.array([0.0, 1.0, 0.0])  # door to the south
    ground = np.array([0.0, 0.0, 0.0])
    in_front = Camera(x=0.0, y=60.0, altitude=8.0, heading=0.0)  # south, looking north
    behind = Camera(x=0.0, y=-60.0, altitude=8.0, heading=math.pi)
    assert draw_mesh(door, ground, facing, UP, in_front, renderer, outline=None) == 1
    assert draw_mesh(door, ground, facing, UP, behind, renderer, outline=None) == 0
    assert draw_mesh(shell, ground, facing, UP, behind, renderer) > 0


@pytest.mark.parametrize("kind", range(len(TYPES)))
def test_title_screen_model_fits_its_frame_at_every_angle(kind):
    """Planes used to reach past the frame's right edge at some angles."""
    from flightsim.render.models import showcase_pose
    from flightsim.render.voxel import VoxelRenderer

    renderer = VoxelRenderer(54, 48)  # the frame's inside
    for degrees in range(0, 360, 5):
        mesh, position, forward, up, camera = showcase_pose(
            kind, math.radians(degrees), renderer
        )
        world = place(mesh, position, forward, up)
        points = [camera.project(p, renderer.focal, 54, 48) for p in world]
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        # A pixel of outline goes round it, so one more pixel must be free.
        assert xs.min() >= 1 and xs.max() <= 53, (degrees, xs.min(), xs.max())
        assert ys.min() >= 1 and ys.max() <= 47, (degrees, ys.min(), ys.max())
        assert xs.max() - xs.min() > 15  # and it is not tiny (end-on, narrow)


def test_title_screen_model_draws_only_inside_its_frame(screen):
    from flightsim.render.framebuffer import screen_array
    from flightsim.render.models import draw_showcase
    from flightsim.render.voxel import VoxelRenderer

    frame = (196, 54, 56, 50)
    renderer = VoxelRenderer(54, 48)
    for degrees in range(0, 360, 30):
        screen_array()[:] = SKY
        draw_showcase(2, math.radians(degrees), (frame[0] + 1, frame[1] + 1), renderer)
        image = screen_array()
        inside = image[55:103, 197:251]
        assert (inside != SKY).any()
        image[55:103, 197:251] = SKY
        assert (image == SKY).all()  # nothing drawn outside the frame


@pytest.mark.parametrize("kind", range(len(TYPES)))
def test_title_screen_model_holds_still_while_its_rotors_turn(kind):
    """The frame was fitted to the blades where they happened to be, so the
    model jumped about as they turned (worst with a helicopter's long rotor)."""
    from flightsim.render.models import showcase_pose
    from flightsim.render.voxel import VoxelRenderer

    renderer = VoxelRenderer(54, 48)
    places = []
    for blades in np.linspace(0, math.tau, 13):
        mesh, position, forward, up, camera = showcase_pose(
            kind, 0.7, renderer, propeller=blades
        )
        places.append(position)
        world = place(mesh, position, forward, up)
        xs, ys = zip(*(camera.project(p, renderer.focal, 54, 48)[:2] for p in world))
        assert min(xs) >= 1 and max(xs) <= 53 and min(ys) >= 1 and max(ys) <= 47
    assert np.allclose(places, places[0])


def test_shapes_reaching_far_off_screen_are_cut_but_look_the_same(screen):
    """Pyxel takes longer the further a shape reaches; cut to a band round the
    screen, a huge triangle or line draws the same pixels, quickly."""
    import time

    from flightsim.render.framebuffer import screen_array
    from flightsim.render.mesh import View, draw_line, fill_polygon
    from flightsim.render.voxel import VoxelRenderer

    view = View(Camera(), VoxelRenderer(256, 144))
    xs, ys = [-2e6, 3e6, 100.0], [-1e6, 50.0, 4e6]
    screen_array()[:] = SKY
    screen.tri(*[v for p in zip(xs, ys) for v in p], screen.COLOR_RED)
    screen.line(-3e6, -1e6, 2e6, 9e5, screen.COLOR_WHITE)
    expected = screen_array().copy()

    screen_array()[:] = SKY
    start = time.perf_counter()
    fill_polygon(view, xs, ys, screen.COLOR_RED)
    draw_line(view, -3e6, -1e6, 2e6, 9e5, screen.COLOR_WHITE)
    took = time.perf_counter() - start
    image = screen_array()
    # The same up to a pixel along the edges (where rounding differs).
    assert (image != expected).mean() < 0.01
    assert (image == screen.COLOR_RED).sum() > 0.9 * (
        expected == screen.COLOR_RED
    ).sum()
    assert took < 0.005

    screen_array()[:] = SKY
    draw_line(view, -5e6, -5e6, -4e6, -4e6, screen.COLOR_WHITE)  # all off screen
    fill_polygon(view, [-5e6, -4e6, -4e6], [0.0, 0.0, 10.0], screen.COLOR_RED)
    assert (screen_array() == SKY).all()
