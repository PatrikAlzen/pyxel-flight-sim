import math

import numpy as np
import pytest

from flightsim.render.buildings import (
    LIGHTHOUSE_LAMP,
    lighthouse_mesh,
    tower_mesh,
)
from flightsim.render.mesh import draw_mesh, lathe, place
from flightsim.sim.camera import Camera
from flightsim.sim.flyby import Flybys
from flightsim.terrain.airfields import (
    TOWER_CAB_BOTTOM,
    TOWER_CAB_RADIUS,
    TOWER_HEIGHT,
    TOWER_SHAFT_RADIUS,
    Airfield,
    tower_hit,
)
from flightsim.terrain.objects import HEIGHT, RADIUS, Kind

NORTH = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
SKY = 1


def facing_from(mesh, eye: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which faces draw_mesh would fill seen from ``eye``, their middles and
    whether they are upright, with the mesh standing at the origin."""
    world = place(mesh, np.zeros(3), NORTH, UP)
    corners = world[mesh.faces]
    normals = np.cross(corners[:, 2] - corners[:, 0], corners[:, 1] - corners[:, 0])
    facing = np.einsum("fk,fk->f", normals, eye - corners[:, 0]) > 0
    upright = np.abs(normals[:, 2]) < 0.9 * np.linalg.norm(normals, axis=1)
    return facing, corners.mean(axis=1), upright


@pytest.mark.parametrize(
    "mesh", [lighthouse_mesh(), tower_mesh()], ids=["light", "tower"]
)
def test_round_buildings_show_the_sides_towards_the_camera(mesh):
    for eye in (np.array([500.0, 0.0, 10.0]), np.array([0.0, -500.0, 10.0])):
        facing, middles, upright = facing_from(mesh, eye)
        towards = middles[:, :2] @ eye[:2] / np.linalg.norm(eye[:2])
        # The near side's walls show and the far side's do not.
        assert facing[upright & (towards > 1.0)].all()
        assert not facing[upright & (towards < -1.0)].any()
    facing, middles, _ = facing_from(mesh, np.array([0.0, 0.0, 500.0]))
    roof = middles[:, 2] > middles[:, 2].max() - 0.3
    assert facing[roof].all()  # seen from straight above, the top shows


def test_lathe_ledges_face_the_right_way():
    """A step out faces down, a step in faces up, the lid faces up."""
    mesh = lathe(
        [(0, 1.0), (2, 1.0), (2, 2.0), (3, 2.0), (3, 1.0), (4, 1.0)], 6, [0] * 5, top=0
    )
    below, middles, _ = facing_from(mesh, np.array([0.0, 0.0, -500.0]))
    above, _, _ = facing_from(mesh, np.array([0.0, 0.0, 500.0]))
    flat_at = middles[:, 2]
    assert below[np.isclose(flat_at, 2.0)].all()  # under the overhang
    assert above[np.isclose(flat_at, 3.0)].all()  # on top of it
    assert above[np.isclose(flat_at, 4.0)].all()  # the lid


def test_buildings_match_their_pictures_and_hit_boxes():
    light = lighthouse_mesh().vertices
    assert light[:, 2].max() == pytest.approx(HEIGHT[Kind.LIGHTHOUSE])
    assert np.hypot(light[:, 0], light[:, 1]).max() <= RADIUS[Kind.LIGHTHOUSE] + 0.2
    assert 18.0 < LIGHTHOUSE_LAMP < HEIGHT[Kind.LIGHTHOUSE] - 1.6  # in the lantern
    tower = tower_mesh().vertices
    assert tower[:, 2].max() == pytest.approx(TOWER_HEIGHT)
    radius = np.hypot(tower[:, 0], tower[:, 1])
    assert radius[tower[:, 2] < TOWER_CAB_BOTTOM].max() == pytest.approx(
        TOWER_SHAFT_RADIUS
    )
    assert radius.max() == pytest.approx(TOWER_CAB_RADIUS + 0.3)


def test_models_draw(screen):
    from flightsim.render.framebuffer import screen_array
    from flightsim.render.voxel import VoxelRenderer

    renderer = VoxelRenderer(256, 144)
    renderer.pixel_distance[:] = np.inf
    camera = Camera(x=0.0, y=60.0, altitude=10.0)  # south of it, looking north
    for mesh in (lighthouse_mesh(), tower_mesh()):
        screen_array()[:] = SKY
        filled = draw_mesh(mesh, np.zeros(3), NORTH, UP, camera, renderer)
        assert filled > 0
        drawn = screen_array() != SKY
        rows = np.flatnonzero(drawn.any(axis=1))
        assert rows.max() - rows.min() > 30  # a tall thing


def test_the_tower_stands_beside_the_runway_clear_of_the_hangar():
    field = Airfield(1, 0.0, 0.0, 0.7, 300.0, 16.0, 10.0)
    props = {kind: (x, y) for kind, x, y in field.props()}
    u, v = field.local(*props["tower"])
    assert abs(v) - TOWER_CAB_RADIUS > field.width / 2 + 10  # off the runway
    hangar_u, _ = field.local(*props["hangar"])
    assert abs(u - hangar_u) > 36 / 2 + TOWER_CAB_RADIUS + 10  # beside the hangar
    assert abs(u) < field.length / 2  # by the runway, not past its end


def test_the_tower_is_solid():
    assert tower_hit(0.0, 1.0, 5.0)  # the shaft
    assert not tower_hit(4.0, 0.0, 5.0)  # beside the shaft
    assert tower_hit(4.0, 0.0, 16.0)  # the cab sticks out
    assert not tower_hit(0.0, 0.0, TOWER_HEIGHT + 0.5)  # over the roof
    assert not tower_hit(6.0, 0.0, 16.0)


def test_buzzing_the_tower_counts_once_per_pass():
    tower = np.array([0.0, 0.0, 16.5])
    flybys = Flybys()

    def fly_past(offset: float, speed: float) -> list[int]:
        """A straight pass east, ``offset`` north of the tower, 1/30 s steps."""
        hits = []
        step = speed / 30
        for k in range(round(600 / step)):
            start = np.array([-300 + k * step, offset, 16.5])
            end = start + [step, 0.0, 0.0]
            if flybys.update("tower", start, end, tower, speed):
                hits.append(k)
        return hits

    assert len(fly_past(12.0, 100.0)) == 1  # fast and close: buzzed
    assert len(fly_past(12.0, 100.0)) == 1  # away and back again: once more
    assert fly_past(12.0, 40.0) == []  # too slow
    assert fly_past(45.0, 100.0) == []  # too far
    # Very fast: steps longer than the gap, still caught between frames.
    assert len(fly_past(8.0, 900.0)) == 1
    assert math.isfinite(flybys.settings.reset)


def test_the_hangar_is_solid_where_its_model_is():
    from flightsim.render.buildings import hangar_mesh
    from flightsim.terrain.airfields import (
        HANGAR_DEPTH,
        HANGAR_ROOF,
        HANGAR_WALL,
        HANGAR_WIDTH,
        hangar_hit,
    )

    shell, _ = hangar_mesh(HANGAR_WIDTH)
    # The model's x runs along the runway, its y (the door's way) across it.
    for x, y, z in shell.vertices:
        if z <= 0:
            continue  # below the ground
        assert hangar_hit(x * 0.97, y * 0.97, z * 0.97), (x, y, z)
        assert not hangar_hit(x * 1.03 + np.sign(x) * 0.1, y, z * 1.03)
    assert hangar_hit(0.0, 0.0, HANGAR_WALL + HANGAR_ROOF - 0.5)  # under the ridge
    assert not hangar_hit(0.0, 0.0, HANGAR_WALL + HANGAR_ROOF + 0.5)  # over it
    assert not hangar_hit(HANGAR_WIDTH / 2 - 1, 0.0, HANGAR_WALL + 3)  # past the eaves
    assert not hangar_hit(0.0, HANGAR_DEPTH / 2 + 0.5, 2.0)  # in front of the door
