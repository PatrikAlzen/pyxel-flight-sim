"""Airfield buildings as 3D models.

From a distance a hangar is a small picture like the trees, but up close a
picture that turns to face the camera looks wrong, so near ones are drawn as
models: walls under a curved roof and a dark open door facing the runway.
They reach a little into the ground so a slope under them never leaves a gap.
"""

import math

import numpy as np

from flightsim.render.mesh import Mesh, box, draw_mesh, prism
from flightsim.render.voxel import VoxelRenderer
from flightsim.sim.camera import Camera
from flightsim.terrain.biomes import Biome

HANGAR_DEPTH = 26.0
HANGAR_WALL = 10.0  # height of the side walls; the roof rises above them
HANGAR_ROOF = 8.0
HANGAR_SINK = 3.0  # below the ground at its middle, for sloping ground
ROOF_SEGMENTS = 6
DOOR_WIDTH = 0.62  # of the hangar's width
DOOR_HEIGHT = 13.0
UP = np.array([0.0, 0.0, 1.0])


def hangar_mesh(width: float) -> tuple[Mesh, Mesh]:
    """A hangar ``width`` wide standing on z = 0, its door facing forward (+y):
    the building, and the door as a single face drawn over its front."""
    half = width / 2
    # The end wall's outline: straight walls, then an arch flattened to the
    # roof's height.
    profile = [(-half, -HANGAR_SINK), (half, -HANGAR_SINK)]
    for k in range(ROOF_SEGMENTS + 1):
        angle = math.pi * k / ROOF_SEGMENTS
        profile.append(
            (half * math.cos(angle), HANGAR_WALL + HANGAR_ROOF * math.sin(angle))
        )
    # Floor and side walls in grey, the curved roof in lighter sheet metal;
    # the end walls grey.
    sides = [Biome.ROCK, Biome.ROCK] + [Biome.SNOW] * ROOF_SEGMENTS + [Biome.ROCK]
    shell = prism(profile, HANGAR_DEPTH, sides, ends=Biome.ROCK)
    # One-sided, so it only shows from the front, never through the building.
    x, y = half * DOOR_WIDTH, HANGAR_DEPTH / 2 + 0.05
    corners = [
        (-x, y, -HANGAR_SINK),
        (x, y, -HANGAR_SINK),
        (x, y, DOOR_HEIGHT),
        (-x, y, DOOR_HEIGHT),
    ]
    door = Mesh.build(corners, [(0, 3, 2, 1)], [Biome.DEEP_WATER])
    return shell, door


def door_facing(field_across: tuple[float, float], side: float) -> np.ndarray:
    """Which way a hangar on ``side`` (+1 or -1) of a runway faces: the runway."""
    return -np.sign(side) * np.array([field_across[0], field_across[1], 0.0])


def draw_hangar(
    meshes: tuple[Mesh, Mesh],
    position: np.ndarray,
    facing: np.ndarray,
    camera: Camera,
    renderer: VoxelRenderer,
) -> int:
    shell, door = meshes
    filled = draw_mesh(shell, position, facing, UP, camera, renderer)
    if filled:
        filled += draw_mesh(door, position, facing, UP, camera, renderer, outline=None)
    return filled


PAD_SINK = 2.0  # the slab reaches this far down, for uneven ground
PAD_MARK = 0.03  # paint stands this far proud of the slab


def helipad_mesh(half: float) -> tuple[Mesh, Mesh]:
    """A square pad ``half`` across each way with its top at z = 0: the slab,
    and its paint (an H and a yellow edge), drawn over the slab."""
    slab = box((0, 0, -PAD_SINK / 2), (half, half, PAD_SINK / 2), Biome.ROCK)
    h = half * 0.45
    z = PAD_MARK / 2
    paint = [
        box((-h * 0.6, 0, z), (h * 0.12, h * 0.75, z), Biome.SNOW),
        box((h * 0.6, 0, z), (h * 0.12, h * 0.75, z), Biome.SNOW),
        box((0, 0, z), (h * 0.6, h * 0.12, z), Biome.SNOW),
    ]
    edge = half * 0.94
    for along in (-1, 1):
        paint.append(
            box((0, along * edge, z), (half * 0.94, half * 0.05, z), Biome.SAND)
        )
        paint.append(
            box((along * edge, 0, z), (half * 0.05, half * 0.94, z), Biome.SAND)
        )
    return slab, Mesh.merge(paint)


def draw_helipad(
    meshes: tuple[Mesh, Mesh],
    pad,
    camera: Camera,
    renderer: VoxelRenderer,
) -> int:
    slab, paint = meshes
    position = np.array([pad.x, pad.y, pad.top])
    forward = np.array([math.sin(pad.heading), -math.cos(pad.heading), 0.0])
    filled = draw_mesh(slab, position, forward, UP, camera, renderer)
    if filled:
        filled += draw_mesh(
            paint, position, forward, UP, camera, renderer, outline=None
        )
    return filled
