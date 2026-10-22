"""3D models of the aircraft types, and choosing between model and sprite.

Each type is built from boxes in the plane's own frame (x right, y forward,
z up, world units, the reference point where the flight model keeps the
plane: wheels and floats reach down ``gear_height``). Colours are terrain
ramps standing in for the sprites' body and trim colours, so the models fade
into the fog and follow the time of day like everything else.

Close up a model shows its real shape from any side; far away it is only a
few pixels, where the hand-drawn sprites read better, so ``draw_plane``
switches to the sprite below ``LOD_PIXELS``.
"""

import math
from dataclasses import dataclass

import numpy as np
import pyxel

from flightsim.render.effects import draw_aircraft
from flightsim.render.mesh import Mesh, box, draw_mesh
from flightsim.render.sprites import TRANSPARENT
from flightsim.render.voxel import VoxelRenderer
from flightsim.sim.camera import Camera
from flightsim.sim.planes import (
    CARGO,
    SCOUT,
    SEAPLANE,
    STUNT,
    TANDEM,
    TRAINER,
    TYPES,
    AircraftType,
)
from flightsim.terrain.biomes import Biome

LOD_PIXELS = 16.0  # narrower than this on screen: the sprite instead
WRECK_TAIL_BEND = 0.45  # radians the tail is twisted aside in a crash

# Sprite colour letters (see sprites._COLORS) as terrain ramps.
RAMPS = {
    "w": Biome.SNOW,
    "r": Biome.ROOF,
    "y": Biome.SAND,
    "g": Biome.ROCK,
    "b": Biome.DESERT,
    "l": Biome.WATER,
}
GLASS = Biome.DEEP_WATER
DARK = Biome.RUNWAY  # tyres, cowlings, propellers


@dataclass(frozen=True)
class Rotor:
    """A propeller or rotor: hub position, blade length, the axis it turns
    about (y: a propeller, z: a main rotor, x: a tail rotor), how many blades
    and how fast it turns compared with the others."""

    x: float
    y: float
    z: float
    radius: float
    axis: str = "y"
    blades: int = 2
    rate: float = 1.0


def _wheels(span: float, track: float, y: float) -> list[Mesh]:
    """Two wheels on struts under the fuselage, ``track`` apart, at ``y``."""
    parts = []
    for side in (-1, 1):
        parts.append(box((side * track / 2, y, -1.25), (0.14, 0.24, 0.24), DARK))
        parts.append(box((side * track / 4, y, -0.8), (track / 4, 0.06, 0.06), DARK))
    return parts


def _trainer(body: Biome, trim: Biome, floats: bool = False) -> tuple[Mesh, list]:
    parts = [
        box((0, 0.3, 0), (0.45, 2.4, 0.45), body, front=0.65),  # fuselage
        box((0, 2.85, 0), (0.3, 0.15, 0.3), DARK),  # cowling
        box((0, 1.2, 0.42), (0.4, 0.5, 0.2), GLASS, top=0.8),  # cabin
        box((0, 0.6, 0.55), (2.35, 0.55, 0.08), body),  # high wing
        box((-2.55, 0.6, 0.55), (0.2, 0.55, 0.09), trim),  # wing tips
        box((2.55, 0.6, 0.55), (0.2, 0.55, 0.09), trim),
        box((0, -2.0, 0.1), (1.1, 0.35, 0.05), body),  # tailplane
        box((0, -2.05, 0.6), (0.06, 0.42, 0.5), trim, top=1.0),  # fin
    ]
    if floats:
        for side in (-1, 1):
            parts.append(
                box((side * 0.95, 0.6, -1.3), (0.2, 1.8, 0.2), body, front=0.6)
            )
            for y in (1.4, -0.4):
                parts.append(box((side * 0.6, y, -0.85), (0.35, 0.05, 0.05), trim))
    else:
        parts += _wheels(5.5, 1.5, 1.0)
        parts.append(box((0, -2.2, -0.5), (0.05, 0.1, 0.12), DARK))  # tail wheel
    return Mesh.merge(parts), [Rotor(0.0, 3.05, 0.0, 0.9)]


def _stunt(body: Biome, trim: Biome) -> tuple[Mesh, list]:
    parts = [
        box((0, 0.2, 0), (0.42, 1.9, 0.45), body, front=0.7),
        box((0, 2.2, 0), (0.32, 0.12, 0.32), DARK),
        box((0, -0.1, 0.4), (0.3, 0.35, 0.12), GLASS),  # open cockpit
        box((0, 0.8, 0.8), (2.25, 0.45, 0.07), body),  # upper wing
        box((0, 0.6, -0.35), (2.0, 0.45, 0.07), body),  # lower wing
        box((-2.0, 0.8, 0.8), (0.25, 0.45, 0.08), trim),
        box((2.0, 0.8, 0.8), (0.25, 0.45, 0.08), trim),
        box((0, -1.6, 0.05), (0.95, 0.3, 0.05), trim),
        box((0, -1.65, 0.5), (0.06, 0.35, 0.45), trim),
    ]
    for side in (-1, 1):  # struts between the wings
        parts.append(box((side * 1.6, 0.7, 0.22), (0.05, 0.05, 0.55), DARK))
    parts += _wheels(4.5, 1.3, 0.8)
    return Mesh.merge(parts), [Rotor(0.0, 2.35, 0.0, 0.8)]


def _cargo(body: Biome, trim: Biome) -> tuple[Mesh, list]:
    parts = [
        box((0, 0.2, -0.05), (0.7, 3.4, 0.65), body, front=0.7),
        box((0, 3.35, 0.2), (0.45, 0.3, 0.25), GLASS, top=0.7),  # cockpit
        box((0, 0.6, 0.75), (3.9, 0.75, 0.1), body),  # high wing
        box((-4.05, 0.6, 0.75), (0.2, 0.75, 0.11), trim),
        box((4.05, 0.6, 0.75), (0.2, 0.75, 0.11), trim),
        box((0, -3.0, 0.3), (1.8, 0.45, 0.07), body),  # tailplane
        box((0, -3.05, 1.1), (0.08, 0.55, 0.75), trim),  # fin
    ]
    props = []
    for side in (-1, 1):  # two engines on the wing
        parts.append(box((side * 1.9, 1.2, 0.5), (0.35, 0.9, 0.32), trim, front=0.7))
        props.append(Rotor(side * 1.9, 2.15, 0.5, 1.0))
    parts += _wheels(8.5, 2.2, 0.4)
    parts += _wheels(8.5, 2.2, 1.0)
    return Mesh.merge(parts), props


def _skids(half_track: float, length: float, y: float) -> list[Mesh]:
    """Two skids on struts under a helicopter, reaching down to the gear height."""
    parts = []
    for side in (-1, 1):
        parts.append(box((side * half_track, y, -1.42), (0.07, length, 0.07), DARK))
        for strut in (y + length * 0.45, y - length * 0.45):
            parts.append(
                box((side * half_track * 0.8, strut, -0.95), (0.05, 0.05, 0.45), DARK)
            )
    return parts


def _helicopter(
    plane_type: AircraftType, body: Biome, trim: Biome
) -> tuple[Mesh, list]:
    """A cabin with glass in front, a tail boom with fin and tail rotor, skids,
    and the main rotor on a mast; the tandem has two rotors and no tail one."""
    radius = plane_type.settings.rotor_radius
    if plane_type is TANDEM:
        parts = [
            box((0, 0, 0), (0.9, 4.0, 0.9), body, front=0.85),
            box((0, 4.05, 0.25), (0.6, 0.25, 0.45), GLASS, front=0.7),
            box((0, -3.4, 1.15), (0.45, 0.8, 0.45), trim),  # rear rotor pylon
            box((0, 3.0, 1.0), (0.3, 0.3, 0.2), trim),
        ]
        for y in (2.2, -2.2):  # wheels under the long fuselage
            for side in (-1, 1):
                parts.append(box((side * 0.8, y, -1.25), (0.14, 0.24, 0.24), DARK))
        rotors = [
            Rotor(0.0, 3.0, 1.3, radius, "z", 3, 1.0),
            Rotor(0.0, -3.4, 1.75, radius, "z", 3, -1.0),  # turning the other way
        ]
        return Mesh.merge(parts), rotors
    small = plane_type is SCOUT
    k = 0.8 if small else 1.0  # the scout is a smaller helicopter all round
    parts = [
        box((0, 0.5 * k, 0), (0.75 * k, 1.5 * k, 0.75), body, front=0.75),
        box((0, 1.95 * k, 0.2), (0.6 * k, 0.4 * k, 0.45), GLASS, front=0.75),
        box((0, 0.0, 0.95), (0.45 * k, 0.9 * k, 0.25), trim),  # engine
        box((0, -2.6 * k, 0.3), (0.12, 1.8 * k, 0.12), body, front=2.0),  # boom
        box((0, -4.35 * k, 0.8), (0.06, 0.3, 0.6), trim),  # fin
        box((0, 0.3 * k, 1.3), (0.08, 0.08, 0.22), DARK),  # mast
    ]
    parts += _skids(0.75 * k, 1.5 * k, 0.4 * k)
    rotors = [
        Rotor(0.0, 0.3 * k, 1.55, radius, "z", 2 if small else 4, 1.0),
        Rotor(0.2, -4.35 * k, 0.8, 0.75 * k, "x", 2, 3.0),  # tail rotor
    ]
    return Mesh.merge(parts), rotors


def _build(plane_type: AircraftType) -> tuple[Mesh, list]:
    body, trim = RAMPS[plane_type.body], RAMPS[plane_type.trim]
    if plane_type.helicopter:
        return _helicopter(plane_type, body, trim)
    if plane_type is STUNT:
        return _stunt(body, trim)
    if plane_type is CARGO:
        return _cargo(body, trim)
    return _trainer(body, trim, floats=plane_type is SEAPLANE)


MODELS = [_build(t) for t in TYPES]  # (body, propellers) per type
assert TYPES[0] is TRAINER


def _blades(rotor: Rotor, angle: float) -> list[Mesh]:
    """The rotor's blades, each reaching out from the hub, turned ``angle``."""
    r = rotor.radius
    blade = {
        "y": box((r / 2, 0, 0), (r / 2, 0.04, 0.1), DARK),  # a propeller
        "z": box((r / 2, 0, 0), (r / 2, 0.22, 0.04), DARK),  # a main rotor
        "x": box((0, r / 2, 0), (0.04, r / 2, 0.1), DARK),  # a tail rotor
    }[rotor.axis]
    out = []
    for k in range(rotor.blades):
        turned = blade.turned(
            rotor.axis, angle * rotor.rate + k * math.tau / rotor.blades
        )
        out.append(
            Mesh(
                turned.vertices + (rotor.x, rotor.y, rotor.z),
                turned.faces,
                turned.ramps,
                turned.edges,
                turned.edge_faces,
            )
        )
    return out


def plane_mesh(kind: int, propeller: float = 0.0) -> Mesh:
    """The model of aircraft type ``kind`` with its propellers or rotors at
    angle ``propeller``."""
    body, rotors = MODELS[kind]
    blades = [blade for rotor in rotors for blade in _blades(rotor, propeller)]
    return Mesh.merge([body, *blades]) if blades else body


def wreck_mesh(kind: int) -> Mesh:
    """What is left after a crash: the right wing broken off near the root,
    the tail bent aside and the paint burnt black in patches."""
    body, _ = MODELS[kind]
    v = body.vertices.copy()
    stump = TYPES[kind].span * 0.15
    v[:, 0] = np.minimum(v[:, 0], stump)  # everything right of the stump is gone
    tail = v[:, 1] < -1.4
    c, s_ = math.cos(WRECK_TAIL_BEND), math.sin(WRECK_TAIL_BEND)
    x, y = v[tail, 0], v[tail, 1] + 1.4
    v[tail, 0], v[tail, 1] = x * c - y * s_, x * s_ + y * c - 1.4
    burnt = (np.arange(len(body.ramps)) * 7919 % 5) < 2  # two faces in five
    ramps = np.where(burnt, DARK, body.ramps)
    return Mesh(v, body.faces, ramps, body.edges, body.edge_faces)


WRECKS = [wreck_mesh(k) for k in range(len(TYPES))]


def plane_size(
    kind: int, position: np.ndarray, camera: Camera, renderer: VoxelRenderer
) -> float:
    """About how many pixels across aircraft type ``kind`` is at ``position``
    (infinite when it is level with or behind the camera: close enough)."""
    f, _, _ = camera.basis()
    depth = float((np.asarray(position) - np.array(camera.position)) @ f)
    return math.inf if depth <= 0 else TYPES[kind].span * renderer.focal / depth


def draw_plane(
    kind: int,
    sprites: dict[str, pyxel.Image],
    position: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    camera: Camera,
    renderer: VoxelRenderer,
    propeller: float = 0.0,
) -> None:
    """Aircraft type ``kind``: the 3D model when it is big enough on screen,
    otherwise its sprites."""
    if plane_size(kind, position, camera, renderer) < LOD_PIXELS:
        span = TYPES[kind].span
        draw_aircraft(
            sprites, TRANSPARENT, position, forward, up, camera, renderer, span
        )
        return
    draw_mesh(plane_mesh(kind, propeller), position, forward, up, camera, renderer)


def wreck_pose(position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where and how a wreck lies: a heading of its own, nose down, on its side."""
    x, y, ground = (float(v) for v in position[:3])
    heading = (math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % math.tau
    forward = np.array([math.sin(heading), -math.cos(heading), -0.15])
    forward /= np.linalg.norm(forward)
    level_right = np.cross(forward, [0.0, 0.0, 1.0])
    level_right /= np.linalg.norm(level_right)
    level_up = np.cross(level_right, forward)
    roll = 0.35
    up = level_up * math.cos(roll) - level_right * math.sin(roll)
    return np.array([x, y, ground + 0.6]), forward, up


def wreck_size(kind: int, distance: float, renderer: VoxelRenderer) -> float:
    """Pixels across a wreck of type ``kind`` this far away."""
    return TYPES[kind].span * renderer.focal / max(distance, 1e-6)


def draw_wreck(
    kind: int, position: np.ndarray, camera: Camera, renderer: VoxelRenderer
) -> None:
    where, forward, up = wreck_pose(position)
    draw_mesh(WRECKS[kind], where, forward, up, camera, renderer)


SHOWCASE_PITCH = -0.35  # the turning model on the title screen is seen from above
SHOWCASE_MARGIN = 2.0  # pixels kept clear inside the frame


def showcase_pose(
    kind: int, angle: float, renderer: VoxelRenderer, propeller: float = 0.0
) -> tuple[Mesh, np.ndarray, np.ndarray, np.ndarray, Camera]:
    """Aircraft type ``kind`` turned ``angle`` radians about its own middle,
    in front of a camera looking straight at it, far enough away that it fits
    ``renderer``'s picture at every angle: (mesh, position, forward, up, camera).

    The fit uses the sphere round the model, which a turning model never
    leaves, so nothing reaches past the edge however it is turned.
    """
    mesh = plane_mesh(kind, propeller)
    v = mesh.vertices
    middle = (v.min(axis=0) + v.max(axis=0)) / 2
    radius = float(np.linalg.norm(v - middle, axis=1).max())
    room = min(renderer.width, renderer.height) / 2 - SHOWCASE_MARGIN
    # A sphere of radius r at distance d looks r / sqrt(d^2 - r^2) wide.
    distance = radius * math.sqrt(1 + (renderer.focal / room) ** 2)
    camera = Camera(pitch=SHOWCASE_PITCH)
    look, _, _ = camera.basis()
    forward = np.array([math.sin(angle), -math.cos(angle), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, forward)
    offset = middle[0] * right + middle[1] * forward + middle[2] * up
    position = np.array(camera.position) + look * distance - offset
    return mesh, position, forward, up, camera


def draw_showcase(
    kind: int,
    angle: float,
    frame: tuple[int, int, int, int],
    renderer: VoxelRenderer,
    propeller: float = 0.0,
) -> None:
    """The model turning inside ``frame`` (x, y, w, h); ``renderer`` is the size
    of the frame's inside, so the camera looks straight at the model."""
    x0, y0, _, _ = frame
    mesh, position, forward, up, camera = showcase_pose(
        kind, angle, renderer, propeller
    )
    # Draw as if the frame's inside were the screen. The clip rectangle is in
    # screen coordinates and does not move with the camera.
    pyxel.camera(-(x0 + 1), -(y0 + 1))
    pyxel.clip(x0 + 1, y0 + 1, renderer.width, renderer.height)
    draw_mesh(mesh, position, forward, up, camera, renderer)
    pyxel.clip()
    pyxel.camera()
