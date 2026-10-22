"""Drawing bridges in 3D over the terrain view.

The terrain renderer is a height field and cannot show anything with air
underneath, so bridges are drawn on top of it, like the time-trial rings. A
deck is a row of short boxes; each box face that looks towards the camera is
cut against a plane just in front of the camera (so a deck right overhead
still shows), projected and filled with Pyxel triangles. Pillars are boxes
too, and the railings are lines. Cutting and projecting are shared with the
models in ``render/mesh.py``.

There is no depth buffer: the pieces of all bridges are drawn far to near,
and each is left out when the terrain (or thick cloud) drawn at its screen
position is nearer than it. Colours come from the terrain palette with the
fog level for their distance, so bridges follow the time of day and weather.
"""

import math

import numpy as np
import pyxel

from flightsim.render import palette
from flightsim.render.mesh import NEAR, View, fill_cut, line_cam
from flightsim.render.voxel import VoxelRenderer
from flightsim.sim.camera import Camera
from flightsim.terrain.biomes import Biome
from flightsim.terrain.bridges import PILLAR_RADIUS, Bridge

SEGMENT = 10.0  # world units of deck per box
MIN_PIXELS = 3.0  # bridges narrower than this on screen are not drawn

# (biome, shade) per surface; the light comes from above.
DECK_TOP = (Biome.RUNWAY, 2)
DECK_SIDE = (Biome.ROCK, 3)
DECK_BOTTOM = (Biome.ROCK, 1)
PILLAR = (Biome.ROCK, 2)
RAILING = (Biome.ROCK, 5)


# A unit box: per face, its outward normal and four corners, in (along,
# across, up) units of the box's half-size. Faces: top, bottom, then sides.
_NORMALS = np.array(
    [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]], float
)
_CORNERS = np.array(
    [
        [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        [[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]],
        [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]],
        [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],
        [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]],
    ],
    float,
)
_DECK_SURFACES = [DECK_TOP, DECK_BOTTOM] + [DECK_SIDE] * 4


def bridge_boxes(bridge: Bridge) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The boxes a bridge is drawn from: centres (n, 3), half-sizes (n, 3) along
    the deck, across it and up, the (3, 3) axes, and how many are deck pieces
    (the rest are pillars)."""
    along = np.array([bridge.ax, bridge.ay, 0.0])
    axes = np.array([along, [-bridge.ay, bridge.ax, 0.0], [0.0, 0.0, 1.0]])
    count = max(1, math.ceil(2 * bridge.half_span / SEGMENT))
    length = 2 * bridge.half_span / count
    offsets = -bridge.half_span + (np.arange(count) + 0.5) * length
    deck = np.array([bridge.x, bridge.y, bridge.deck - bridge.thickness / 2])
    centres = [deck + along * o for o in offsets]
    halves = [(length / 2, bridge.width / 2, bridge.thickness / 2)] * count
    for px, py, ground in bridge.pillars:
        height = bridge.bottom - ground
        centres.append(np.array([px, py, ground + height / 2]))
        halves.append((PILLAR_RADIUS, PILLAR_RADIUS, height / 2))
    return np.array(centres), np.array(halves), axes, count


def _color(surface: tuple[Biome, int], fog: int) -> int:
    return int(palette.biome_color(*surface)) + fog * palette.TERRAIN_COLORS


def draw_bridges(
    bridges: list[Bridge], camera: Camera, renderer: VoxelRenderer, max_distance: float
) -> int:
    """Draw bridges far to near; returns how many faces were filled."""
    view = View(camera, renderer)
    origin = view.origin
    to_camera = view.to_camera
    h, w = renderer.pixel_distance.shape
    jobs = []
    for bridge in bridges:
        distance = math.hypot(bridge.x - origin[0], bridge.y - origin[1])
        if distance - bridge.half_span > max_distance:
            continue
        if 2 * bridge.half_span * renderer.focal / max(distance, 1.0) < MIN_PIXELS:
            continue
        centres, halves, axes, decks = bridge_boxes(bridge)
        # Every corner of every face of every box at once: (boxes, 6, 4, 3).
        corners = (
            centres[:, None, None, :] + (_CORNERS * halves[:, None, None, :]) @ axes
        )
        normals = _NORMALS @ axes
        facing = np.einsum("bfk,fk->bf", origin - corners[:, :, 0], normals) > 0
        cam = (corners - origin) @ to_camera
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = view.cx + cam[..., 0] / cam[..., 2] * view.focal
            sy = view.cy - cam[..., 1] / cam[..., 2] * view.focal
        ahead = cam[..., 2] >= NEAR
        horizontal = np.hypot(centres[:, 0] - origin[0], centres[:, 1] - origin[1])
        fog = np.floor(renderer.fog_for(horizontal) + 0.5).astype(int)
        # Hidden behind terrain or cloud? Tested at each box's centre.
        mid = (centres - origin) @ to_camera
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = view.cx + mid[:, 0] / mid[:, 2] * view.focal
            my = view.cy - mid[:, 1] / mid[:, 2] * view.focal
        on_screen = (mid[:, 2] >= NEAR) & (mx >= 0) & (mx < w) & (my >= 0) & (my < h)
        col = np.clip(np.nan_to_num(mx), 0, w - 1).astype(int)
        row = np.clip(np.nan_to_num(my), 0, h - 1).astype(int)
        slack = 3.0 + 0.02 * horizontal + bridge.width
        seen = ~on_screen | (horizontal <= renderer.pixel_distance[row, col] + slack)
        seen &= fog < palette.FOG_LEVELS
        depth = np.linalg.norm(centres - origin, axis=1)
        if origin[2] > bridge.bottom:
            # Seen from above, the deck hides the pillars under it: draw them
            # first, whatever their distance.
            depth[decks:] += 1e6
        for i in np.flatnonzero(seen):
            jobs.append((depth[i], bridge, i, decks, facing, cam, sx, sy, ahead, fog))
    jobs.sort(key=lambda job: -job[0])
    filled = 0
    for _, bridge, i, decks, facing, cam, sx, sy, ahead, fog in jobs:
        deck = i < decks
        for f in np.flatnonzero(facing[i]):
            surface = _DECK_SURFACES[f] if deck else PILLAR
            color = _color(surface, fog[i])
            if ahead[i, f].all():
                x, y = sx[i, f], sy[i, f]
                pyxel.tri(x[0], y[0], x[1], y[1], x[2], y[2], color)
                pyxel.tri(x[0], y[0], x[2], y[2], x[3], y[3], color)
            elif not fill_cut(view, cam[i, f], color):
                continue
            filled += 1
        if deck:
            rise = np.array([0.0, 0.0, bridge.railing]) @ to_camera
            _railings(view, cam[i], rise, _color(RAILING, fog[i]))
    return filled


def _railings(view: View, box: np.ndarray, rise: np.ndarray, color: int) -> None:
    """Rails along both top edges of one deck box (its faces' corners in camera
    space), ``rise`` above it, with a post at one end of each."""
    top = box[0].tolist()  # the top face: corners (-a,-c), (a,-c), (a,c), (-a,c)
    up = rise.tolist()
    for a, b in ((top[0], top[1]), (top[3], top[2])):
        ra = [a[0] + up[0], a[1] + up[1], a[2] + up[2]]
        rb = [b[0] + up[0], b[1] + up[1], b[2] + up[2]]
        line_cam(view, ra, rb, color)
        line_cam(view, a, ra, color)
