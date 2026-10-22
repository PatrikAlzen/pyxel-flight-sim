"""Small 3D models drawn with Pyxel triangles over the terrain view.

A ``Mesh`` is corners and flat faces (quads, or triangles that repeat their
last corner) in its own frame: x to the right, y forward, z up. ``draw_mesh``
places it with a position and forward and up vectors, the way a plane is
oriented, and draws the faces that look towards the camera far to near,
since Pyxel has no depth buffer. Faces are lit flat from a fixed sun: the
light picks the shade within the face's terrain ramp, and the distance picks
the fog level, so models follow the time of day and the weather like the
terrain. Edges between a face that looks towards the camera and one that
looks away are silhouettes; drawn a pixel out all round before the faces,
they leave a dark rim round the model that keeps it readable against the
sky. Close enough to cut through the camera, a model has no outline.

The cutting and projection helpers are shared with the bridges.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pyxel

from flightsim.render import palette
from flightsim.render.voxel import VoxelRenderer
from flightsim.sim.camera import Camera

NEAR = 0.5  # faces are cut this far in front of the camera
SUN = np.array([0.4, 0.3, 0.86]) / np.linalg.norm([0.4, 0.3, 0.86])
OUTLINE = pyxel.COLOR_BLACK
_AROUND = ((-1, 0), (1, 0), (0, -1), (0, 1))  # the outline is drawn shifted


class View:
    """The camera's frame, for cutting and projecting world points."""

    def __init__(self, camera: Camera, renderer: VoxelRenderer) -> None:
        self.origin = np.array(camera.position, dtype=float)
        self.forward, self.right, self.up = camera.basis()
        self.to_camera = np.array([self.right, self.up, self.forward]).T
        self.focal = renderer.focal
        self.cx, self.cy = renderer.width / 2, renderer.height / 2

    def project(self, cam: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                self.cx + cam[:, 0] / cam[:, 2] * self.focal,
                self.cy - cam[:, 1] / cam[:, 2] * self.focal,
            ],
            axis=-1,
        )


def cut(polygon: np.ndarray) -> np.ndarray:
    """The part of a polygon (camera space) at least NEAR in front of the camera."""
    out = []
    count = len(polygon)
    for i in range(count):
        a, b = polygon[i], polygon[(i + 1) % count]
        a_in, b_in = a[2] >= NEAR, b[2] >= NEAR
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * t)
    return np.array(out)


def fill_cut(view: View, polygon: np.ndarray, color: int) -> bool:
    """Fill a camera-space polygon that reaches behind the camera, cut first."""
    polygon = cut(polygon)
    if len(polygon) < 3:
        return False
    screen = view.project(polygon)
    x0, y0 = screen[0]
    for (x1, y1), (x2, y2) in itertools.pairwise(screen[1:]):
        pyxel.tri(x0, y0, x1, y1, x2, y2, color)
    return True


def line_cam(view: View, a: list[float], b: list[float], color: int) -> None:
    """A line between two camera-space points, cut where it goes behind the camera."""
    if a[2] < NEAR and b[2] < NEAR:
        return
    if a[2] < NEAR or b[2] < NEAR:
        if a[2] < NEAR:
            a, b = b, a
        t = (a[2] - NEAR) / (a[2] - b[2])
        b = [a[k] + (b[k] - a[k]) * t for k in range(3)]
    f, cx, cy = view.focal, view.cx, view.cy
    pyxel.line(
        cx + a[0] / a[2] * f,
        cy - a[1] / a[2] * f,
        cx + b[0] / b[2] * f,
        cy - b[1] / b[2] * f,
        color,
    )


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (n, 3): right, forward, up
    faces: np.ndarray  # (m, 4) corner indices, counter-clockwise seen from outside
    ramps: np.ndarray  # (m,) terrain ramp (Biome) per face
    edges: np.ndarray  # (e, 2) corner pairs shared by two faces...
    edge_faces: np.ndarray  # (e, 2) ...and those two faces

    @classmethod
    def build(cls, vertices, faces, ramps) -> Mesh:
        vertices = np.asarray(vertices, float)
        faces = np.asarray(faces, np.int64)
        shared: dict[tuple[int, int], list[int]] = {}
        for f, face in enumerate(faces):
            corners = list(dict.fromkeys(face.tolist()))  # a triangle repeats one
            for a, b in zip(corners, corners[1:] + corners[:1]):
                shared.setdefault((min(a, b), max(a, b)), []).append(f)
        pairs = [(edge, fs) for edge, fs in shared.items() if len(fs) == 2]
        return cls(
            vertices,
            faces,
            np.asarray(ramps, np.int64),
            np.array([e for e, _ in pairs], np.int64).reshape(-1, 2),
            np.array([fs for _, fs in pairs], np.int64).reshape(-1, 2),
        )

    @classmethod
    def merge(cls, parts: list[Mesh]) -> Mesh:
        vertices, faces, ramps, edges, edge_faces = [], [], [], [], []
        corner = face = 0
        for part in parts:
            vertices.append(part.vertices)
            faces.append(part.faces + corner)
            ramps.append(part.ramps)
            edges.append(part.edges + corner)
            edge_faces.append(part.edge_faces + face)
            corner += len(part.vertices)
            face += len(part.faces)
        return cls(
            np.concatenate(vertices),
            np.concatenate(faces),
            np.concatenate(ramps),
            np.concatenate(edges),
            np.concatenate(edge_faces),
        )

    def turned(self, axis: str, angle: float, about=(0.0, 0.0, 0.0)) -> Mesh:
        """The mesh rotated ``angle`` radians about its x, y or z axis through
        ``about`` (right-handed: positive turns y towards z about x, and so on)."""
        c, s = math.cos(angle), math.sin(angle)
        i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
        v = self.vertices - about
        out = v.copy()
        out[:, i] = v[:, i] * c - v[:, j] * s
        out[:, j] = v[:, i] * s + v[:, j] * c
        return Mesh(out + about, self.faces, self.ramps, self.edges, self.edge_faces)


# A box's corners in units of its half-size, and its faces (outward winding).
_BOX = np.array([[x, y, z] for y in (-1, 1) for z in (-1, 1) for x in (-1, 1)], float)
_BOX_FACES = [
    (0, 1, 3, 2),  # back
    (4, 6, 7, 5),  # front
    (0, 4, 5, 1),  # bottom
    (2, 3, 7, 6),  # top
    (0, 2, 6, 4),  # left
    (1, 5, 7, 3),  # right
]


def box(centre, half, ramp, front=1.0, top=1.0) -> Mesh:
    """A box at ``centre`` with half-sizes ``half`` (right, forward, up). The
    front end is scaled by ``front`` across and up (a tapered nose or tail),
    and the top face by ``top`` across (a sloped cabin or fin)."""
    v = _BOX * np.asarray(half, float)
    v[_BOX[:, 1] > 0, 0] *= front
    v[_BOX[:, 1] > 0, 2] *= front
    v[_BOX[:, 2] > 0, 0] *= top
    return Mesh.build(v + np.asarray(centre, float), _BOX_FACES, [ramp] * 6)


def _outwards(vertices: np.ndarray, face: tuple, centre: np.ndarray) -> tuple:
    """``face`` wound so its normal points away from ``centre`` (a convex shape)."""
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    if np.cross(b - a, c - a) @ ((a + b + c) / 3 - centre) >= 0:
        return face
    if face[2] == face[3]:  # a triangle: (a, b, c, c) becomes (c, b, a, a)
        return (face[2], face[1], face[0], face[0])
    return face[::-1]


def prism(profile, depth: float, ramp, ends=None, y: float = 0.0) -> Mesh:
    """A convex ``profile`` of (x, z) points stretched ``depth`` along y (centred
    on ``y``): sides in ``ramp`` (or a ramp per side, side ``i`` running from
    point ``i`` to the next), the two end faces in ``ends`` (default the first
    side's). Faces are wound to face outwards whatever order the profile is in."""
    sides = list(ramp) if isinstance(ramp, (list, tuple)) else [ramp] * len(profile)
    ends = sides[0] if ends is None else ends
    profile = [(float(x), float(z)) for x, z in profile]
    n = len(profile)
    back = [(x, y - depth / 2, z) for x, z in profile]
    front = [(x, y + depth / 2, z) for x, z in profile]
    vertices = np.array(back + front)
    faces, ramps = [], []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
        ramps.append(sides[i])
    for start in (0, n):  # the ends, as fans of triangles
        for i in range(1, n - 1):
            faces.append((start, start + i, start + i + 1, start + i + 1))
            ramps.append(ends)
    centre = vertices.mean(axis=0)
    wound = [_outwards(vertices, face, centre) for face in faces]
    return Mesh.build(vertices, wound, ramps)


def place(
    mesh: Mesh, position: np.ndarray, forward: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """The mesh's corners in the world, for a plane at ``position`` with these
    forward and up vectors. Its x goes along the right wing, which is
    ``up x forward`` as in the flight model (east when heading north)."""
    right = np.cross(up, forward)
    return np.asarray(position, float) + mesh.vertices @ np.array([right, forward, up])


def mesh_visible(
    renderer: VoxelRenderer, view: View, position: np.ndarray, slack: float
) -> bool:
    """Whether a model at ``position`` is nearer than the terrain or cloud drawn
    where it shows (tested at its centre)."""
    cam = (position - view.origin) @ view.to_camera
    if cam[2] < NEAR:
        return True
    x, y = view.project(cam[None, :])[0]
    h, w = renderer.pixel_distance.shape
    if not (0 <= x < w and 0 <= y < h):
        return True
    horizontal = math.hypot(position[0] - view.origin[0], position[1] - view.origin[1])
    return horizontal <= renderer.pixel_distance[int(y), int(x)] + slack


def draw_mesh(
    mesh: Mesh,
    position: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    camera: Camera,
    renderer: VoxelRenderer,
    outline: int | None = OUTLINE,
) -> int:
    """Draw ``mesh`` placed at ``position`` facing ``forward`` with ``up`` up;
    returns how many faces were filled (0 if hidden or lost in the fog)."""
    view = View(camera, renderer)
    position = np.asarray(position, float)
    horizontal = math.hypot(position[0] - view.origin[0], position[1] - view.origin[1])
    fog = math.floor(float(renderer.fog_for(np.array([horizontal]))[0]) + 0.5)
    size = float(np.abs(mesh.vertices).max())
    if fog >= palette.FOG_LEVELS or not mesh_visible(
        renderer, view, position, 3.0 + 0.02 * horizontal + size
    ):
        return 0
    world = place(mesh, position, forward, up)
    cam = (world - view.origin) @ view.to_camera
    corners = world[mesh.faces]
    # The model frame (right, forward, up) is left-handed (right x forward is
    # down), so placing it mirrors the winding: cross the other way round to
    # get outward normals.
    normals = np.cross(corners[:, 2] - corners[:, 0], corners[:, 1] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    facing = (np.einsum("fk,fk->f", normals, view.origin - corners[:, 0]) > 0) & (
        lengths > 1e-12
    )
    light = np.clip((normals @ SUN) / np.maximum(lengths, 1e-12), 0.0, 1.0)
    shade = 1 + np.round(light * (palette.SHADES - 2)).astype(np.int64)
    colors = (
        palette.TERRAIN_BASE
        + mesh.ramps * palette.SHADES
        + shade
        + fog * palette.TERRAIN_COLORS
    ).tolist()
    depth = cam[mesh.faces][..., 2].mean(axis=1)
    ahead = cam[:, 2] >= NEAR
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = (view.cx + cam[:, 0] / cam[:, 2] * view.focal).tolist()
        sy = (view.cy - cam[:, 1] / cam[:, 2] * view.focal).tolist()
    if outline is not None and len(mesh.edges):
        # The silhouette, a pixel wider all round, under the faces: the faces
        # then cover the inner edges and only a thin rim round the whole model
        # is left, like the sprites' outline.
        rim = facing[mesh.edge_faces[:, 0]] != facing[mesh.edge_faces[:, 1]]
        for a, b in mesh.edges[rim].tolist():
            if ahead[a] and ahead[b]:
                for dx, dy in _AROUND:
                    pyxel.line(sx[a] + dx, sy[a] + dy, sx[b] + dx, sy[b] + dy, outline)
    order = np.flatnonzero(facing)
    order = order[np.argsort(-depth[order])]
    faces = mesh.faces.tolist()
    filled = 0
    for f in order.tolist():
        a, b, c, d = faces[f]
        if ahead[a] and ahead[b] and ahead[c] and ahead[d]:
            pyxel.tri(sx[a], sy[a], sx[b], sy[b], sx[c], sy[c], colors[f])
            if d != c:
                pyxel.tri(sx[a], sy[a], sx[c], sy[c], sx[d], sy[d], colors[f])
        elif not fill_cut(view, cam[list(dict.fromkeys((a, b, c, d)))], colors[f]):
            continue
        filled += 1
    return filled
