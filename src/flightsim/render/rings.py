"""Drawing of time-trial rings and the arrow pointing to the next one.

Rings are circles in 3D: each is sampled as points, projected with the camera
and drawn as line segments. A segment is left out where either end is behind
the camera or hidden by terrain, so rings disappear behind hills like
everything else.
"""

import math

import numpy as np
import pyxel

from flightsim.render.effects import in_front_of_terrain, project_points
from flightsim.render.voxel import VoxelRenderer
from flightsim.sim.camera import Camera
from flightsim.sim.race import Race, Ring

NEXT_COLOR = pyxel.COLOR_YELLOW
LATER_COLOR = pyxel.COLOR_WHITE
RINGS_AHEAD = 2  # the next ring and the one after it are drawn


def draw_ring(ring: Ring, color: int, camera: Camera, renderer: VoxelRenderer) -> None:
    points = ring.points(32)
    w, h = renderer.width, renderer.height
    sx, sy, depth = project_points(camera, points, renderer.focal, w, h)
    seen = (depth > 1.0) & in_front_of_terrain(renderer, camera, points, sx, sy)
    # Closer rings get a thicker outline so they read as solid hoops.
    centre_depth = float(np.mean(depth))
    thick = centre_depth > 1.0 and ring.radius / centre_depth * renderer.focal > 18
    for i in range(len(points)):
        j = (i + 1) % len(points)
        if not (seen[i] and seen[j]):
            continue
        pyxel.line(sx[i], sy[i], sx[j], sy[j], color)
        if thick:
            pyxel.line(sx[i] + 1, sy[i], sx[j] + 1, sy[j], color)
            pyxel.line(sx[i], sy[i] + 1, sx[j], sy[j] + 1, color)


def draw_rings(race: Race, camera: Camera, renderer: VoxelRenderer) -> None:
    """The next ring in yellow and the one after in white; passed rings are gone."""
    upcoming = race.course.rings[race.next_ring : race.next_ring + RINGS_AHEAD]
    for k, ring in reversed(list(enumerate(upcoming))):  # far one first
        draw_ring(ring, NEXT_COLOR if k == 0 else LATER_COLOR, camera, renderer)


def draw_ring_arrow(ring: Ring, camera: Camera, renderer: VoxelRenderer) -> None:
    """When the next ring is off screen, an arrow at the edge points the way to it."""
    w, h = renderer.width, renderer.height
    spot = camera.project(np.asarray(ring.centre), renderer.focal, w, h)
    if spot is not None and 0 <= spot[0] < w and 0 <= spot[1] < h:
        return
    _, r, u = camera.basis()
    offset = np.asarray(ring.centre) - camera.position
    angle = math.atan2(-float(offset @ u), float(offset @ r))  # screen y grows down
    cx, cy = w / 2, h / 2
    x, y = cx + math.cos(angle) * 70, cy + math.sin(angle) * 44
    tip = (x + math.cos(angle) * 6, y + math.sin(angle) * 6)
    back = angle + math.pi
    left = (x + math.cos(back + 0.6) * 5, y + math.sin(back + 0.6) * 5)
    right = (x + math.cos(back - 0.6) * 5, y + math.sin(back - 0.6) * 5)
    pyxel.tri(*tip, *left, *right, NEXT_COLOR)
    pyxel.trib(*tip, *left, *right, pyxel.COLOR_BLACK)
