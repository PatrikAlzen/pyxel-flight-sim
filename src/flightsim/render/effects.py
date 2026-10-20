"""Drawing of particles and the crash wreck over the 3D view.

Both are placed with the camera's projection and hidden behind terrain using
the renderer's per-pixel terrain distance, so smoke rising behind a ridge is
cut off by it.
"""

import numpy as np
import pyxel

from flightsim.render import palette
from flightsim.render.voxel import VoxelRenderer
from flightsim.sim.camera import Camera
from flightsim.sim.particles import Kind, Particles
from flightsim.terrain.biomes import Biome

# The wreck sprite is drawn at full size when this far from the camera.
WRECK_FULL_SIZE_DISTANCE = 40.0
WRECK_TILT = -14.0  # degrees: lying on its side


def project_points(
    camera: Camera, points: np.ndarray, focal: float, width: int, height: int
):
    """Screen x, y and depth of world points (rows of ``points``)."""
    f, r, u = camera.basis()
    offset = points - camera.position
    depth = offset @ f
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = width / 2 + (offset @ r) / depth * focal
        sy = height / 2 - (offset @ u) / depth * focal
    return sx, sy, depth


def in_front_of_terrain(
    renderer: VoxelRenderer,
    camera: Camera,
    points: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
) -> np.ndarray:
    """Whether each point is nearer than the terrain drawn at its screen position."""
    h, w = renderer.pixel_distance.shape
    col = np.clip(sx.astype(np.int64), 0, w - 1)
    row = np.clip(sy.astype(np.int64), 0, h - 1)
    horizontal = np.hypot(points[:, 0] - camera.x, points[:, 1] - camera.y)
    # A little slack: a particle sitting on the ground should not vanish into it.
    return horizontal <= renderer.pixel_distance[row, col] + 3.0


def smoke_color(fade: float) -> int:
    """Thick and dark when fresh, thin and pale as it ages."""
    if fade > 0.75:
        return int(palette.biome_color(Biome.ROCK, 0))
    if fade > 0.45:
        return int(palette.biome_color(Biome.ROCK, 2))
    return int(palette.biome_color(Biome.ROCK, 4))


def fire_color(fade: float) -> int:
    if fade > 0.65:
        return pyxel.COLOR_YELLOW
    if fade > 0.3:
        return pyxel.COLOR_ORANGE
    return pyxel.COLOR_RED


def draw_particles(
    particles: Particles, camera: Camera, renderer: VoxelRenderer
) -> None:
    if not len(particles):
        return
    w, h = renderer.width, renderer.height
    sx, sy, depth = project_points(camera, particles.position, renderer.focal, w, h)
    radius = particles.radius / np.where(depth > 0, depth, 1) * renderer.focal
    visible = (
        (depth > 1.0)
        & (sx > -radius)
        & (sx < w + radius)
        & (sy > -radius)
        & (sy < h + radius)
    )
    visible &= in_front_of_terrain(renderer, camera, particles.position, sx, sy)
    fade = particles.fade
    # Fresh smoke fades in, so the base of the plume does not hide the wreck.
    smoke_alpha = (
        0.7 * np.minimum(1.0, particles.age / 0.5) * np.minimum(1.0, fade * 1.3)
    )
    for i in np.flatnonzero(visible)[np.argsort(-depth[visible])]:  # far to near
        x, y, r = float(sx[i]), float(sy[i]), max(float(radius[i]), 0.5)
        match particles.kind[i]:
            case Kind.SMOKE:
                pyxel.dither(float(smoke_alpha[i]))
                pyxel.circ(x, y, r, smoke_color(fade[i]))
            case Kind.FIRE:
                pyxel.dither(min(1.0, fade[i] * 2.0))
                pyxel.circ(x, y, r, fire_color(fade[i]))
            case Kind.SPARK:
                pyxel.dither(1.0)
                color = pyxel.COLOR_YELLOW if fade[i] > 0.5 else pyxel.COLOR_ORANGE
                if r < 1.0:
                    pyxel.pset(x, y, color)
                else:
                    pyxel.circ(x, y, r, color)
    pyxel.dither(1.0)


def draw_wreck(
    sprite: pyxel.Image,
    colkey: int,
    at: np.ndarray,
    camera: Camera,
    renderer: VoxelRenderer,
    particles: Particles,
) -> None:
    """The wrecked plane lying at world point ``at``, scaled for its distance.

    Drawn after the particles: the flames start right at the wreck, and half
    of them would otherwise be in front of it and hide it.
    """
    draw_particles(particles, camera, renderer)
    w, h = renderer.width, renderer.height
    sx, sy, depth = project_points(camera, at[None, :], renderer.focal, w, h)
    if (
        depth[0] <= 1.0
        or not in_front_of_terrain(renderer, camera, at[None, :], sx, sy)[0]
    ):
        return
    scale = min(2.0, WRECK_FULL_SIZE_DISTANCE / float(depth[0]))
    # Pyxel scales about the middle of the copied area; lift it so it rests on
    # the ground instead of being half buried.
    centre_y = float(sy[0]) - sprite.height * scale / 2
    pyxel.blt(
        float(sx[0]) - sprite.width / 2,
        centre_y - sprite.height / 2,
        sprite,
        0,
        0,
        sprite.width,
        sprite.height,
        colkey,
        rotate=WRECK_TILT,
        scale=scale,
    )
