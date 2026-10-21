"""Drawing of particles and the crash wreck over the 3D view.

Both are placed with the camera's projection and hidden behind terrain using
the renderer's per-pixel terrain distance, so smoke rising behind a ridge is
cut off by it.
"""

from collections.abc import Sequence

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
# Below this scale (about three pixels wide) a sprite is not worth drawing.
MIN_SCALE = 0.12


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
    """Whether each point is nearer than the terrain drawn at its screen position.

    With some slack, growing with distance: a point on the ground should not
    vanish into it, and the terrain kept for rendering gets coarser further out.
    """
    h, w = renderer.pixel_distance.shape
    col = np.clip(sx.astype(np.int64), 0, w - 1)
    row = np.clip(sy.astype(np.int64), 0, h - 1)
    horizontal = np.hypot(points[:, 0] - camera.x, points[:, 1] - camera.y)
    slack = 3.0 + 0.02 * horizontal
    return horizontal <= renderer.pixel_distance[row, col] + slack


def billboard_visible(
    renderer: VoxelRenderer,
    camera: Camera,
    positions: np.ndarray,
    height: float | np.ndarray,
) -> np.ndarray:
    """Whether sprites ``height`` world units tall standing at ``positions`` are
    in front of the terrain.

    Tested at the sprite's middle, not its foot: seen at a shallow angle, a
    tiny height difference between the foot and the drawn ground moves the
    terrain hit at that pixel many units nearer, and the sprite would blink
    in and out as the camera moves.
    """
    middle = np.array(positions, dtype=float)
    middle[:, 2] += np.asarray(height) / 2
    w, h = renderer.width, renderer.height
    sx, sy, depth = project_points(camera, middle, renderer.focal, w, h)
    return (depth > 1.0) & in_front_of_terrain(renderer, camera, middle, sx, sy)


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


def full_size_distance(sprite: pyxel.Image, world_width: float, focal: float) -> float:
    """Distance at which ``sprite`` drawn 1:1 is ``world_width`` units wide."""
    return world_width * focal / sprite.width


def draw_billboards(
    sprite: pyxel.Image | Sequence[pyxel.Image],
    colkey: int,
    positions: np.ndarray,
    camera: Camera,
    renderer: VoxelRenderer,
    full_size: float | Sequence[float],
    tilt: float = 0.0,
    variants: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    limit: int | None = None,
) -> int:
    """Sprites standing on world points (rows of ``positions``), facing the camera.

    ``sprite`` is one image or a list chosen per point by ``variants``. Each
    image is drawn 1:1 at distance ``full_size`` (one value, or one per image),
    times the point's entry in ``scales``. Sprites are scaled for their own
    distance, hidden behind terrain and drawn far to near, all kinds mixed, so
    nearer ones cover further ones. Sprites too far away to be more than a
    couple of pixels are skipped, and with ``limit`` only that many of the
    nearest are drawn. Returns how many were drawn.

    Draw particles first: flames start right at a burning wreck, and half of
    them would otherwise be in front of it and hide it.
    """
    if not len(positions):
        return 0
    sprites = [sprite] if isinstance(sprite, pyxel.Image) else list(sprite)
    fulls = np.broadcast_to(np.asarray(full_size, dtype=float), (len(sprites),))
    which = (
        np.zeros(len(positions), np.int64)
        if variants is None
        else variants.astype(np.int64)
    )
    full = fulls[which] * (1.0 if scales is None else scales)
    widths = np.array([img.width for img in sprites])[which]
    heights = np.array([img.height for img in sprites])[which]

    w, h = renderer.width, renderer.height
    sx, sy, depth = project_points(camera, positions, renderer.focal, w, h)
    scale = full / np.where(depth > 0, depth, np.inf)
    size = widths * scale
    visible = (depth > 1.0) & (scale >= MIN_SCALE)
    visible &= (sx > -size) & (sx < w + size) & (sy > -size) & (sy < h + size)
    world_height = heights * full / renderer.focal
    visible &= billboard_visible(renderer, camera, positions, world_height)
    centre_x, centre_y, angle, upright = standing_pose(
        camera, positions, world_height, renderer
    )
    visible &= upright
    shown = np.flatnonzero(visible)
    if limit is not None and len(shown) > limit:
        shown = shown[np.argpartition(depth[shown], limit)[:limit]]
    for i in shown[np.argsort(-depth[shown])]:
        img = sprites[which[i]]
        # Pyxel scales and rotates about the middle of the copied area, so
        # place that middle halfway between the foot and the top.
        pyxel.blt(
            float(centre_x[i]) - img.width / 2,
            float(centre_y[i]) - img.height / 2,
            img,
            0,
            0,
            img.width,
            img.height,
            colkey,
            rotate=float(angle[i]) + tilt,
            scale=min(4.0, float(scale[i])),
        )
    return len(shown)


def standing_pose(
    camera: Camera,
    positions: np.ndarray,
    height: np.ndarray | float,
    renderer: VoxelRenderer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where and how to draw sprites standing upright on the ground.

    A sprite has to stand along the world's vertical, not the screen's, or it
    leans with the camera when it rolls. The foot and the top of each object
    are projected; the sprite is centred halfway between them and rotated so
    its up points from the foot to the top.

    Returns the screen centre (x, y), the clockwise rotation in degrees (Pyxel's
    convention) and whether both ends are in front of the camera.
    """
    w, h = renderer.width, renderer.height
    tops = np.array(positions, dtype=float)
    tops[:, 2] += np.asarray(height)
    fx, fy, foot_depth = project_points(camera, positions, renderer.focal, w, h)
    tx, ty, top_depth = project_points(camera, tops, renderer.focal, w, h)
    upright = (foot_depth > 1.0) & (top_depth > 1.0)
    dx, dy = tx - fx, ty - fy
    angle = np.degrees(np.arctan2(dx, -dy))
    return (fx + tx) / 2, (fy + ty) / 2, np.nan_to_num(angle), upright
