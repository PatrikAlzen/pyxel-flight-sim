"""Main Pyxel application: title screen, flight, pause menu and map.

Keyboard / gamepad (see also ``controls.py``):

    up / down, left stick Y       nose down / nose up (menus: move)
    left / right, left stick X    roll (title: change seed)
    W / S, R1 / L1, triggers      throttle
    A / D, right stick X          rudder (on the ground: steer)
    B, gamepad B                  wheel brakes
    R, gamepad BACK               respawn here (title: random seed)
    M, gamepad X                  map (Z / X or shoulders to zoom)
    ESC / P, gamepad START        pause menu
    ENTER / SPACE, gamepad A      confirm
    C, gamepad Y                  cockpit / chase camera
    T, right stick click          time of day: day, dusk, night
    G                             free "ghost" camera; leaving it respawns there
    F1                            debug info

``flightsim --bench`` times terrain generation and rendering without a window.
"""

import argparse
import math
import random
import time
from enum import Enum, auto

import numpy as np
import pyxel

from flightsim import config
from flightsim.audio import Audio
from flightsim.controls import Action, flight_controls, pressed
from flightsim.render import palette
from flightsim.render.clouds import CloudLayer
from flightsim.render.effects import draw_particles, draw_wreck
from flightsim.render.framebuffer import image_array, screen_array
from flightsim.render.hud import HUD, TEXT, Hud, centered_text, shaded_rect, shadow_text
from flightsim.render.sprites import PLANE_REAR, PLANE_WRECK, TRANSPARENT, make_image
from flightsim.render.voxel import VoxelRenderer
from flightsim.render.worldmap import (
    ZOOMS,
    WorldMap,
    draw_map_overlay,
    draw_plane_marker,
)
from flightsim.sim.aircraft import Aircraft, AircraftSettings, FlightState
from flightsim.sim.camera import Camera
from flightsim.sim.chase import ChaseCamera, sprite_angle
from flightsim.sim.particles import Particles
from flightsim.terrain.generator import TerrainGenerator
from flightsim.terrain.world import TerrainWorld

W, H = config.SCREEN_W, config.SCREEN_H
DT = 1 / config.FPS
MAX_PITCH = math.radians(89)  # ghost camera; it can roll freely
TITLE_MAP_STEP = 32
PAUSE_ITEMS = ["RESUME", "RESPAWN", "NEW WORLD", "QUIT"]


class Mode(Enum):
    TITLE = auto()
    LOADING = auto()
    FLYING = auto()
    PAUSED = auto()
    MAP = auto()


def make_world(seed: int) -> TerrainWorld:
    return TerrainWorld(
        TerrainGenerator(seed), config.TERRAIN_LEVELS, config.TERRAIN_WORKERS
    )


class Timer:
    """Exponential moving average of a duration in milliseconds."""

    def __init__(self) -> None:
        self.ms = 0.0

    def add(self, seconds: float) -> None:
        self.ms += (seconds * 1000 - self.ms) * 0.1


def make_title_images(text: str) -> tuple[pyxel.Image, pyxel.Image]:
    """The title drawn once into small images, to be blitted scaled up.

    Colour 2 is the transparent background so black can be used for the shadow.
    """
    images = []
    for color in (pyxel.COLOR_BLACK, pyxel.COLOR_WHITE):
        img = pyxel.Image(len(text) * 4, 6)
        img.cls(2)
        img.text(0, 0, text, color)
        images.append(img)
    return images[0], images[1]


class App:
    def __init__(self) -> None:
        self.seed = config.SEED
        self.preview = WorldMap(TerrainGenerator(self.seed))
        self.world: TerrainWorld | None = None
        self.world_map: WorldMap | None = None
        self.renderer = VoxelRenderer(W, H)
        self.hud = Hud(self.renderer.focal, config.RESPAWN_DELAY)
        self.camera = Camera()
        self.chase = ChaseCamera()
        self.chase_view = False
        self.aircraft = Aircraft()
        self.clouds: CloudLayer | None = None
        self.particles = Particles()
        # Where the last crash happened; the wreck stays there, smoking,
        # until the next crash or a new world.
        self.wreck_at: np.ndarray | None = None
        self.time = 0.0  # simulated seconds, drives waves and cloud drift
        self.lighting = 0  # index into palette.LIGHTINGS
        self.mode = Mode.TITLE
        self.ghost = False
        self.menu_index = 0
        self.map_zoom = ZOOMS.index(16)
        self.load_total = 1
        self.progress = 0.0
        self.show_debug = False
        self.render_time = Timer()
        self.frame_time = Timer()
        self.last_frame = time.perf_counter()

        pyxel.init(
            W,
            H,
            title=config.TITLE,
            fps=config.FPS,
            display_scale=config.DISPLAY_SCALE,
            quit_key=pyxel.KEY_NONE,  # ESC opens the pause menu instead
        )
        palette.install()
        self.title_images = make_title_images(config.TITLE.upper())
        self.plane_sprite = make_image(PLANE_REAR)
        self.wreck_sprite = make_image(PLANE_WRECK)
        self.audio = Audio()
        pyxel.run(self.update, self.draw)

    # --- update -----------------------------------------------------------

    def update(self) -> None:
        if pressed(Action.DEBUG):
            self.show_debug = not self.show_debug
        if pressed(Action.TIME_OF_DAY):
            self.lighting = (self.lighting + 1) % len(palette.LIGHTINGS)
            palette.install(palette.LIGHTINGS[self.lighting])
        if self.mode is not Mode.FLYING or self.ghost:
            self.audio.silence()
        match self.mode:
            case Mode.TITLE:
                self.update_title()
            case Mode.LOADING:
                self.update_loading()
            case Mode.FLYING:
                self.update_flying()
            case Mode.PAUSED:
                self.update_paused()
            case Mode.MAP:
                self.update_map()

    def update_title(self) -> None:
        seed = self.seed
        if pressed(Action.LEFT):
            seed -= 1
        if pressed(Action.RIGHT):
            seed += 1
        if pressed(Action.RANDOM):
            seed = random.randrange(1, 100_000)
        if seed != self.seed:
            self.seed = seed
            self.preview = WorldMap(TerrainGenerator(seed))
            self.audio.menu_move()
        if pressed(Action.CONFIRM):
            self.audio.menu_confirm()
            self.start_world()
        elif pyxel.btnp(pyxel.KEY_ESCAPE) or pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()

    def start_world(self) -> None:
        self.world = make_world(self.seed)
        self.world_map = WorldMap(self.world.generator)
        sea_level = self.world.generator.settings.sea_level
        self.aircraft = Aircraft(settings=AircraftSettings(sea_level=sea_level))
        self.camera = Camera()
        self.chase.reset()
        self.clouds = CloudLayer(self.seed)
        self.particles.clear()
        self.wreck_at = None
        self.time = 0.0
        self.ghost = False
        # Background workers generate the starting area while a progress bar
        # is shown; afterwards they keep generating as the camera moves.
        self.world.update(0.0, 0.0)
        self.load_total = max(1, self.world.pending)
        self.progress = 0.0
        self.mode = Mode.LOADING

    def close_world(self) -> None:
        if self.world:
            self.world.close()
        self.world = self.world_map = None

    def update_loading(self) -> None:
        self.world.update(0.0, 0.0)
        pending = self.world.pending
        self.progress = 1 - pending / self.load_total
        if pending == 0:
            self.aircraft.respawn(0.0, 0.0, 0.0, self.world.height_at)
            self.follow_aircraft()
            self.mode = Mode.FLYING

    def update_flying(self) -> None:
        if pressed(Action.PAUSE):
            self.menu_index = 0
            self.mode = Mode.PAUSED
            return
        if pressed(Action.MAP):
            self.mode = Mode.MAP
            return
        if pressed(Action.GHOST):
            self.toggle_ghost()
        if pressed(Action.CAMERA):
            self.chase_view = not self.chase_view
            self.chase.reset()
        self.time += DT
        self.particles.update(DT, self.clouds.settings.drift)
        if self.ghost:
            self.move_ghost()
        else:
            self.fly()
            plane = self.aircraft
            agl = plane.altitude - self.world.height_at(plane.x, plane.y)
            self.audio.update(plane, agl)
        self.world.update(self.camera.x, self.camera.y)

    def update_paused(self) -> None:
        if pressed(Action.PAUSE) or pressed(Action.BACK):
            self.mode = Mode.FLYING
            return
        if pressed(Action.UP):
            self.menu_index = (self.menu_index - 1) % len(PAUSE_ITEMS)
            self.audio.menu_move()
        if pressed(Action.DOWN):
            self.menu_index = (self.menu_index + 1) % len(PAUSE_ITEMS)
            self.audio.menu_move()
        if not pressed(Action.CONFIRM):
            return
        self.audio.menu_confirm()
        match PAUSE_ITEMS[self.menu_index]:
            case "RESUME":
                self.mode = Mode.FLYING
            case "RESPAWN":
                self.respawn_here()
                self.mode = Mode.FLYING
            case "NEW WORLD":
                self.close_world()
                self.mode = Mode.TITLE
            case "QUIT":
                pyxel.quit()

    def update_map(self) -> None:
        if pressed(Action.MAP) or pressed(Action.BACK) or pressed(Action.PAUSE):
            self.mode = Mode.FLYING
        if pressed(Action.ZOOM_IN):
            self.map_zoom = max(0, self.map_zoom - 1)
        if pressed(Action.ZOOM_OUT):
            self.map_zoom = min(len(ZOOMS) - 1, self.map_zoom + 1)

    def toggle_ghost(self) -> None:
        self.ghost = not self.ghost
        if not self.ghost:
            cam = self.camera
            self.aircraft.respawn(cam.x, cam.y, cam.heading, self.world.height_at)
            self.aircraft.altitude = max(self.aircraft.altitude, cam.altitude)

    def respawn_here(self) -> None:
        plane = self.aircraft
        plane.respawn(plane.x, plane.y, plane.heading, self.world.height_at)
        self.chase.reset()

    def fly(self) -> None:
        plane = self.aircraft
        if pressed(Action.RESPAWN) or (
            plane.state is FlightState.CRASHED
            and plane.state_time > config.RESPAWN_DELAY
        ):
            self.respawn_here()
        was_crashed = plane.state is FlightState.CRASHED
        plane.update(flight_controls(), DT, self.world.height_at)
        if plane.state is FlightState.CRASHED:
            if not was_crashed:
                self.crashed()
            self.particles.burn(self.wreck_at + [0.0, 0.0, 3.0], DT)
        self.follow_aircraft()

    def crashed(self) -> None:
        """The moment of impact: leave a wreck on the ground and blow up."""
        plane = self.aircraft
        ground = max(self.world.height_at(plane.x, plane.y), plane.settings.sea_level)
        self.wreck_at = np.array([plane.x, plane.y, ground])
        self.particles.explode(self.wreck_at + [0.0, 0.0, 2.0])

    def follow_aircraft(self) -> None:
        plane, cam = self.aircraft, self.camera
        if plane.state is FlightState.CRASHED:
            self.watch_crash_site()
            return
        if self.chase_view:
            self.chase.update(plane, cam, DT, self.world.height_at)
            return
        cam.x, cam.y = plane.x, plane.y
        cam.altitude = plane.altitude + config.COCKPIT_HEIGHT
        cam.set_vectors(plane.forward, plane.up)

    def watch_crash_site(self) -> None:
        """Circle slowly round the wreck from above, outside any hillside."""
        plane, cam = self.aircraft, self.camera
        back = config.CRASH_VIEW_DISTANCE
        heading = plane.heading + config.CRASH_ORBIT_SPEED * plane.state_time
        fx, fy = math.sin(heading), -math.cos(heading)
        cam.x, cam.y = plane.x - fx * back, plane.y - fy * back
        site = max(plane.altitude, self.world.height_at(plane.x, plane.y))
        ground = max(self.world.height_at(cam.x, cam.y), site)
        cam.altitude = ground + config.CRASH_VIEW_HEIGHT
        cam.heading = heading % math.tau
        cam.pitch = -math.atan2(cam.altitude - site, back)
        cam.roll = 0.0

    def move_ghost(self) -> None:
        cam = self.camera
        speed = config.GHOST_SPEED * (
            config.GHOST_FAST if pyxel.btn(pyxel.KEY_SHIFT) else 1
        )

        cam.heading += config.GHOST_TURN * (
            pyxel.btn(pyxel.KEY_RIGHT) - pyxel.btn(pyxel.KEY_LEFT)
        )
        ahead = speed * (pyxel.btn(pyxel.KEY_UP) - pyxel.btn(pyxel.KEY_DOWN))
        side = speed * (pyxel.btn(pyxel.KEY_D) - pyxel.btn(pyxel.KEY_A))
        fx, fy = cam.forward
        rx, ry = cam.right
        cam.x += fx * ahead + rx * side
        cam.y += fy * ahead + ry * side
        cam.altitude += speed * (pyxel.btn(pyxel.KEY_W) - pyxel.btn(pyxel.KEY_S))

        tilt = config.GHOST_TURN
        cam.pitch += tilt * (pyxel.btn(pyxel.KEY_R) - pyxel.btn(pyxel.KEY_F))
        cam.roll += tilt * (pyxel.btn(pyxel.KEY_X) - pyxel.btn(pyxel.KEY_Z))
        if pyxel.btnp(pyxel.KEY_V):
            cam.pitch = cam.roll = 0.0
        cam.pitch = max(-MAX_PITCH, min(MAX_PITCH, cam.pitch))
        cam.roll = (cam.roll + math.pi) % math.tau - math.pi

        ground = self.world.height_at(cam.x, cam.y)
        cam.altitude = max(cam.altitude, ground + config.GHOST_CLEARANCE)

    # --- draw -------------------------------------------------------------

    def draw(self) -> None:
        now = time.perf_counter()
        self.frame_time.add(now - self.last_frame)
        self.last_frame = now

        match self.mode:
            case Mode.TITLE:
                self.draw_title()
            case Mode.LOADING:
                self.draw_loading()
            case Mode.MAP:
                plane = self.aircraft
                draw_map_overlay(
                    self.world_map,
                    plane.x,
                    plane.y,
                    plane.heading,
                    ZOOMS[self.map_zoom],
                )
            case Mode.FLYING | Mode.PAUSED:
                self.draw_view()
                if self.mode is Mode.PAUSED:
                    self.draw_pause_menu()
        if self.show_debug and self.world:
            self.draw_debug()

    def draw_view(self) -> None:
        start = time.perf_counter()
        self.renderer.render(
            screen_array(), self.world.levels, self.camera, self.time, self.clouds
        )
        self.render_time.add(time.perf_counter() - start)
        if self.wreck_at is not None:
            draw_wreck(
                self.wreck_sprite,
                TRANSPARENT,
                self.wreck_at,
                self.camera,
                self.renderer,
                self.particles,
            )
        else:
            draw_particles(self.particles, self.camera, self.renderer)
        if self.ghost:
            centered_text(H - 9, "GHOST CAMERA - G TO FLY FROM HERE", TEXT)
            return
        chase = self.chase_view and self.aircraft.state is not FlightState.CRASHED
        if chase:
            self.draw_plane()
        self.hud.draw(self.aircraft, self.camera, self.world, cockpit=not chase)

    def draw_plane(self) -> None:
        """The aircraft sprite as seen from the chase camera."""
        plane, cam = self.aircraft, self.camera
        position = np.array([plane.x, plane.y, plane.altitude])
        spot = cam.project(position, self.renderer.focal, W, H)
        if spot is None:
            return
        x, y, _ = spot
        img = self.plane_sprite
        pyxel.blt(
            x - img.width / 2,
            y - img.height / 2,
            img,
            0,
            0,
            img.width,
            img.height,
            TRANSPARENT,
            rotate=sprite_angle(plane, cam),
        )

    def draw_title(self) -> None:
        self.preview.draw(0.0, 0.0, TITLE_MAP_STEP)
        sx, sy = self.preview.to_screen(0.0, 0.0)
        draw_plane_marker(sx, sy, 0.0)

        shaded_rect(0, 18, W, 30)
        shadow, title = self.title_images
        x = (W - title.width) / 2
        pyxel.blt(x + 1, 30, shadow, 0, 0, title.width, title.height, 2, scale=3)
        pyxel.blt(x, 29, title, 0, 0, title.width, title.height, 2, scale=3)

        shaded_rect(0, H - 34, W, 34)
        centered_text(H - 30, f"<  SEED {self.seed}  >", pyxel.COLOR_YELLOW)
        centered_text(H - 21, "LEFT/RIGHT CHANGE SEED   R RANDOM", HUD)
        if pyxel.frame_count // 15 % 2 == 0:
            centered_text(H - 11, "PRESS ENTER OR (A) TO FLY", TEXT)

    def draw_loading(self) -> None:
        pyxel.cls(pyxel.COLOR_NAVY)
        centered_text(H // 2 - 22, f"SEED {self.seed}", pyxel.COLOR_YELLOW)
        centered_text(H // 2 - 10, "GENERATING TERRAIN", TEXT)
        bar_w = 120
        x = (W - bar_w) // 2
        pyxel.rectb(x, H // 2, bar_w, 6, TEXT)
        pyxel.rect(x + 1, H // 2 + 1, int((bar_w - 2) * self.progress), 4, HUD)

    def draw_pause_menu(self) -> None:
        width, height = 100, 22 + len(PAUSE_ITEMS) * 10
        x, y = (W - width) // 2, (H - height) // 2
        shaded_rect(0, 0, W, H)
        pyxel.rect(x, y, width, height, pyxel.COLOR_BLACK)
        pyxel.rectb(x, y, width, height, HUD)
        centered_text(y + 5, "PAUSED", pyxel.COLOR_YELLOW)
        for i, item in enumerate(PAUSE_ITEMS):
            selected = i == self.menu_index
            label = f"> {item} <" if selected else item
            centered_text(
                y + 17 + i * 10, label, TEXT if selected else pyxel.COLOR_GRAY
            )

    def draw_debug(self) -> None:
        cam = self.camera
        ground = self.world.height_at(cam.x, cam.y)
        fps = 1000 / self.frame_time.ms if self.frame_time.ms else 0
        plane = self.aircraft
        lines = [
            f"X {cam.x:.0f} Y {cam.y:.0f}  ALT {cam.altitude:.0f}  AGL {cam.altitude - ground:.0f}",
            f"RENDER {self.render_time.ms:4.1f}MS  {fps:4.1f}FPS  SEED {self.seed}",
            f"TILES PENDING {self.world.pending}  STALE {self.world.stale_tiles(cam.x, cam.y)}",
            f"{plane.state.name}  LIFT {plane.lift:.2f}  SINK {plane.sink:.1f}",
        ]
        for i, line in enumerate(lines):
            shadow_text(2, 24 + i * 7, line, TEXT)


def bench(frames: int, speed: float) -> None:
    """Time terrain generation and rendering without opening a window.

    Flies straight ahead at ``speed`` world units per frame, paced to the real
    frame rate like the game.
    """
    world = make_world(config.SEED)
    start = time.perf_counter()
    for _ in world.fill(0.0, 0.0):
        pass
    print(
        f"initial terrain {config.TERRAIN_LEVELS}: {time.perf_counter() - start:.2f} s"
    )

    renderer = VoxelRenderer(W, H)
    print(f"resolution {W}x{H}, depth slices {len(renderer.z)}")
    img = pyxel.Image(W, H)  # keep a reference: the view does not keep the image alive
    out = image_array(img)
    camera = Camera(heading=math.radians(70))
    frame_budget = 1 / config.FPS
    render_ms, frame_ms, stale_frames, max_pending = [], [], 0, 0
    for _ in range(frames):
        frame_start = time.perf_counter()
        fx, fy = camera.forward
        camera.x += fx * speed
        camera.y += fy * speed
        camera.altitude = world.height_at(camera.x, camera.y) + 60
        world.update(camera.x, camera.y)
        stale_frames += world.stale_tiles(camera.x, camera.y) > 0
        max_pending = max(max_pending, world.pending)
        render_start = time.perf_counter()
        renderer.render(out, world.levels, camera)
        now = time.perf_counter()
        render_ms.append((now - render_start) * 1000)
        frame_ms.append((now - frame_start) * 1000)
        time.sleep(max(0.0, frame_budget - (now - frame_start)))
    world.close()

    distance = frames * speed
    for name, values in [("render", render_ms), ("update+render", frame_ms)]:
        v = np.array(values[3:])  # skip warm-up
        print(
            f"{name:<14} mean {v.mean():5.2f} ms  p95 {np.percentile(v, 95):5.2f} ms  "
            f"max {v.max():5.2f} ms  ({1000 * frame_budget:.1f} ms budget)"
        )
    print(
        f"flew {distance:.0f} units at {speed:g}/frame: max pending tiles {max_pending}, "
        f"frames with stale terrain {stale_frames}/{frames}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyxel flight simulator")
    parser.add_argument(
        "--bench", action="store_true", help="time rendering headless and exit"
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument(
        "--speed", type=float, default=10.0, help="bench flight speed, units per frame"
    )
    args = parser.parse_args()
    if args.bench:
        bench(args.frames, args.speed)
    else:
        App()
