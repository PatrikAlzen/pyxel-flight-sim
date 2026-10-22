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
from collections.abc import Callable
from dataclasses import replace
from enum import Enum, auto

import numpy as np
import pyxel

from flightsim import config
from flightsim.audio import Audio
from flightsim.controls import Action, flight_controls, pressed
from flightsim.persist import Place, Profile, SaveStore, WorldSave
from flightsim.render import palette
from flightsim.render.bridges import draw_bridges
from flightsim.render.buildings import door_facing, draw_hangar, hangar_mesh
from flightsim.render.clouds import CloudLayer
from flightsim.render.effects import (
    WRECK_FULL_SIZE_DISTANCE,
    WRECK_TILT,
    draw_billboards,
    draw_particles,
    full_size_distance,
    in_front_of_terrain,
    project_points,
)
from flightsim.render.framebuffer import image_array, screen_array
from flightsim.render.hud import (
    HUD,
    TEXT,
    Hud,
    HudMode,
    centered_text,
    shaded_rect,
    shadow_text,
)
from flightsim.render.models import LOD_PIXELS, draw_plane, draw_wreck, wreck_size
from flightsim.render.rain import Rain
from flightsim.render.rings import draw_ring_arrow, draw_rings
from flightsim.render.sprites import (
    BROADLEAF,
    HANGAR,
    LIGHTHOUSE,
    SNOWY_SPRUCE,
    SPRUCE,
    TRANSPARENT,
    WINDSOCK,
    WORLD_COLORS,
    house_images,
    make_image,
    plane_sprites,
)
from flightsim.render.voxel import VoxelRenderer
from flightsim.render.worldmap import (
    ZOOMS,
    WorldMap,
    draw_map_overlay,
    draw_plane_marker,
)
from flightsim.sim.achievements import ACHIEVEMENTS, BY_KEY, Sample, Tracker
from flightsim.sim.aircraft import Aircraft, FlightState
from flightsim.sim.camera import Camera
from flightsim.sim.chase import ChaseCamera
from flightsim.sim.events import EventKind, EventLog
from flightsim.sim.freecam import FreeCamera, FreeMode, look_at
from flightsim.sim.particles import Particles
from flightsim.sim.planes import TRAINER, TYPES, AircraftType, by_name
from flightsim.sim.race import (
    Course,
    Phase,
    Race,
    build_course,
    format_time,
    ghost_position,
)
from flightsim.sim.replay import Playback, Recorder, tower_due, tower_spot
from flightsim.sim.traffic import Flight, Traffic, closest_approach, midair
from flightsim.sim.weather import PRESETS, Weather, turbulence_intensity
from flightsim.sim.wrecks import Wrecks
from flightsim.terrain.airfields import (
    Airfield,
    landing_report,
    water_landing_report,
)
from flightsim.terrain.bridges import Bridge
from flightsim.terrain.generator import TerrainGenerator
from flightsim.terrain.objects import HEIGHT, Kind, Objects, hit
from flightsim.terrain.settlements import Lighthouse
from flightsim.terrain.world import TerrainWorld

W, H = config.SCREEN_W, config.SCREEN_H
DT = 1 / config.FPS
MAX_PITCH = math.radians(89)  # ghost camera; it can roll freely
TITLE_MAP_STEP = 32
PLACE_ROWS = 7  # places listed at once in the places menu; the list scrolls
TOAST_SECONDS = 2.0
RESULT_SECONDS = 6.0  # how long a time-trial result stays on screen
PROPELLER_SPEED = 30.0  # radians per second at full throttle
ACHIEVEMENT_ROWS = 9  # shown at once in the pause menu's list
TITLE_ROWS = ("START", "AIRCRAFT", "SEED")  # top to bottom on the title screen
REPLAY_VIEWS = ("CHASE", "COCKPIT", "TOWER", "FREE")  # C cycles through them


class Mode(Enum):
    TITLE = auto()
    LOADING = auto()
    FLYING = auto()
    PAUSED = auto()
    MAP = auto()
    PHOTO = auto()
    REPLAY = auto()


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


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}S"


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
        self.base_renderer = self.renderer  # the photo mode zooms with others
        self.zoomed_renderers: dict[float, VoxelRenderer] = {}
        self.hud = Hud(self.renderer.focal, config.RESPAWN_DELAY)
        self.camera = Camera()
        self.chase = ChaseCamera()
        self.chase_view = False
        self.aircraft = Aircraft()
        self.clouds: CloudLayer | None = None
        self.particles = Particles()
        # Every crash site in this world; the wrecks stay where they fell.
        self.wrecks = Wrecks(config.MAX_WRECKS)
        self.time = 0.0  # simulated seconds, drives waves and cloud drift
        self.lighting = 0  # index into palette.LIGHTINGS
        self.weather: Weather | None = None
        self.weather_lock: str | None = None  # preset picked in the pause menu
        self.wind = (0.0, 0.0)
        self.rain = Rain()
        self.turbulence = 0.0  # 0..1 where the plane is, for the HUD
        self.installed_palette: tuple[int, int] | None = None  # (lighting, overcast)
        self.mode = Mode.TITLE
        self.ghost = False
        self.menu_index = 0
        self.places_menu = False
        self.store: SaveStore | None = None  # needs Pyxel, so set after init
        self.saved: WorldSave | None = None
        self.profile = Profile()  # loaded with the store
        self.refuelling = False
        self.title_summary: tuple[int, tuple[int, int]] | None = None  # (seed, counts)
        # Where loading drops the plane: x, y, heading, altitude (None = default).
        self.spawn: tuple[float, float, float, float | None] = (0.0, 0.0, 0.0, None)
        self.spawn_on_ground = False
        self.start_on_runway = False  # chosen on the title screen
        self.title_row = TITLE_ROWS.index("SEED")
        self.target_field: Airfield | None = None  # nearest airfield, for the HUD
        self.field_props: dict[int, dict[str, np.ndarray]] = {}  # per field number
        self.race: Race | None = None
        self.race_flash: tuple[float, float] | None = None  # (delta, shown until)
        self.record_ghost: list[list[float]] | None = None  # best run, replayed
        self.courses: dict[int, Course] = {}  # per field number
        self.objects_drawn = 0  # trees, houses and lighthouses in the last frame
        self.last_trial_field: Airfield | None = None  # for RETRY TIME TRIAL
        self.toast = ""
        self.toast_until = 0.0
        self.map_zoom = ZOOMS.index(16)
        self.load_total = 1
        self.progress = 0.0
        self.show_debug = False
        self.hud_mode = HudMode.FULL
        self.photo: FreeCamera | None = None
        self.photo_fov = 0  # index into config.PHOTO_FOVS
        self.photo_return: tuple[float, ...] = ()  # camera to restore afterwards
        self.photo_plane = True  # show the plane in photos
        self.photo_frame = False  # letterbox bars
        self.photo_help = True
        self.photo_requested = False
        self.photo_flash_until = 0.0
        # Replays: the recorder hears about crashes, landings and rings.
        self.events = EventLog()
        self.recorder = Recorder()
        self.events.subscribe(self.recorder.mark)
        self.events.subscribe(self.on_game_event)
        self.tracker: Tracker | None = None  # achievements, per world
        self.awards: list[str] = []  # achievement names still to announce
        self.award_until = 0.0
        self.over_river = False  # updated every few frames
        self.achievements_menu = False
        self.playback: Playback | None = None
        self.traffic: Traffic | None = None
        self.flights: list[Flight] = []  # other aircraft near the camera now
        self.near_misses: set[tuple] = set()  # flights already reported
        self.closest: dict[tuple, float] = {}  # nearest each flight has come
        self.flights_before: dict[tuple, np.ndarray] = {}  # positions a frame ago
        self.replay_plane = Aircraft()  # posed from the recording each frame
        self.replay_chase = ChaseCamera()
        self.replay_view = 0  # index into REPLAY_VIEWS
        self.replay_free: FreeCamera | None = None
        self.tower: np.ndarray | None = None
        self.tower_side = 1.0
        self.replay_help = True
        self.live_particles = self.particles  # set aside during a replay
        self.replay_return: tuple[float, ...] = ()
        self.after_loading: Callable[[], None] | None = None
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
        # Rear, front, side, top and wreck images for each aircraft type.
        self.type_views = [
            {
                view: make_image(rows)
                for view, rows in plane_sprites(
                    t.body, t.trim, t.settings.floats
                ).items()
            }
            for t in TYPES
        ]
        self.hangar_model = hangar_mesh(config.HANGAR_WIDTH)
        self.prop_sprites = {
            "hangar": (make_image(HANGAR), config.HANGAR_WIDTH),
            # Blowing right and, mirrored, left as seen from the camera.
            "windsock": (
                [make_image(WINDSOCK), make_image([r[::-1] for r in WINDSOCK])],
                config.WINDSOCK_WIDTH,
            ),
        }
        # Trees (one image per Tree variant), houses (one per roof) and the
        # lighthouse, in that order; OBJECT_SPRITE_BASE maps a Kind to its first.
        self.object_sprites = [
            make_image(BROADLEAF, WORLD_COLORS),
            make_image(SPRUCE, WORLD_COLORS),
            make_image(SNOWY_SPRUCE, WORLD_COLORS),
            *house_images(),
            make_image(LIGHTHOUSE, WORLD_COLORS),
        ]
        kinds = [Kind.TREE] * 3 + [Kind.HOUSE] * 4 + [Kind.LIGHTHOUSE]
        # Full-size distance per sprite, per unit of focal length (zoom changes it).
        self.object_full_size_per_focal = [
            HEIGHT[kind] / img.height for kind, img in zip(kinds, self.object_sprites)
        ]
        self.audio = Audio()
        self.store = SaveStore(pyxel.user_data_dir(config.SAVE_VENDOR, config.TITLE))
        self.profile = self.store.load_profile()
        self.audio.set_engine(self.plane_type.engine_note)
        pyxel.run(self.update, self.draw)

    # --- update -----------------------------------------------------------

    def update(self) -> None:
        if pressed(Action.HUD_MODE) and self.mode is not Mode.PHOTO:
            self.hud_mode = self.hud_mode.next()
            self.show_toast(self.hud_mode.value)
        if pressed(Action.DEBUG):
            self.show_debug = not self.show_debug
        if pressed(Action.TIME_OF_DAY):
            self.lighting = (self.lighting + 1) % len(palette.LIGHTINGS)
            self.install_palette()
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
            case Mode.PHOTO:
                self.update_photo()
            case Mode.REPLAY:
                self.update_replay()

    def update_title(self) -> None:
        seed = self.seed
        row = TITLE_ROWS[self.title_row]
        step = pressed(Action.RIGHT) - pressed(Action.LEFT)
        if row == "SEED":
            seed += step
        elif row == "START" and step:
            self.start_on_runway = not self.start_on_runway
            self.audio.menu_move()
        elif row == "AIRCRAFT" and step:
            self.choose_aircraft(
                TYPES[(TYPES.index(self.plane_type) + step) % len(TYPES)]
            )
            self.audio.menu_move()
        if pressed(Action.RANDOM):
            seed = random.randrange(1, 100_000)
        if seed != self.seed:
            self.seed = seed
            self.preview = WorldMap(TerrainGenerator(seed))
            self.audio.menu_move()
        rows = len(TITLE_ROWS)
        if pressed(Action.UP):
            self.title_row = (self.title_row - 1) % rows
            self.audio.menu_move()
        if pressed(Action.DOWN):
            self.title_row = (self.title_row + 1) % rows
            self.audio.menu_move()
        if pressed(Action.CONFIRM):
            self.audio.menu_confirm()
            self.title_summary = None
            self.start_world()
        elif pyxel.btnp(pyxel.KEY_ESCAPE) or pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()

    def start_world(self) -> None:
        self.world = make_world(self.seed)
        self.world_map = WorldMap(self.world.generator)
        sea_level = self.world.generator.settings.sea_level
        self.aircraft = Aircraft(
            settings=replace(self.plane_type.settings, sea_level=sea_level)
        )
        self.aircraft.uses_fuel = self.profile.fuel
        self.camera = Camera()
        self.chase.reset()
        self.clouds = CloudLayer(self.seed)
        self.weather = Weather(self.seed)
        if self.weather_lock is not None:
            self.weather.lock(next(p for p in PRESETS if p.name == self.weather_lock))
            self.weather.update(self.weather.transition)  # start in it
        self.wind = (0.0, 0.0)
        self.turbulence = 0.0
        self.apply_weather()
        self.particles.clear()
        self.wrecks.clear()
        self.recorder.clear()
        self.tracker = Tracker(self.profile.earned, self.profile.progress, self.seed)
        self.traffic = Traffic(self.seed, self.world.generator, len(TYPES))
        self.flights = []
        self.near_misses.clear()
        self.closest.clear()
        self.saved = self.store.load(self.seed)
        for wreck in self.saved.wrecks:
            kind = int(wreck[3]) if len(wreck) > 3 else 0  # older: the trainer
            self.wrecks.add(np.array(wreck[:3]), min(kind, len(TYPES) - 1))
        # Saved under older terrain (before rivers, say) a wreck could float
        # above a valley or sink into a hill; the ground decides.
        generator = self.world.generator
        self.wrecks.settle(
            lambda x, y: np.maximum(generator.elevation(x, y), sea_level)
        )
        self.time = 0.0
        self.ghost = False
        self.field_props.clear()
        self.courses.clear()
        self.race = None
        self.last_trial_field = None
        self.target_field = None
        field = (
            self.world.generator.airfields.nearest(0.0, 0.0, config.FIELD_SEARCH_RINGS)
            if self.start_on_runway
            else None
        )
        if field is None:
            self.begin_loading(0.0, 0.0, 0.0)
        else:
            # At the start of the runway, pointing down it.
            x, y = field.point(-field.length / 2 + config.RUNWAY_START_OFFSET)
            self.begin_loading(x, y, field.heading, on_ground=True)

    def begin_loading(
        self,
        x: float,
        y: float,
        heading: float,
        altitude: float | None = None,
        on_ground: bool = False,
        then: Callable[[], None] | None = None,
    ) -> None:
        """Show a progress bar while the terrain around (x, y) is generated.

        Background workers generate the area; afterwards they keep generating
        as the camera moves. The plane is placed there once everything is in,
        or, with ``then``, that is called instead (the camera just looks there).
        """
        self.after_loading = then
        self.spawn = (x, y, heading, altitude)
        self.spawn_on_ground = on_ground
        self.world.update(x, y)
        self.load_total = max(1, self.world.pending)
        self.progress = 0.0
        self.mode = Mode.LOADING

    def close_world(self) -> None:
        if self.tracker is not None and self.tracker.dirty:
            self.save_profile()
        if self.world:
            self.world.close()
        self.world = self.world_map = None

    def update_loading(self) -> None:
        x, y, heading, altitude = self.spawn
        self.world.update(x, y)
        pending = self.world.pending
        self.progress = 1 - pending / self.load_total
        if pending == 0:
            if self.after_loading is not None:
                then, self.after_loading = self.after_loading, None
                then()
                return
            if self.spawn_on_ground:
                self.aircraft.park(x, y, heading, self.world.height_at)
                self.ghost = False
                self.chase.reset()
                self.follow_aircraft()
            else:
                self.place_aircraft(x, y, heading, altitude)
            self.mode = Mode.FLYING

    def place_aircraft(
        self, x: float, y: float, heading: float, altitude: float | None
    ) -> None:
        """Fly on from (x, y), at ``altitude`` if given but never close to the ground."""
        plane = self.aircraft
        plane.respawn(x, y, heading, self.world.height_at)
        if altitude is not None:
            ground = self.world.height_at(x, y)
            plane.altitude = max(altitude, ground + config.PLACE_MIN_CLEARANCE)
        self.ghost = False
        self.chase.reset()
        self.follow_aircraft()

    def update_flying(self) -> None:
        if pressed(Action.PAUSE):
            if self.tracker.dirty:
                self.save_profile()
            self.menu_index = 0
            self.places_menu = False
            self.mode = Mode.PAUSED
            return
        if pressed(Action.SAVE_PLACE):
            self.save_place()
        if pressed(Action.PHOTO):
            self.enter_photo()
            return
        if pressed(Action.REPLAY):
            self.start_replay()
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
        if pyxel.frame_count % 10 == 0:
            self.target_field = self.world.generator.airfields.nearest(
                self.camera.x, self.camera.y, config.FIELD_SEARCH_RINGS
            )
            self.discover_landmarks()
        self.weather.update(DT)
        self.wind = self.weather.wind()
        self.clouds.advance(self.wind, DT)
        self.apply_weather()
        self.particles.update(DT, self.wind)
        cam = self.camera
        self.particles.smoulder(
            self.wrecks.near(cam.x, cam.y, config.WRECK_SMOKE_DISTANCE), DT
        )
        self.update_traffic()
        if self.ghost:
            self.move_ghost()
        else:
            self.fly()
            self.watch_flight()
            self.recorder.record(self.aircraft, DT)
            plane = self.aircraft
            agl = plane.altitude - self.world.height_at(plane.x, plane.y)
            self.audio.update(plane, agl, self.weather.current.rain)
        self.rain.update(DT, self.wind, self.camera)
        self.world.update(self.camera.x, self.camera.y)

    def update_paused(self) -> None:
        if self.places_menu:
            self.update_places_menu()
            return
        if self.achievements_menu:
            self.update_achievements_menu()
            return
        if pressed(Action.PAUSE) or pressed(Action.BACK):
            self.mode = Mode.FLYING
            return
        items = self.pause_items()
        if pressed(Action.UP):
            self.menu_index = (self.menu_index - 1) % len(items)
            self.audio.menu_move()
        if pressed(Action.DOWN):
            self.menu_index = (self.menu_index + 1) % len(items)
            self.audio.menu_move()
        if not pressed(Action.CONFIRM):
            return
        self.audio.menu_confirm()
        match items[self.menu_index]:
            case "RESUME":
                self.mode = Mode.FLYING
            case "RESPAWN":
                self.respawn_here()
                self.mode = Mode.FLYING
            case "PLACES":
                self.places_menu = True
                self.menu_index = 0
            case "TIME TRIAL":
                self.start_time_trial()
            case "RETRY TIME TRIAL":
                self.start_time_trial(self.last_trial_field)
            case "END TIME TRIAL":
                self.end_race("TIME TRIAL ENDED")
                self.mode = Mode.FLYING
            case item if item.startswith("WEATHER"):
                self.cycle_weather()
            case item if item.startswith("FUEL"):
                self.toggle_fuel()
            case item if item.startswith("TRAFFIC"):
                self.profile.traffic = not self.profile.traffic
                self.save_profile()
            case item if item.startswith("AIRCRAFT"):
                self.change_aircraft()
            case "PHOTO MODE":
                self.enter_photo()
            case item if item.startswith("ACHIEVEMENTS"):
                self.achievements_menu = True
                self.menu_index = 0
            case "REPLAY":
                self.start_replay()
            case "NEW WORLD":
                self.close_world()
                self.mode = Mode.TITLE
            case "QUIT":
                pyxel.quit()

    def pause_items(self) -> list[str]:
        if self.race:
            trial = ["END TIME TRIAL"]
        else:
            trial = ["TIME TRIAL"]
            if self.last_trial_field is not None:
                trial.append("RETRY TIME TRIAL")
        weather = f"WEATHER: {self.weather_lock or 'AUTO'}"
        return [
            "RESUME",
            "RESPAWN",
            "PLACES",
            *trial,
            weather,
            f"FUEL: {'ON' if self.profile.fuel else 'OFF'}",
            f"AIRCRAFT: {self.plane_type.name}",
            f"TRAFFIC: {'ON' if self.profile.traffic else 'OFF'}",
            "PHOTO MODE",
            "REPLAY",
            f"ACHIEVEMENTS {len(self.profile.earned)}/{len(ACHIEVEMENTS)}",
            "NEW WORLD",
            "QUIT",
        ]

    def places_menu_items(self) -> list[str | Place]:
        return ["SAVE HERE", *self.saved.places, "BACK"]

    def update_places_menu(self) -> None:
        """Save, visit or delete places: a list under PLACES in the pause menu."""
        items = self.places_menu_items()
        if pressed(Action.BACK) or pressed(Action.PAUSE):
            self.places_menu = False
            self.menu_index = self.pause_items().index("PLACES")
            return
        if pressed(Action.UP):
            self.menu_index = (self.menu_index - 1) % len(items)
            self.audio.menu_move()
        if pressed(Action.DOWN):
            self.menu_index = (self.menu_index + 1) % len(items)
            self.audio.menu_move()
        item = items[self.menu_index]
        if pressed(Action.DELETE) and isinstance(item, Place):
            self.saved.places.remove(item)
            self.persist()
            self.menu_index = min(self.menu_index, len(items) - 2)
            self.audio.menu_move()
            return
        if not pressed(Action.CONFIRM):
            return
        self.audio.menu_confirm()
        if item == "SAVE HERE":
            self.save_place()
        elif item == "BACK":
            self.places_menu = False
            self.menu_index = self.pause_items().index("PLACES")
        else:
            self.go_to(item)

    def save_place(self) -> None:
        """Remember where the plane (or the ghost or photo camera) is now."""
        free = self.ghost or self.mode is Mode.PHOTO
        if self.aircraft.state is FlightState.CRASHED and not free:
            return
        source = self.camera if free else self.aircraft
        place = self.saved.add_place(
            source.x, source.y, source.altitude, source.heading
        )
        if self.persist():
            self.show_toast(f"{place.name} SAVED")

    def go_to(self, place: Place) -> None:
        """Fly on from a saved place; far jumps wait for the terrain there."""
        self.places_menu = False
        self.end_race()
        here = math.hypot(place.x - self.camera.x, place.y - self.camera.y)
        if here > self.world.levels[0].safe_radius:
            self.begin_loading(place.x, place.y, place.heading, place.altitude)
        else:
            self.place_aircraft(place.x, place.y, place.heading, place.altitude)
            self.mode = Mode.FLYING
        self.show_toast(place.name)

    def persist(self) -> bool:
        """Write this world's save file; tell the player if that fails."""
        try:
            self.store.save(self.saved)
            return True
        except OSError:
            self.show_toast("COULD NOT SAVE")
            return False

    # --- weather ----------------------------------------------------------

    def apply_weather(self) -> None:
        """Hand the current weather to the clouds, the fog and the palette."""
        c = self.weather.current
        self.clouds.coverage = c.cloud_cover
        self.clouds.altitude = c.cloud_base
        self.renderer.visibility = c.visibility
        self.install_palette()

    def install_palette(self) -> None:
        """The palette for the time of day under the current cloud cover.

        Overcast is rounded to a few steps: rebuilding the palette every frame
        would be wasted work, and the steps are too small to notice.
        """
        overcast = 0.0 if self.weather is None else self.weather.current.overcast
        key = (self.lighting, round(overcast * config.OVERCAST_STEPS))
        if key == self.installed_palette:
            return
        self.installed_palette = key
        lighting = palette.LIGHTINGS[self.lighting]
        palette.install(palette.overcast(lighting, key[1] / config.OVERCAST_STEPS))

    def cycle_weather(self) -> None:
        """Pause menu: AUTO, then each preset in turn, then back to AUTO."""
        names = [p.name for p in PRESETS]
        if self.weather_lock is None:
            self.weather_lock = names[0]
        elif self.weather_lock == names[-1]:
            self.weather_lock = None
        else:
            self.weather_lock = names[names.index(self.weather_lock) + 1]
        lock = next((p for p in PRESETS if p.name == self.weather_lock), None)
        self.weather.lock(lock)

    def turbulence_here(self) -> float:
        """How bumpy the air is around the plane: weather, ridges and cloud."""
        plane = self.aircraft
        if plane.state is not FlightState.FLYING:
            return 0.0
        d = config.TURBULENCE_PROBE
        heights = [
            self.world.height_at(plane.x + dx, plane.y + dy)
            for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d))
        ]
        slope = max(abs(heights[0] - heights[1]), abs(heights[2] - heights[3])) / (
            2 * d
        )
        agl = plane.altitude - self.world.height_at(plane.x, plane.y)
        in_cloud = self.clouds.inside(plane.x, plane.y, plane.altitude)
        return turbulence_intensity(self.weather.current, agl, slope, in_cloud)

    # --- replay -----------------------------------------------------------

    def start_replay(self) -> None:
        """Play back the last minute; the live game waits where it is."""
        recording = self.recorder.recording()
        if recording is None or recording.duration < config.REPLAY_MIN:
            self.show_toast("NOTHING TO REPLAY")
            return
        cam = self.camera
        self.replay_return = (cam.x, cam.y, cam.altitude, cam.heading, cam.pitch)
        self.replay_return += (cam.roll,)
        self.playback = Playback(recording)
        self.live_particles, self.particles = self.particles, Particles()
        self.replay_chase.reset()
        self.replay_free = None
        self.tower = None
        self.pose_replay_plane()
        self.follow_replay(DT)
        self.load_around_camera(Mode.REPLAY, self.replay_return[:2])

    def load_around_camera(self, mode: Mode, came_from: tuple[float, float]) -> None:
        """Switch to ``mode``, through the loading screen if the camera has
        moved too far from ``came_from`` for the terrain there to be ready."""
        cam = self.camera
        far = self.world.levels[0].safe_radius * config.REPLAY_RELOAD
        if math.hypot(cam.x - came_from[0], cam.y - came_from[1]) > far:
            self.begin_loading(
                cam.x, cam.y, cam.heading, then=lambda: self.set_mode(mode)
            )
        else:
            self.mode = mode

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode

    def exit_replay(self) -> None:
        self.particles = self.live_particles
        self.playback = None
        cam = self.camera
        came_from = (cam.x, cam.y)
        cam.x, cam.y, cam.altitude, cam.heading, cam.pitch, cam.roll = (
            self.replay_return
        )
        if not self.ghost:
            self.chase.reset()
            self.follow_aircraft()
        self.load_around_camera(Mode.FLYING, came_from)

    def pose_replay_plane(self) -> None:
        pose = self.playback.pose()
        plane = self.replay_plane
        plane.x, plane.y, plane.altitude = (float(v) for v in pose.position)
        plane.forward, plane.up = pose.forward, pose.up
        plane.throttle, plane.speed, plane.state = pose.throttle, pose.speed, pose.state

    def update_replay(self) -> None:
        playback = self.playback
        if pressed(Action.REPLAY) or pressed(Action.PAUSE):
            self.exit_replay()
            return
        if pressed(Action.CONFIRM):
            playback.paused = not playback.paused
            if playback.finished:
                playback.time, playback.paused = 0.0, False
                self.particles.clear()
        if pressed(Action.UP):
            playback.faster(1)
        if pressed(Action.DOWN):
            playback.faster(-1)
        if pressed(Action.CAMERA):
            self.replay_view = (self.replay_view + 1) % len(REPLAY_VIEWS)
            self.replay_free = None
            self.tower = None
            self.replay_chase.reset()
        if pressed(Action.HUD_MODE):
            self.replay_help = not self.replay_help
        seek = config.REPLAY_SEEK * (pressed(Action.RIGHT) - pressed(Action.LEFT))
        before = (self.camera.x, self.camera.y)
        if seek:
            playback.seek(seek)
            self.particles.clear()  # smoke from another moment
            self.replay_chase.reset()
            self.tower = None

        dt = 0.0 if playback.paused else DT * playback.speed
        for event in playback.step(DT):
            if event.kind is EventKind.CRASH:
                self.particles.explode(event.position + [0.0, 0.0, 2.0])
            elif event.text:
                self.show_toast(event.text)
        self.pose_replay_plane()
        crash = playback.recording.crash()
        if crash and self.replay_plane.state is FlightState.CRASHED and dt:
            self.particles.burn(crash[1].position + [0.0, 0.0, 3.0], dt)
        if dt:
            self.particles.update(dt, self.wind)
        self.follow_replay(dt)
        if seek:
            self.load_around_camera(Mode.REPLAY, before)
        self.world.update(self.camera.x, self.camera.y)

    def follow_replay(self, dt: float) -> None:
        """Move the camera for the replay view chosen with C."""
        plane, cam = self.replay_plane, self.camera
        ground = self.world.height_at
        target = np.array([plane.x, plane.y, plane.altitude])
        crashed = plane.state is FlightState.CRASHED
        match REPLAY_VIEWS[self.replay_view]:
            case "CHASE" | "COCKPIT" if crashed:
                look_at(cam, target)  # watch the crash from where the camera was
            case "CHASE":
                self.replay_chase.update(plane, cam, max(dt, 1e-3), ground)
            case "COCKPIT":
                cam.x, cam.y = plane.x, plane.y
                cam.altitude = plane.altitude + config.COCKPIT_HEIGHT
                cam.set_vectors(plane.forward, plane.up)
            case "TOWER":
                if tower_due(self.tower, target, plane.forward):
                    self.tower = tower_spot(
                        self.playback.recording,
                        self.playback.time,
                        ground,
                        side=self.tower_side,
                    )
                    self.tower_side = -self.tower_side
                cam.x, cam.y, cam.altitude = (float(v) for v in self.tower)
                look_at(cam, target)
            case "FREE":
                if self.replay_free is None:
                    self.replay_free = FreeCamera(cam, target)
                free = self.replay_free
                free.target = free.anchor = target

                def keys(positive: int, negative: int) -> float:
                    return float(pyxel.btn(positive) - pyxel.btn(negative))

                free.orbit(
                    keys(pyxel.KEY_D, pyxel.KEY_A),
                    keys(pyxel.KEY_E, pyxel.KEY_Q),
                    keys(pyxel.KEY_W, pyxel.KEY_S),
                    DT,
                    ground,
                )

    def draw_replay(self) -> None:
        self.draw_view()
        plane = self.replay_plane
        cockpit = REPLAY_VIEWS[self.replay_view] == "COCKPIT"
        if plane.state is not FlightState.CRASHED and not cockpit:
            self.draw_model(plane, self.playback.time)
        playback = self.playback
        duration = playback.recording.duration
        shaded_rect(0, H - 12, W, 12)
        x0, x1 = 4, W - 4
        pyxel.line(x0, H - 11, x1, H - 11, pyxel.COLOR_GRAY)
        done = x0 + (x1 - x0) * playback.time / duration
        pyxel.line(x0, H - 11, done, H - 11, pyxel.COLOR_YELLOW)
        state = "END" if playback.finished else "PAUSED" if playback.paused else ""
        shadow_text(
            4,
            H - 8,
            f"REPLAY {format_time(playback.time)}/{format_time(duration)}"
            f"  {playback.speed:g}X  {REPLAY_VIEWS[self.replay_view]}  {state}",
            TEXT,
        )
        if self.replay_help:
            lines = ["</> SEEK  SPACE PAUSE  ^/v SPEED  C CAMERA  F3 BACK"]
            if REPLAY_VIEWS[self.replay_view] == "FREE":
                lines.insert(0, "A/D CIRCLE  Q/E TILT  W/S CLOSER/FURTHER")
            for i, line in enumerate(lines):
                shadow_text(4, H - 20 - 8 * (len(lines) - 1 - i), line, TEXT)

    # --- photo mode -------------------------------------------------------

    @property
    def zoom(self) -> float:
        """How much larger things look than at the normal field of view."""
        return self.renderer.focal / self.base_renderer.focal

    def enter_photo(self) -> None:
        """Freeze everything and hand the camera to the player."""
        cam, plane = self.camera, self.aircraft
        self.photo_return = (cam.x, cam.y, cam.altitude, cam.heading, cam.pitch)
        self.photo_return += (cam.roll,)
        target = np.array([plane.x, plane.y, plane.altitude])
        self.photo = FreeCamera(cam, target)
        if self.ghost:  # nothing to circle: look around from here
            self.photo.anchor = np.array(cam.position, dtype=float)
            self.photo.toggle_mode()
        self.photo_requested = False
        self.mode = Mode.PHOTO

    def exit_photo(self) -> None:
        self.set_fov(0)
        cam = self.camera
        cam.x, cam.y, cam.altitude, cam.heading, cam.pitch, cam.roll = self.photo_return
        self.photo = None
        self.mode = Mode.FLYING
        if not self.ghost:
            self.chase.reset()
            self.follow_aircraft()

    def set_fov(self, index: int) -> None:
        """Zoom by swapping in a renderer for another field of view."""
        self.photo_fov = index
        fov = config.PHOTO_FOVS[index]
        base = self.base_renderer
        if index == 0:
            renderer = base
        else:
            if fov not in self.zoomed_renderers:
                settings = replace(base.settings, fov_degrees=fov)
                self.zoomed_renderers[fov] = VoxelRenderer(W, H, settings)
            renderer = self.zoomed_renderers[fov]
        renderer.visibility = self.renderer.visibility
        self.renderer = renderer

    def update_photo(self) -> None:
        """Move the camera; nothing else in the world moves."""
        if pressed(Action.PHOTO) or pressed(Action.PAUSE):
            self.exit_photo()
            return
        photo, ground = self.photo, self.world.height_at
        if pyxel.btnp(pyxel.KEY_TAB):
            photo.toggle_mode()
        if pressed(Action.ZOOM_IN) and self.photo_fov < len(config.PHOTO_FOVS) - 1:
            self.set_fov(self.photo_fov + 1)
        if pressed(Action.ZOOM_OUT) and self.photo_fov > 0:
            self.set_fov(self.photo_fov - 1)
        if pyxel.btnp(pyxel.KEY_V):
            self.cycle_weather()
            self.weather.update(self.weather.transition)  # no blend: show it now
            self.apply_weather()
            self.show_toast(f"WEATHER: {self.weather.current.name}")
        if pyxel.btnp(pyxel.KEY_F):
            self.photo_frame = not self.photo_frame
        if pyxel.btnp(pyxel.KEY_O):
            self.photo_plane = not self.photo_plane
        if pressed(Action.HUD_MODE):
            self.photo_help = not self.photo_help
        if pressed(Action.SAVE_PLACE):
            self.save_place()
        if pressed(Action.CONFIRM):
            self.photo_requested = True  # taken when the frame is drawn

        def keys(positive: int, negative: int) -> float:
            return float(pyxel.btn(positive) - pyxel.btn(negative))

        turn = keys(pyxel.KEY_RIGHT, pyxel.KEY_LEFT)
        tilt = keys(pyxel.KEY_UP, pyxel.KEY_DOWN)
        if photo.mode is FreeMode.ORBIT:
            photo.orbit(turn, tilt, keys(pyxel.KEY_W, pyxel.KEY_S), DT, ground)
        else:
            photo.fly(
                turn,
                tilt,
                keys(pyxel.KEY_W, pyxel.KEY_S),
                keys(pyxel.KEY_D, pyxel.KEY_A),
                keys(pyxel.KEY_E, pyxel.KEY_Q),
                DT,
                ground,
                fast=pyxel.btn(pyxel.KEY_SHIFT),
            )
        self.world.update(self.camera.x, self.camera.y)

    def take_photo(self) -> None:
        """Save the screen as it is now, before any help text is drawn on it."""
        self.photo_requested = False
        cam = self.camera
        lighting = palette.LIGHTINGS[self.lighting].name
        try:
            path = self.store.photo_path(self.seed, cam.x, cam.y, lighting)
            pyxel.screen.save(str(path), config.PHOTO_SCALE)
        except OSError:
            self.show_toast("COULD NOT SAVE PHOTO")
            return
        self.photo_flash_until = time.perf_counter() + config.PHOTO_FLASH
        self.show_toast(f"PHOTO SAVED: {path.name}.png", 3.0)
        self.award(self.tracker.took_photo())

    def show_toast(self, text: str, seconds: float = TOAST_SECONDS) -> None:
        self.toast = text
        self.toast_until = time.perf_counter() + seconds

    # --- time trials ------------------------------------------------------

    def start_time_trial(self, field: Airfield | None = None) -> None:
        """Line up on a runway (the nearest one by default) for its ring course."""
        if field is None:
            field = self.world.generator.airfields.nearest(
                self.camera.x, self.camera.y, config.FIELD_SEARCH_RINGS
            )
        self.mode = Mode.FLYING
        if field is None:
            self.show_toast("NO AIRFIELD NEARBY")
            return
        if field.number not in self.courses:
            self.courses[field.number] = build_course(
                self.world.generator.elevation, field, self.seed
            )
        course = self.courses[field.number]
        self.last_trial_field = field
        record = self.saved.records.get(field.number, {}).get(self.plane_type.name)
        self.race = Race(course, record=record["splits"] if record else None)
        self.record_ghost = record["ghost"] if record else None
        self.race_flash = None
        x, y = field.point(-field.length / 2 + config.RUNWAY_START_OFFSET)
        if (
            math.hypot(x - self.camera.x, y - self.camera.y)
            > self.world.levels[0].safe_radius
        ):
            self.begin_loading(x, y, field.heading, on_ground=True)
        else:
            self.aircraft.park(x, y, field.heading, self.world.height_at)
            self.ghost = False
            self.chase.reset()
            self.follow_aircraft()
        self.show_toast(f"TIME TRIAL: FIELD {field.number}, {len(course.rings)} RINGS")

    def end_race(self, message: str | None = None) -> None:
        if self.race is None:
            return
        self.race = None
        if message:
            self.show_toast(message)

    def ring_passed(self, ring: int) -> None:
        race = self.race
        plane = self.aircraft
        rings = len(race.course.rings)
        self.events.emit(
            EventKind.RING,
            (plane.x, plane.y, plane.altitude),
            f"RING {ring + 1}/{rings}",
        )
        if race.phase is Phase.FINISHED:
            self.finish_race()
            return
        self.audio.ring_passed()
        delta = race.delta(ring)
        if delta is not None:
            self.race_flash = (delta, time.perf_counter() + 2.0)

    def finish_race(self) -> None:
        race = self.race
        self.audio.race_finished()
        best = race.record[-1] if race.record else None
        self.award(self.tracker.finished_trial(race.time, best))
        result = f"FINISHED {format_time(race.time)}"
        if race.record is not None:
            result += f" ({race.time - race.record[-1]:+.1f})"
        if race.is_record:
            records = self.saved.records.setdefault(race.course.field_number, {})
            records[self.plane_type.name] = {
                "splits": [round(t, 3) for t in race.splits],
                "ghost": race.ghost,
            }
            self.persist()
            result += " NEW RECORD"
        self.race = None
        self.show_toast(result, RESULT_SECONDS)

    def update_map(self) -> None:
        if pressed(Action.MAP) or pressed(Action.BACK) or pressed(Action.PAUSE):
            self.mode = Mode.FLYING
        if pressed(Action.ZOOM_IN):
            self.map_zoom = max(0, self.map_zoom - 1)
        if pressed(Action.ZOOM_OUT):
            self.map_zoom = min(len(ZOOMS) - 1, self.map_zoom + 1)

    def toggle_ghost(self) -> None:
        self.end_race("TIME TRIAL ENDED")
        self.ghost = not self.ghost
        if not self.ghost:
            cam = self.camera
            self.aircraft.respawn(cam.x, cam.y, cam.heading, self.world.height_at)
            self.aircraft.altitude = max(self.aircraft.altitude, cam.altitude)

    def respawn_here(self) -> None:
        self.end_race("TIME TRIAL ENDED")
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
        was_flying = plane.state is FlightState.FLYING
        start = np.array([plane.x, plane.y, plane.altitude])
        self.turbulence = self.turbulence_here()
        kick = self.weather.turbulence_kick(self.turbulence)
        bump = (
            kick[0] * self.plane_type.turbulence,
            kick[1] * self.plane_type.turbulence,
        )
        plane.update(
            flight_controls(),
            DT,
            self.world.height_at,
            self.wind,
            bump,
            runway=self.on_runway,
        )
        if was_flying and plane.state is FlightState.LANDED:
            self.touched_down()
        self.refuel()
        if plane.state is not FlightState.CRASHED:
            self.check_object_collision()
        if plane.state is not FlightState.CRASHED:
            self.check_bridge_passes(
                start, np.array([plane.x, plane.y, plane.altitude])
            )
            self.check_traffic(start)
        if self.race and plane.state is not FlightState.CRASHED:
            end = np.array([plane.x, plane.y, plane.altitude])
            passed = self.race.update(DT, start, end, moving=plane.speed > 2.0)
            if passed is not None:
                self.ring_passed(passed)
        if plane.state is FlightState.CRASHED:
            if not was_crashed:
                self.crashed()
            self.particles.burn(self.wrecks.latest + [0.0, 0.0, 3.0], DT)
        self.follow_aircraft()

    def check_object_collision(self) -> None:
        """Crash into a tree, house or lighthouse if the plane is inside one."""
        plane = self.aircraft
        near = self.world.objects_near(plane.x, plane.y, config.COLLISION_SEARCH)
        bottom = plane.altitude - plane.settings.gear_height
        match hit(near, plane.x, plane.y, bottom):
            case Kind.TREE:
                plane.crash("HIT A TREE")
            case Kind.HOUSE:
                plane.crash("HIT A HOUSE")
            case Kind.LIGHTHOUSE:
                plane.crash("HIT THE LIGHTHOUSE")
        for bridge in self.bridges_near(plane.x, plane.y, config.BRIDGE_SEARCH):
            if bridge.hit(plane.x, plane.y, plane.altitude):
                plane.crash("HIT A BRIDGE")
                return

    def bridges_near(self, x: float, y: float, radius: float) -> list[Bridge]:
        return self.world.generator.bridges.near(x, y, radius)

    def check_bridge_passes(self, start: np.ndarray, end: np.ndarray) -> None:
        """Flying under a bridge earns a cheer, and discovers it the first time."""
        for bridge in self.bridges_near(end[0], end[1], config.BRIDGE_SEARCH):
            if not bridge.passed_under(start, end):
                continue
            self.audio.ring_passed()
            self.show_toast(f"UNDER {bridge.name}!")
            self.events.emit(EventKind.BRIDGE, end, f"UNDER {bridge.name}!")
            self.award(self.tracker.flew_under_bridge(bridge.number))
            if bridge.number not in self.saved.bridges:
                self.saved.bridges.append(bridge.number)
                self.persist()

    def lighthouses_near(self, radius: float) -> list[Lighthouse]:
        cam = self.camera
        return self.world.generator.settlements.lighthouses(
            cam.x - radius, cam.y - radius, cam.x + radius, cam.y + radius
        )

    def discover_landmarks(self) -> None:
        """Flying close to a lighthouse for the first time discovers it."""
        plane = self.aircraft
        for light in self.lighthouses_near(config.DISCOVER_DISTANCE):
            near = (
                math.hypot(light.x - plane.x, light.y - plane.y)
                < config.DISCOVER_DISTANCE
            )
            if near and light.number not in self.saved.landmarks:
                self.saved.landmarks.append(light.number)
                self.persist()
                self.show_toast(f"DISCOVERED {light.name}")
                self.award(self.tracker.discovered_lighthouse(light.number))

    @property
    def plane_type(self) -> AircraftType:
        return by_name(self.profile.aircraft)

    @property
    def plane_views(self) -> dict[str, pyxel.Image]:
        return self.type_views[TYPES.index(self.plane_type)]

    def choose_aircraft(self, plane_type: AircraftType) -> None:
        """Fly ``plane_type`` from now on (also in later sessions)."""
        self.profile.aircraft = plane_type.name
        self.audio.set_engine(plane_type.engine_note)
        self.save_profile()

    def change_aircraft(self) -> None:
        """Pause menu: swap to the next type, only standing still on a runway."""
        plane = self.aircraft
        on_runway = self.on_runway(plane.x, plane.y)
        if plane.state is not FlightState.LANDED or plane.speed > 1.0 or not on_runway:
            self.show_toast("STOP ON A RUNWAY TO CHANGE AIRCRAFT")
            return
        new = TYPES[(TYPES.index(self.plane_type) + 1) % len(TYPES)]
        self.choose_aircraft(new)
        plane.settings = replace(new.settings, sea_level=plane.settings.sea_level)
        plane.fuel = 1.0
        self.end_race("TIME TRIAL ENDED")
        self.show_toast(f"{new.name}: {new.description}", 3.0)

    def on_runway(self, x: float, y: float) -> bool:
        return self.world.generator.airfields.runway_at(x, y) is not None

    # --- achievements -------------------------------------------------------

    def watch_flight(self) -> None:
        """Show the achievement detectors this frame of flight."""
        plane = self.aircraft
        if pyxel.frame_count % config.RIVER_CHECK_FRAMES == 0:
            valley = self.world.generator.river_valley(plane.x, plane.y)
            self.over_river = valley > config.RIVER_VALLEY
        ground = max(
            self.world.height_at(plane.x, plane.y),
            self.world.generator.settings.sea_level,
        )
        sample = Sample(
            position=np.array([plane.x, plane.y, plane.altitude]),
            forward=plane.forward.copy(),
            up=plane.up.copy(),
            speed=plane.speed,
            agl=plane.altitude - plane.settings.gear_height - ground,
            flying=plane.state is FlightState.FLYING,
            in_cloud=self.clouds.inside(plane.x, plane.y, plane.altitude),
            storm=self.weather.current.name == "STORM",
            over_river=self.over_river,
        )
        self.award(self.tracker.update(sample, DT))
        if self.tracker.dirty and pyxel.frame_count % config.PROGRESS_SAVE_FRAMES == 0:
            self.save_profile()

    def on_game_event(self, event) -> None:
        if self.tracker is not None:
            self.award(self.tracker.on_event(event))

    def award(self, keys: list[str]) -> None:
        """Announce newly earned achievements and save them."""
        if not keys:
            return
        if not self.awards:
            self.award_until = time.perf_counter() + config.AWARD_SECONDS
        self.awards += [BY_KEY[key].name for key in keys]
        self.audio.achievement()
        self.save_profile()

    def draw_award(self) -> None:
        """The current achievement in a banner, one after another."""
        if not self.awards:
            return
        now = time.perf_counter()
        if now > self.award_until:
            self.awards.pop(0)
            self.award_until = now + config.AWARD_SECONDS
            if not self.awards:
                return
        shaded_rect(0, 40, W, 11)
        centered_text(42, f"ACHIEVEMENT: {self.awards[0]}", pyxel.COLOR_YELLOW)

    def update_achievements_menu(self) -> None:
        if pressed(Action.BACK) or pressed(Action.PAUSE):
            self.achievements_menu = False
            self.menu_index = next(
                i
                for i, item in enumerate(self.pause_items())
                if item.startswith("ACHIEVE")
            )
            return
        if pressed(Action.UP):
            self.menu_index = (self.menu_index - 1) % len(ACHIEVEMENTS)
            self.audio.menu_move()
        if pressed(Action.DOWN):
            self.menu_index = (self.menu_index + 1) % len(ACHIEVEMENTS)
            self.audio.menu_move()

    def draw_achievements_menu(self) -> None:
        """All achievements: earned ones bright, the rest grey, hidden ones ???."""
        rows = ACHIEVEMENT_ROWS
        first = max(0, min(self.menu_index - rows // 2, len(ACHIEVEMENTS) - rows))
        width, height = 200, 46 + rows * 8
        x, y = (W - width) // 2, (H - height) // 2
        shaded_rect(0, 0, W, H)
        pyxel.rect(x, y, width, height, pyxel.COLOR_BLACK)
        pyxel.rectb(x, y, width, height, HUD)
        earned = self.profile.earned
        centered_text(
            y + 4, f"ACHIEVEMENTS {len(earned)}/{len(ACHIEVEMENTS)}", pyxel.COLOR_YELLOW
        )
        for row, item in enumerate(ACHIEVEMENTS[first : first + rows]):
            i = first + row
            got = item.key in earned
            name = item.name if got or not item.hidden else "???"
            mark = "*" if got else " "
            color = pyxel.COLOR_YELLOW if got else pyxel.COLOR_GRAY
            if i == self.menu_index:
                color = TEXT
                name = f"> {name}"
            pyxel.text(x + 8, y + 14 + row * 8, f"{mark} {name}", color)
        item = ACHIEVEMENTS[self.menu_index]
        got = item.key in earned
        about = "???" if item.hidden and not got else item.description.upper()
        if got:
            about += f"  ({earned[item.key].get('date', '')})"
        centered_text(y + height - 22, about, TEXT)
        centered_text(y + height - 10, "UP/DOWN SCROLL   ESC BACK", HUD)

    # --- traffic ------------------------------------------------------------

    def update_traffic(self) -> None:
        """Where the other aircraft near the camera are now (none in a race)."""
        if not self.profile.traffic or self.race is not None:
            self.flights = []
            return
        cam = self.camera
        self.flights_before = {f.ident: f.position for f in self.flights}
        self.flights = self.traffic.flights(cam.x, cam.y, self.time)

    def check_traffic(self, start: np.ndarray) -> None:
        """Collide with other aircraft, and note close shaves. ``start`` is where
        the plane was a frame ago."""
        plane = self.aircraft
        if plane.state is not FlightState.FLYING:
            return
        here = np.array([plane.x, plane.y, plane.altitude])
        for flight in self.flights:
            other = flight.position
            before = self.flights_before.get(flight.ident, other)
            gap = closest_approach(here, other, start, before)
            span, other_span = self.plane_type.span, TYPES[flight.kind].span
            if midair(gap, span, other_span, config.MIDAIR_DISTANCE):
                plane.crash("MIDAIR COLLISION")
                return
            # A near miss is only one once the two are drawing apart again.
            closest = min(self.closest.get(flight.ident, math.inf), gap)
            self.closest[flight.ident] = closest
            passed = gap > closest + 1.0
            near = closest < config.NEAR_MISS_DISTANCE
            if near and passed and flight.ident not in self.near_misses:
                self.near_misses.add(flight.ident)
                self.events.emit(EventKind.NEAR_MISS, here, "NEAR MISS!")
                self.show_toast("NEAR MISS!")

    def draw_traffic(self) -> None:
        """Other aircraft, with navigation lights at dusk and night."""
        lights = palette.LIGHTINGS[self.lighting] is not palette.DAY
        for flight in self.flights:
            plane_type = TYPES[flight.kind]
            draw_plane(
                flight.kind,
                self.type_views[flight.kind],
                flight.position,
                flight.forward,
                flight.up,
                self.camera,
                self.renderer,
                propeller=self.time * PROPELLER_SPEED,
            )
            if lights:
                self.draw_nav_lights(flight, plane_type.span)

    def draw_nav_lights(self, flight: Flight, span: float) -> None:
        """Red on the left wing tip, green on the right, a white flash at the tail."""
        right = np.cross(flight.up, flight.forward)
        tips = [
            (flight.position - right * span / 2, pyxel.COLOR_RED),
            (flight.position + right * span / 2, pyxel.COLOR_LIME),
        ]
        if self.time % 1.0 < 0.15:
            tips.append(
                (flight.position - flight.forward * span / 2, pyxel.COLOR_WHITE)
            )
        points = np.array([p for p, _ in tips])
        sx, sy, depth = project_points(self.camera, points, self.renderer.focal, W, H)
        seen = (depth > 1.0) & in_front_of_terrain(
            self.renderer, self.camera, points, sx, sy
        )
        for (_, color), x, y, ok in zip(tips, sx, sy, seen):
            if ok:
                pyxel.pset(x, y, color)

    def save_profile(self) -> None:
        try:
            self.store.save_profile(self.profile)
        except OSError:
            self.show_toast("COULD NOT SAVE")

    def toggle_fuel(self) -> None:
        """Pause menu: fuel on (it runs out) or off (it never does)."""
        self.profile.fuel = not self.profile.fuel
        self.aircraft.uses_fuel = self.profile.fuel
        self.aircraft.fuel = 1.0
        self.save_profile()

    def refuel(self) -> None:
        """Fill the tank while standing still on a runway."""
        plane = self.aircraft
        standing = (
            plane.uses_fuel
            and plane.state is FlightState.LANDED
            and plane.speed < config.REFUEL_MAX_SPEED
            and plane.fuel < 1.0
            and self.world.generator.airfields.runway_at(plane.x, plane.y) is not None
        )
        if not standing:
            self.refuelling = False
            return
        self.refuelling = True
        plane.refuel(DT)
        if plane.fuel >= 1.0:
            self.refuelling = False
            self.audio.tank_full()
            self.show_toast("TANK FULL")
        else:
            self.show_toast(f"REFUELLING {plane.fuel * 100:.0f}%", 0.5)

    def touched_down(self) -> None:
        """Grade a landing if it was on a runway, and remember the airfield."""
        plane = self.aircraft
        field = self.world.generator.airfields.runway_at(plane.x, plane.y)
        if field is None:
            if plane.on_water:
                report = water_landing_report(
                    plane.touchdown_sink, plane.touchdown_engine_out
                )
                self.show_toast(report)
                self.events.emit(
                    EventKind.LANDING, (plane.x, plane.y, plane.altitude), report
                )
            return
        report = landing_report(
            field, plane.x, plane.y, plane.touchdown_sink, plane.touchdown_engine_out
        )
        if field.number not in self.saved.visited:
            self.saved.visited.append(field.number)
            self.persist()
            report = "NEW " + report
        self.show_toast(report)
        self.events.emit(EventKind.LANDING, (plane.x, plane.y, plane.altitude), report)
        ax, ay = field.across
        crosswind = abs(self.wind[0] * ax + self.wind[1] * ay)
        self.award(self.tracker.landed_at(field.number, crosswind))

    def crashed(self) -> None:
        """The moment of impact: leave a wreck on the ground and blow up."""
        plane = self.aircraft
        ground = max(self.world.height_at(plane.x, plane.y), plane.settings.sea_level)
        site = np.array([plane.x, plane.y, ground])
        self.wrecks.add(site, TYPES.index(self.plane_type))
        self.saved.wrecks = self.wrecks.to_list()
        self.persist()
        self.particles.explode(site + [0.0, 0.0, 2.0])
        self.events.emit(EventKind.CRASH, site, plane.crash_reason)
        if self.race:
            self.race = None  # the crash screen says what happened

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
                    self.wrecks.positions,
                    self.saved.places,
                    self.map_airfields(),
                    set(self.saved.visited),
                    self.race.course.rings[self.race.next_ring :] if self.race else (),
                    *self.map_settlements(),
                    set(self.saved.landmarks),
                    plane.range_estimate() if plane.uses_fuel else None,
                    self.map_bridges(),
                    set(self.saved.bridges),
                )
            case Mode.PHOTO:
                self.draw_photo()
            case Mode.REPLAY:
                self.draw_replay()
            case Mode.FLYING | Mode.PAUSED:
                self.draw_view()
                if self.mode is Mode.PAUSED:
                    if self.places_menu:
                        self.draw_places_menu()
                    elif self.achievements_menu:
                        self.draw_achievements_menu()
                    else:
                        self.draw_pause_menu()
        if self.toast and time.perf_counter() < self.toast_until:
            # Below the time-trial panel while racing, under the compass otherwise.
            racing = self.race is not None and self.mode is Mode.FLYING
            if self.mode is Mode.PAUSED:
                y = H - 12  # clear of the pause menu
            else:
                y = 52 if racing else 32
            centered_text(y, self.toast, pyxel.COLOR_YELLOW)
        if self.mode in (Mode.FLYING, Mode.PAUSED, Mode.PHOTO):
            self.draw_award()
        if self.show_debug and self.world:
            self.draw_debug()

    def draw_view(self) -> None:
        start = time.perf_counter()
        self.renderer.render(
            screen_array(), self.world.levels, self.camera, self.time, self.clouds
        )
        self.render_time.add(time.perf_counter() - start)
        draw_particles(self.particles, self.camera, self.renderer)
        hangars = self.draw_airfield_props()
        bridges = self.bridges_near(
            self.camera.x, self.camera.y, config.BRIDGE_DRAW_DISTANCE
        )
        # Trees and houses beyond the nearest bridge or hangar model are drawn
        # before them, nearer ones after.
        split = min(
            self.nearest_bridge_distance(bridges),
            min((d for d, _, _ in hangars), default=math.inf),
        )
        self.draw_objects(beyond=split)
        if bridges:
            draw_bridges(
                bridges, self.camera, self.renderer, config.BRIDGE_DRAW_DISTANCE
            )
        for _, position, facing in sorted(hangars, key=lambda h: -h[0]):
            draw_hangar(self.hangar_model, position, facing, self.camera, self.renderer)
        if bridges or hangars:
            self.draw_objects(within=split)
        nearby, kinds = self.wrecks.near_with_kinds(
            self.camera.x, self.camera.y, config.WRECK_DRAW_DISTANCE
        )
        if self.mode is Mode.REPLAY:
            keep = self.not_yet_crashed(nearby)
            nearby, kinds = nearby[keep], kinds[keep]
        away = np.hypot(nearby[:, 0] - self.camera.x, nearby[:, 1] - self.camera.y)
        big = np.array(
            [
                wreck_size(k, d, self.renderer) >= LOD_PIXELS
                for k, d in zip(kinds, away)
            ],
            dtype=bool,
        )
        draw_billboards(
            [views["wreck"] for views in self.type_views],
            TRANSPARENT,
            nearby[~big],
            self.camera,
            self.renderer,
            [
                WRECK_FULL_SIZE_DISTANCE * self.zoom * t.span / TRAINER.span
                for t in TYPES
            ],
            WRECK_TILT,
            variants=kinds[~big],
        )
        for i in np.flatnonzero(big)[np.argsort(-away[big])]:
            draw_wreck(int(kinds[i]), nearby[i], self.camera, self.renderer)
        if self.race:
            draw_rings(self.race, self.camera, self.renderer)
            self.draw_record_ghost()
        chase = self.chase_view and self.aircraft.state is not FlightState.CRASHED
        if chase and not self.ghost and self.mode is Mode.FLYING:
            self.draw_plane()
        if self.mode is not Mode.REPLAY:  # traffic is at the live time
            self.draw_traffic()
        self.draw_rain()
        if self.mode in (Mode.PHOTO, Mode.REPLAY):
            return
        match self.hud_mode:
            case HudMode.NONE:
                return
            case HudMode.MINIMAP:
                self.hud.minimap(
                    self.aircraft if not self.ghost else self.camera,
                    self.world,
                    [flight.position for flight in self.flights],
                )
                return
        if self.ghost:
            centered_text(H - 9, "GHOST CAMERA - G TO FLY FROM HERE", TEXT)
            return
        self.hud.draw(
            self.aircraft,
            self.camera,
            self.world,
            cockpit=not chase,
            target=None if self.race else self.target_field,
            wind=self.wind,
            turbulence=self.turbulence,
            traffic=[flight.position for flight in self.flights],
        )
        if self.race:
            ring = self.race.course.rings[self.race.next_ring]
            draw_ring_arrow(ring, self.camera, self.renderer)
            self.hud.race_panel(self.race, self.race_flash)

    def draw_photo(self) -> None:
        self.draw_view()
        plane = self.aircraft
        if (
            self.photo_plane
            and not self.ghost
            and plane.state is not FlightState.CRASHED
        ):
            self.draw_model(plane, self.time)
        if self.photo_frame:
            bar = H // 9
            pyxel.rect(0, 0, W, bar, pyxel.COLOR_BLACK)
            pyxel.rect(0, H - bar, W, bar, pyxel.COLOR_BLACK)
        if self.photo_requested:
            self.take_photo()
        if time.perf_counter() < self.photo_flash_until:
            pyxel.dither(0.5)
            pyxel.rect(0, 0, W, H, pyxel.COLOR_WHITE)
            pyxel.dither(1.0)
        if self.photo_help:
            self.draw_photo_help()

    def draw_photo_help(self) -> None:
        orbit = self.photo.mode is FreeMode.ORBIT
        fov = config.PHOTO_FOVS[self.photo_fov]
        shadow_text(4, 4, f"PHOTO  {self.photo.mode.value}  FOV {fov:.0f}", HUD)
        lines = [
            "ARROWS CIRCLE  W/S CLOSER/FURTHER  TAB FLY"
            if orbit
            else "ARROWS LOOK  W/S/A/D MOVE  Q/E DOWN/UP  TAB ORBIT",
            "Z/X ZOOM  T TIME  V WEATHER  F FRAME  O PLANE",
            "ENTER SAVE  L PLACE  H HELP  F2 BACK",
        ]
        for i, line in enumerate(lines):
            shadow_text(4, H - 8 * (len(lines) - i), line, TEXT)

    def not_yet_crashed(self, wrecks: np.ndarray) -> np.ndarray:
        """Which wrecks to show in a replay: not the one of the crash still to come."""
        keep = np.ones(len(wrecks), dtype=bool)
        crash = self.playback.recording.crash()
        if crash is None or crash[0] <= self.playback.time or not len(wrecks):
            return keep
        site = crash[1].position
        return np.hypot(wrecks[:, 0] - site[0], wrecks[:, 1] - site[1]) > 1.0

    def draw_rain(self) -> None:
        """Rain falls below the clouds; it thins out climbing up into them."""
        cam = self.camera
        above = cam.altitude - self.clouds.altitude
        amount = self.weather.current.rain * min(1.0, max(0.0, 1.0 - above / 40.0))
        self.rain.draw(amount, cam, self.renderer)

    def draw_record_ghost(self) -> None:
        """The record run, replayed as a see-through plane flying alongside."""
        race = self.race
        if race.phase is not Phase.RUNNING or not self.record_ghost:
            return
        where = ghost_position(self.record_ghost, race.elapsed, race.ghost_interval)
        if where is None:
            return
        ahead = ghost_position(
            self.record_ghost, race.elapsed + 0.3, race.ghost_interval
        )
        forward = (ahead if ahead is not None else where) - where
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, -1.0, 0.0])
        forward = forward / np.linalg.norm(forward)
        side = np.cross(forward, [0.0, 0.0, 1.0])
        up = np.cross(side / max(np.linalg.norm(side), 1e-9), forward)
        pyxel.dither(0.5)
        draw_plane(
            TYPES.index(self.plane_type),
            self.plane_views,
            where,
            forward,
            up,
            self.camera,
            self.renderer,
            propeller=race.elapsed * PROPELLER_SPEED,
        )
        pyxel.dither(1.0)

    def nearest_bridge_distance(self, bridges: list[Bridge]) -> float:
        """Horizontal distance from the camera to the nearest point of any deck."""
        cam = self.camera
        nearest = math.inf
        for bridge in bridges:
            along, across = bridge.local(cam.x, cam.y)
            along = max(abs(float(along)) - bridge.half_span, 0.0)
            across = max(abs(float(across)) - bridge.width / 2, 0.0)
            nearest = min(nearest, math.hypot(along, across))
        return nearest

    def draw_objects(self, beyond: float = 0.0, within: float = math.inf) -> None:
        """Trees and houses near the camera, and lighthouses further out: those
        more than ``beyond`` and at most ``within`` away (horizontally), so
        objects on both sides of a bridge can be drawn before and after it."""
        cam = self.camera
        reach = min(
            config.OBJECT_DRAW_DISTANCE * self.zoom, self.world.levels[0].safe_radius
        )
        near = self.world.objects_near(cam.x, cam.y, reach)
        near = near.select(near.kind != Kind.LIGHTHOUSE)  # drawn below, further out
        lights = self.lighthouses_near(config.LIGHTHOUSE_DRAW_DISTANCE)
        if lights:
            towers = Objects(
                np.array([[t.x, t.y, t.z] for t in lights], dtype=np.float32),
                np.full(len(lights), Kind.LIGHTHOUSE, np.uint8),
                np.zeros(len(lights), np.uint8),
                np.ones(len(lights), np.float32),
            )
            near = Objects.concat([near, towers])
        away = np.hypot(near.position[:, 0] - cam.x, near.position[:, 1] - cam.y)
        near = near.select((away > beyond) & (away <= within))
        base = np.array([0, 3, 7])  # first sprite of each Kind
        self.objects_drawn = draw_billboards(
            self.object_sprites,
            TRANSPARENT,
            near.position,
            cam,
            self.renderer,
            [f * self.renderer.focal for f in self.object_full_size_per_focal],
            variants=base[near.kind] + near.variant,
            scales=near.scale,
            limit=config.OBJECT_LIMIT,
        )
        if within == math.inf and palette.LIGHTINGS[self.lighting] is not palette.DAY:
            self.draw_lighthouse_lamps(lights)

    def draw_lighthouse_lamps(self, lights: list[Lighthouse]) -> None:
        """Blinking lamps, visible from far away at dusk and night."""
        if not lights or self.time % config.LAMP_PERIOD > config.LAMP_ON:
            return
        lamps = np.array(
            [[t.x, t.y, t.z + HEIGHT[Kind.LIGHTHOUSE] * 0.93] for t in lights]
        )
        sx, sy, depth = project_points(self.camera, lamps, self.renderer.focal, W, H)
        seen = (depth > 1.0) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        seen &= in_front_of_terrain(self.renderer, self.camera, lamps, sx, sy)
        for x, y in zip(sx[seen], sy[seen]):
            pyxel.circ(x, y, 1, pyxel.COLOR_YELLOW)
            pyxel.pset(x, y, pyxel.COLOR_WHITE)

    def draw_airfield_props(self) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Hangars and windsocks at the airfields around the camera, as pictures.
        Hangars close enough to look flat are left out and returned instead, as
        (distance to their nearest wall, position, way the door faces), to be
        drawn as models."""
        cam = self.camera
        r = config.PROP_DRAW_DISTANCE
        fields = self.world.generator.airfields.overlapping(
            cam.x - r, cam.y - r, cam.x + r, cam.y + r
        )
        models = []
        for field in fields:
            for kind, positions in self.props_of(field).items():
                if kind == "hangar":
                    positions = self.split_hangars(field, positions, models)
                    if not len(positions):
                        continue
                sprite, width = self.prop_sprites[kind]
                variants = None
                if kind == "windsock":
                    # Mirror the sock when the wind blows to the camera's left.
                    right = (math.cos(cam.heading), math.sin(cam.heading))
                    left = self.wind[0] * right[0] + self.wind[1] * right[1] < 0
                    variants = np.full(len(positions), int(left))
                first = sprite if isinstance(sprite, pyxel.Image) else sprite[0]
                full = full_size_distance(first, width, self.renderer.focal)
                draw_billboards(
                    sprite,
                    TRANSPARENT,
                    positions,
                    cam,
                    self.renderer,
                    full,
                    variants=variants,
                )
        return models

    def split_hangars(
        self, field: Airfield, positions: np.ndarray, models: list
    ) -> np.ndarray:
        """The hangars to draw as pictures; the near ones go into ``models``."""
        cam = self.camera
        f, _, _ = cam.basis()
        far = []
        for position in positions:
            depth = float((position - np.array(cam.position)) @ f)
            wide = (
                depth <= 0
                or config.HANGAR_WIDTH * self.renderer.focal / depth >= LOD_PIXELS
            )
            if not wide:
                far.append(position)
                continue
            _, side = field.local(position[0], position[1])
            facing = door_facing(field.across, float(side))
            away = math.hypot(position[0] - cam.x, position[1] - cam.y)
            models.append((max(away - config.HANGAR_WIDTH / 2, 0.0), position, facing))
        return np.array(far).reshape(-1, 3)

    def props_of(self, field: Airfield) -> dict[str, np.ndarray]:
        """World positions of a field's buildings, on the ground (computed once)."""
        if field.number not in self.field_props:
            props: dict[str, list] = {}
            elevation = self.world.generator.elevation
            for kind, x, y in field.props():
                z = float(elevation(np.array(x), np.array(y)))
                props.setdefault(kind, []).append([x, y, z])
            self.field_props[field.number] = {k: np.array(v) for k, v in props.items()}
        return self.field_props[field.number]

    def map_settlements(self) -> tuple[list, list]:
        """Villages and lighthouses on the map, when zoomed in enough to be cheap."""
        step = ZOOMS[self.map_zoom]
        if step > config.MAP_SETTLEMENT_ZOOM:
            return [], []
        plane, s = self.aircraft, self.world.generator.settlements
        rx, ry = W / 2 * step, H / 2 * step
        box = (plane.x - rx, plane.y - ry, plane.x + rx, plane.y + ry)
        return s.villages(*box), s.lighthouses(*box)

    def map_bridges(self) -> list[Bridge]:
        """Bridges on the map, at the same zoom levels as villages."""
        step = ZOOMS[self.map_zoom]
        if step > config.MAP_SETTLEMENT_ZOOM:
            return []
        plane = self.aircraft
        rx, ry = W / 2 * step, H / 2 * step
        return self.world.generator.bridges.overlapping(
            plane.x - rx, plane.y - ry, plane.x + rx, plane.y + ry
        )

    def map_airfields(self) -> list[Airfield]:
        """Airfields in and around the area the map shows."""
        plane, step = self.aircraft, ZOOMS[self.map_zoom]
        rx, ry = W / 2 * step, H / 2 * step
        return self.world.generator.airfields.overlapping(
            plane.x - rx, plane.y - ry, plane.x + rx, plane.y + ry
        )

    def draw_plane(self) -> None:
        """The aircraft as seen from the chase camera."""
        self.draw_model(self.aircraft, self.time)

    def draw_model(self, plane: Aircraft, clock: float) -> None:
        """The player's aircraft (or its replay), propeller turning with the
        throttle at time ``clock``."""
        draw_plane(
            TYPES.index(self.plane_type),
            self.plane_views,
            np.array([plane.x, plane.y, plane.altitude]),
            plane.forward,
            plane.up,
            self.camera,
            self.renderer,
            propeller=clock * PROPELLER_SPEED * (0.3 + plane.throttle),
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
        self.title_line(H - 30, "SEED", f"SEED {self.seed}")
        if self.title_summary is None or self.title_summary[0] != self.seed:
            self.title_summary = (self.seed, self.store.summary(self.seed))
        wrecks, places = self.title_summary[1]
        if wrecks or places:
            shaded_rect(0, 50, W, 10)
            saved = f"{plural(wrecks, 'WRECK')}, {plural(places, 'PLACE')}"
            centered_text(52, f"SAVED HERE: {saved}", TEXT)
        centered_text(H - 21, "UP/DOWN CHOOSE  LEFT/RIGHT CHANGE  R RANDOM", HUD)
        start = "ON A RUNWAY" if self.start_on_runway else "IN THE AIR"
        shaded_rect(0, 62, W, 28)
        self.title_line(64, "START", f"START {start}")
        plane = self.plane_type
        self.title_line(74, "AIRCRAFT", f"AIRCRAFT {plane.name}")
        centered_text(83, plane.description, pyxel.COLOR_GRAY)
        if pyxel.frame_count // 15 % 2 == 0:
            centered_text(H - 11, "PRESS ENTER OR (A) TO FLY", TEXT)

    def title_line(self, y: int, row: str, text: str) -> None:
        """One choice on the title screen; the selected one has arrows."""
        if TITLE_ROWS[self.title_row] == row:
            centered_text(y, f"<  {text}  >", pyxel.COLOR_YELLOW)
        else:
            centered_text(y, text, pyxel.COLOR_LIGHT_BLUE)

    def draw_loading(self) -> None:
        pyxel.cls(pyxel.COLOR_NAVY)
        centered_text(H // 2 - 22, f"SEED {self.seed}", pyxel.COLOR_YELLOW)
        centered_text(H // 2 - 10, "GENERATING TERRAIN", TEXT)
        bar_w = 120
        x = (W - bar_w) // 2
        pyxel.rectb(x, H // 2, bar_w, 6, TEXT)
        pyxel.rect(x + 1, H // 2 + 1, int((bar_w - 2) * self.progress), 4, HUD)

    def draw_pause_menu(self) -> None:
        items = self.pause_items()
        # Rows close up when the menu is long, so it stays clear of the toast.
        spacing = min(10, (H - 40) // len(items))
        width, height = 100, 22 + len(items) * spacing
        x, y = (W - width) // 2, (H - 14 - height) // 2
        shaded_rect(0, 0, W, H)
        pyxel.rect(x, y, width, height, pyxel.COLOR_BLACK)
        pyxel.rectb(x, y, width, height, HUD)
        centered_text(y + 5, "PAUSED", pyxel.COLOR_YELLOW)
        for i, item in enumerate(items):
            selected = i == self.menu_index
            label = f"> {item} <" if selected else item
            centered_text(
                y + 17 + i * spacing, label, TEXT if selected else pyxel.COLOR_GRAY
            )

    def draw_places_menu(self) -> None:
        items = self.places_menu_items()
        first = max(0, min(self.menu_index - PLACE_ROWS // 2, len(items) - PLACE_ROWS))
        shown = items[first : first + PLACE_ROWS]
        width, height = 150, 34 + len(shown) * 10
        x, y = (W - width) // 2, (H - height) // 2
        shaded_rect(0, 0, W, H)
        pyxel.rect(x, y, width, height, pyxel.COLOR_BLACK)
        pyxel.rectb(x, y, width, height, HUD)
        centered_text(y + 5, "PLACES", pyxel.COLOR_YELLOW)
        for row, item in enumerate(shown):
            i = first + row
            selected = i == self.menu_index
            if isinstance(item, Place):
                away = math.hypot(item.x - self.camera.x, item.y - self.camera.y)
                label = f"{item.name}  {away:5.0f} AWAY"
            else:
                label = item
            label = f"> {label} <" if selected else label
            color = TEXT if selected else pyxel.COLOR_GRAY
            centered_text(y + 17 + row * 10, label, color)
        hint = "ENTER GO   DEL/X DELETE   ESC BACK"
        centered_text(y + height - 10, hint, HUD)

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
            f"OBJECTS {self.objects_drawn}",
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
