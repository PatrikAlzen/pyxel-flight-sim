"""Head-up display: instruments, minimap and warnings drawn over the 3D view.

Layout (256x144):

    compass band and heading            top centre
    speed tape                          left
    altitude tape, vertical speed       right
    pitch ladder and aircraft symbol    centre
    throttle                            bottom left
    minimap (heading up)                bottom right
"""

import math
import time
from enum import Enum

import numpy as np
import pyxel

from flightsim.render.framebuffer import screen_array
from flightsim.sim.aircraft import Aircraft, Alert, FlightState
from flightsim.sim.camera import Camera
from flightsim.sim.race import Phase, Race, format_time
from flightsim.terrain.airfields import Airfield
from flightsim.terrain.world import TerrainWorld

HUD = pyxel.COLOR_LIME
TEXT = pyxel.COLOR_WHITE
WARN = pyxel.COLOR_RED
CAUTION = pyxel.COLOR_YELLOW

# Speed and altitude tapes; the minimap sits below the altitude tape.
TAPE_TOP = 24
TAPE_HEIGHT = 56
TAPE_WIDTH = 26
TAPE_MID = TAPE_TOP + TAPE_HEIGHT // 2

MINIMAP_SIZE = 40
MINIMAP_SCALE = 16.0  # world units per minimap pixel
TURBULENCE_WARNING = 0.45  # turbulence (0..1) above which the HUD warns
FUEL_LOW = 0.15  # share of a full tank below which the HUD warns


def shadow_text(x: float, y: float, text: str, color: int) -> None:
    pyxel.text(x + 1, y + 1, text, pyxel.COLOR_BLACK)
    pyxel.text(x, y, text, color)


def centered_text(y: float, text: str, color: int) -> None:
    shadow_text((pyxel.width - len(text) * 4) // 2, y, text, color)


def shaded_rect(x: float, y: float, w: float, h: float) -> None:
    """Half-transparent dark backing so instruments stay readable over terrain."""
    pyxel.dither(0.5)
    pyxel.rect(x, y, w, h, pyxel.COLOR_BLACK)
    pyxel.dither(1.0)


class HudMode(Enum):
    """What is drawn over the view; H cycles through them in this order."""

    FULL = "HUD ON"
    MINIMAP = "MINIMAP ONLY"
    NONE = "HUD OFF"

    def next(self) -> HudMode:
        modes = list(HudMode)
        return modes[(modes.index(self) + 1) % len(modes)]


class Hud:
    def __init__(self, focal: float, respawn_delay: float) -> None:
        # Pixels per unit of tan(angle); must match the renderer so the pitch
        # ladder lines up with the real horizon.
        self.focal = focal
        self.respawn_delay = respawn_delay

    def draw(
        self,
        plane: Aircraft,
        camera: Camera,
        world: TerrainWorld,
        cockpit: bool = True,
        target: Airfield | None = None,
        wind: tuple[float, float] = (0.0, 0.0),
        turbulence: float = 0.0,
    ) -> None:
        """Draw all instruments. The pitch ladder and aircraft symbol only make
        sense looking out of the cockpit, so the chase view leaves them out.
        ``target`` is shown on the compass with its distance; ``wind`` is the
        wind velocity and ``turbulence`` 0..1 how bumpy the air is."""
        ground = world.height_at(plane.x, plane.y)
        agl = plane.altitude - plane.settings.gear_height - ground
        self.minimap(plane, world)  # writes pixels directly, so it goes first
        if plane.state is FlightState.CRASHED:
            self.crash_message(plane)
            return
        if cockpit:
            self.pitch_ladder(camera)
            self.aircraft_symbol()
        self.compass(plane.heading)
        if target is not None:
            self.waypoint(plane, target)
        self.speed_tape(plane)
        self.altitude_tape(plane, agl)
        self.throttle(plane.throttle)
        if plane.uses_fuel:
            self.fuel(plane)
        self.wind(plane, wind)
        self.messages(plane, agl)
        if turbulence > TURBULENCE_WARNING and pyxel.frame_count // 8 % 2 == 0:
            centered_text(pyxel.height // 2 + 22, "TURBULENCE", CAUTION)

    def pitch_ladder(self, camera: Camera) -> None:
        """Rungs every 10 degrees of elevation, straight ahead along the heading.

        Each rung is the screen projection of a real direction, so the ladder
        lines up with the horizon in any attitude, upside down included.
        """
        w, h = pyxel.width, pyxel.height
        _, r, u = camera.basis()
        sh, ch = math.sin(camera.heading), math.cos(camera.heading)
        # Rungs run parallel to the horizon: along the level right-hand direction.
        level_right = np.array([ch, sh, 0.0])
        rx, ry = float(level_right @ r), -float(level_right @ u)
        length = math.hypot(rx, ry) or 1.0
        rx, ry = rx / length, ry / length
        pyxel.clip(w / 2 - 64, 22, 128, h - 50)
        for degrees in range(-90, 91, 10):
            e = math.radians(degrees)
            ahead = np.array([sh * math.cos(e), -ch * math.cos(e), math.sin(e)])
            spot = camera.project_direction(ahead, self.focal, w, h)
            if spot is None:
                continue
            cx, cy, _ = spot
            inner, outer = (14, 64) if degrees == 0 else (14, 34)
            for side in (-1, 1):
                x1, y1 = cx + side * inner * rx, cy + side * inner * ry
                x2, y2 = cx + side * outer * rx, cy + side * outer * ry
                if degrees < 0:  # dashed below the horizon
                    for t in (0.0, 0.4, 0.8):
                        a, b = t, min(t + 0.25, 1.0)
                        pyxel.line(
                            x1 + (x2 - x1) * a,
                            y1 + (y2 - y1) * a,
                            x1 + (x2 - x1) * b,
                            y1 + (y2 - y1) * b,
                            HUD,
                        )
                else:
                    pyxel.line(x1, y1, x2, y2, HUD)
                if degrees:
                    label = str(abs(degrees))
                    lx = x2 + side * 3 - (len(label) * 4 if side < 0 else 0)
                    pyxel.text(lx, y2 - 2, label, HUD)
        pyxel.clip()

    def aircraft_symbol(self) -> None:
        cx, cy = pyxel.width // 2, pyxel.height // 2
        pyxel.line(cx - 10, cy, cx - 4, cy, TEXT)
        pyxel.line(cx - 4, cy, cx - 2, cy + 2, TEXT)
        pyxel.line(cx - 2, cy + 2, cx, cy, TEXT)
        pyxel.line(cx, cy, cx + 2, cy + 2, TEXT)
        pyxel.line(cx + 2, cy + 2, cx + 4, cy, TEXT)
        pyxel.line(cx + 4, cy, cx + 10, cy, TEXT)

    def compass(self, heading: float) -> None:
        cx, half = pyxel.width // 2, 60
        top = 2
        degrees = math.degrees(heading) % 360
        pyxel.rect(cx - half, top, half * 2, 11, pyxel.COLOR_BLACK)
        pyxel.clip(cx - half, top, half * 2, 11)
        for d in range(int(degrees) - half - 10, int(degrees) + half + 10):
            if d % 10:
                continue
            x = cx + (d - degrees)
            mark = d % 360
            if mark % 30 == 0:
                pyxel.line(x, top, x, top + 3, HUD)
                label = {0: "N", 90: "E", 180: "S", 270: "W"}.get(mark, str(mark // 10))
                if abs(x - cx) > half - len(label) * 2 - 1:
                    continue  # would be cut off at the edge of the band
                pyxel.text(
                    x - len(label) * 2 + 1,
                    top + 5,
                    label,
                    TEXT if mark % 90 == 0 else HUD,
                )
            else:
                pyxel.line(x, top, x, top + 1, HUD)
        pyxel.clip()
        label = f"{round(degrees) % 360:03d}"
        pyxel.rect(cx - 7, top + 12, 15, 8, pyxel.COLOR_BLACK)
        pyxel.rectb(cx - 7, top + 12, 15, 8, HUD)
        pyxel.text(cx - 5, top + 14, label, TEXT)
        pyxel.tri(cx - 2, top + 12, cx + 2, top + 12, cx, top + 10, HUD)

    def waypoint(self, plane: Aircraft, field: Airfield) -> None:
        """A marker on the compass band towards ``field``, plus its number and distance."""
        cx, half, top = pyxel.width // 2, 60, 2
        dx, dy = field.x - plane.x, field.y - plane.y
        bearing = math.degrees(math.atan2(dx, -dy))
        relative = (bearing - math.degrees(plane.heading) + 180) % 360 - 180
        x = cx + max(-half + 3, min(half - 3, relative))
        pyxel.tri(x - 3, top + 10, x + 3, top + 10, x, top + 6, CAUTION)
        if abs(relative) > half - 3:  # off the band: point the way to turn
            arrow = "<" if relative < 0 else ">"
            pyxel.text(x - 1 + (-6 if relative < 0 else 4), top + 3, arrow, CAUTION)
        distance = math.hypot(dx, dy)
        away = f"{distance / 1000:.1f}K" if distance >= 1000 else f"{distance:.0f}"
        shadow_text(cx + 12, top + 14, f"FLD {field.number} {away}", CAUTION)

    def _tape(
        self,
        x: int,
        value: float,
        scale: float,
        minor: int,
        label_every: int,
        ticks_right: bool,
        low_limit: float | None = None,
    ) -> None:
        """Vertical scrolling scale centred on ``value`` (``scale`` pixels per unit)."""
        h = pyxel.height
        top, height, width, mid = TAPE_TOP, TAPE_HEIGHT, TAPE_WIDTH, TAPE_MID
        pyxel.rect(x, top, width, height, pyxel.COLOR_BLACK)
        pyxel.clip(x, top, width, height)
        span = height / 2 / scale
        start = math.floor((value - span) / minor) * minor
        for mark in range(start, math.ceil(value + span) + minor, minor):
            y = mid - (mark - value) * scale
            major = mark % label_every == 0
            length = 5 if major else 2
            x0 = x + width - length if ticks_right else x
            pyxel.line(x0, y, x0 + length - 1, y, HUD)
            # Labels next to the value box would peek out from behind it.
            if major and mark >= 0 and abs(y - mid) > 7:
                tx = x + 1 if ticks_right else x + width - len(str(mark)) * 4
                pyxel.text(tx, y - 2, str(mark), HUD)
        if low_limit is not None:
            y = mid - (low_limit - value) * scale
            bar = x + width - 2 if ticks_right else x
            pyxel.rect(bar, y, 2, h, WARN)
        pyxel.clip()

        text = f"{value:.0f}"
        box_w = max(len(text), 3) * 4 + 4
        bx = x + (width - box_w) // 2
        pyxel.rect(bx, mid - 4, box_w, 9, pyxel.COLOR_BLACK)
        pyxel.rectb(bx, mid - 4, box_w, 9, TEXT)
        pyxel.text(bx + box_w - 2 - len(text) * 4, mid - 2, text, TEXT)

    def speed_tape(self, plane: Aircraft) -> None:
        x = 6
        self._tape(x, plane.speed, 1.2, 10, 20, True, plane.settings.stall_speed)
        shadow_text(x + 7, TAPE_TOP - 7, "SPD", HUD)

    def altitude_tape(self, plane: Aircraft, agl: float) -> None:
        x = pyxel.width - TAPE_WIDTH - 6
        self._tape(x, plane.altitude, 0.3, 25, 50, False)
        shadow_text(x + 7, TAPE_TOP - 7, "ALT", HUD)

        # Vertical speed: a bar growing up or down from the tape's centre line,
        # full scale at +-30 units per second.
        mid, half = TAPE_MID, TAPE_HEIGHT // 2 - 2
        vs = max(-30.0, min(30.0, plane.vertical_speed)) * half / 30
        bar_x = x + TAPE_WIDTH + 2
        pyxel.line(bar_x, mid - half, bar_x, mid + half, pyxel.COLOR_GRAY)
        color = (
            WARN if plane.vertical_speed < -plane.settings.max_touchdown_sink else HUD
        )
        pyxel.rect(bar_x - 1, min(mid, mid - vs), 3, abs(vs) + 1, color)
        below = TAPE_TOP + TAPE_HEIGHT + 3
        shadow_text(x - 2, below, f"VS{round(plane.vertical_speed):+4d}", HUD)
        if agl < 300:
            radar = CAUTION if agl < 40 else HUD
            shadow_text(x + 2, below + 7, f"R {max(agl, 0):3.0f}", radar)

    def throttle(self, throttle: float) -> None:
        x, bottom, height = 6, pyxel.height - 6, 28
        pyxel.rect(x, bottom - height, 6, height, pyxel.COLOR_BLACK)
        fill = round((height - 2) * throttle)
        pyxel.rect(x + 1, bottom - 1 - fill, 4, fill, HUD)
        pyxel.rectb(x, bottom - height, 6, height, HUD)
        shadow_text(x + 9, bottom - 6, f"THR {throttle * 100:3.0f}%", HUD)

    def fuel(self, plane: Aircraft) -> None:
        """Fuel gauge beside the throttle: tank level and time left at this
        throttle; a warning when low and when the engine has stopped."""
        x, bottom, height = 54, pyxel.height - 6, 28
        low = plane.fuel < FUEL_LOW
        color = WARN if not plane.engine_running else CAUTION if low else HUD
        pyxel.rect(x, bottom - height, 6, height, pyxel.COLOR_BLACK)
        fill = round((height - 2) * plane.fuel)
        pyxel.rect(x + 1, bottom - 1 - fill, 4, fill, color)
        pyxel.rectb(x, bottom - height, 6, height, color)
        left = plane.endurance()
        minutes, seconds = divmod(round(left), 60)
        shadow_text(x + 9, bottom - 12, f"FUEL {plane.fuel * 100:3.0f}%", color)
        shadow_text(x + 9, bottom - 6, f"{minutes}:{seconds:02d} LEFT", color)
        blink = pyxel.frame_count // 8 % 2 == 0
        if not plane.engine_running:
            centered_text(pyxel.height // 2 + 30, "OUT OF FUEL - GLIDE", WARN)
        elif low and blink and plane.state is FlightState.FLYING:
            centered_text(pyxel.height // 2 + 30, "FUEL LOW", CAUTION)

    def wind(self, plane: Aircraft, wind: tuple[float, float]) -> None:
        """Wind under the speed tape: an arrow pointing where it blows, relative
        to the nose (up = tailwind), and its speed."""
        speed = math.hypot(*wind)
        x, y = 10, TAPE_TOP + TAPE_HEIGHT + 10
        if speed < 0.5:
            shadow_text(x - 4, y - 2, "CALM", HUD)
            return
        towards = math.atan2(wind[0], -wind[1]) - plane.heading
        dx, dy = math.sin(towards), -math.cos(towards)
        tip_x, tip_y = x + dx * 4, y + dy * 4
        pyxel.line(x - dx * 4, y - dy * 4, tip_x, tip_y, HUD)
        side_x, side_y = -dy * 2, dx * 2
        pyxel.tri(
            tip_x + dx,
            tip_y + dy,
            tip_x - dx * 2 + side_x,
            tip_y - dy * 2 + side_y,
            tip_x - dx * 2 - side_x,
            tip_y - dy * 2 - side_y,
            HUD,
        )
        shadow_text(x + 7, y - 2, f"W{speed:3.0f}", HUD)

    def minimap(self, plane: Aircraft, world: TerrainWorld) -> None:
        """Terrain around the plane, rotated so the nose points up."""
        size = MINIMAP_SIZE
        x0, y0 = pyxel.width - size - 4, pyxel.height - size - 4
        offsets = (np.arange(size) - size / 2 + 0.5) * MINIMAP_SCALE
        right = offsets[None, :]  # screen x -> to the right of the nose
        ahead = -offsets[:, None]  # screen y up -> ahead
        fx, fy = math.sin(plane.heading), -math.cos(plane.heading)
        rx, ry = -fy, fx
        wx = plane.x + right * rx + ahead * fx
        wy = plane.y + right * ry + ahead * fy
        # Finest level that still covers the minimap's corners.
        reach = size / 2 * MINIMAP_SCALE * math.sqrt(2)
        level = next(lv for lv in world.levels if lv.safe_radius >= reach)
        iy, ix = level.indices(wx, wy)
        screen_array()[y0 : y0 + size, x0 : x0 + size] = level.colors[iy, ix]

        pyxel.rectb(x0 - 1, y0 - 1, size + 2, size + 2, HUD)
        cx, cy = x0 + size // 2, y0 + size // 2
        pyxel.tri(cx, cy - 4, cx - 3, cy + 3, cx + 3, cy + 3, TEXT)
        pyxel.pset(cx, cy + 2, pyxel.COLOR_BLACK)
        # North marker on the rim.
        angle = -plane.heading
        nx = cx + math.sin(angle) * (size / 2 - 4)
        ny = cy - math.cos(angle) * (size / 2 - 4)
        shadow_text(nx - 1, ny - 2, "N", CAUTION)

    def race_panel(self, race: Race, flash: tuple[float, float] | None) -> None:
        """Time-trial status under the compass: rings left, time, record and the
        gain or loss against the record at the last ring (``flash``: delta, until)."""
        y = 24
        best = format_time(race.record[-1]) if race.record else "NONE"
        if race.phase is Phase.READY:
            centered_text(y, "TIME TRIAL - THROTTLE UP TO START", CAUTION)
            centered_text(y + 8, f"{len(race.course.rings)} RINGS   BEST {best}", TEXT)
            return
        ring = min(race.next_ring + 1, len(race.course.rings))
        centered_text(
            y,
            f"RING {ring}/{len(race.course.rings)}   {format_time(race.elapsed)}",
            CAUTION,
        )
        centered_text(y + 8, f"BEST {best}", pyxel.COLOR_GRAY)
        if flash is not None and time.perf_counter() < flash[1]:
            delta = round(flash[0], 1) + 0.0  # + 0.0 turns -0.0 into 0.0
            color = pyxel.COLOR_LIME if delta < 0 else WARN if delta > 0 else TEXT
            centered_text(y + 16, f"{delta:+.1f}", color)

    def messages(self, plane: Aircraft, agl: float) -> None:
        blink = pyxel.frame_count // 8 % 2 == 0
        h = pyxel.height
        match plane.alert(agl):
            case Alert.STALL if blink:
                centered_text(h // 2 + 14, "STALL", WARN)
            case Alert.PULL_UP if blink:
                centered_text(h // 2 + 14, "PULL UP", CAUTION)
        if plane.state is FlightState.LANDED:
            hint = (
                "LANDED"
                if plane.speed > 1
                else "LANDED - THROTTLE UP, PULL BACK TO LIFT OFF"
            )
            centered_text(h // 2 + 14, hint, HUD)

    def crash_message(self, plane: Aircraft) -> None:
        w, h = pyxel.width, pyxel.height
        if plane.state_time < 0.25:
            pyxel.dither(0.5)
            pyxel.rect(0, 0, w, h, WARN)
            pyxel.dither(1.0)
        # Text at the top and bottom: the wreck is in the middle of the view.
        centered_text(8, "CRASHED", WARN)
        centered_text(16, plane.crash_reason, TEXT)
        countdown = max(0, math.ceil(self.respawn_delay - plane.state_time))
        centered_text(
            h - 12, f"RESPAWN IN {countdown} (R)   REPLAY (F3)", pyxel.COLOR_GRAY
        )
