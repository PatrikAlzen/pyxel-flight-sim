"""Engine, wind, warning and crash sounds using Pyxel's built-in synth.

Channel use:

    0  engine: a short looping rumble whose pitch follows the throttle
    1  wind: looping noise that gets louder with airspeed; a hiss in rain
    2  alerts: stall and pull-up tones, looping while the alert lasts
    3  one-shots: crash, menu blips

Pyxel cannot bend the pitch of a playing sound, so each loop exists in a few
pre-built variants and the channel switches between them when the level
changes. The loops are a tenth of a second long, so a switch is inaudible.

``SoundState.choose`` decides what should be playing and is free of Pyxel
calls, so it can be tested without an audio device.
"""

from dataclasses import dataclass, replace

import pyxel

from flightsim.sim.aircraft import Aircraft, Alert, FlightState

ENGINE_LEVELS = 8
WIND_LEVELS = 5  # including 0 = silent
RAIN_LEVELS = 4  # including 0 = dry

_NOTE_NAMES = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]

CH_ENGINE, CH_WIND, CH_ALERT, CH_EFFECT = 0, 1, 2, 3


def note_name(note: int) -> str:
    """Pyxel note string for note number 0-59 (33 = A2)."""
    return f"{_NOTE_NAMES[note % 12]}{note // 12}"


def make_sound(
    notes: str, tones: str, volumes: str, effects: str, speed: int
) -> pyxel.Sound:
    sound = pyxel.Sound()
    sound.set(notes, tones, volumes, effects, speed)
    return sound


@dataclass(frozen=True)
class SoundState:
    """What each looping channel should play: a level index, an alert, or None."""

    engine: int | None
    wind: int
    alert: Alert | None
    rain: int = 0  # replaces the wind on its channel while it rains

    @staticmethod
    def choose(plane: Aircraft, agl: float, rain: float = 0.0) -> SoundState:
        """``rain`` is the weather's rain, 0..1."""
        level = min(RAIN_LEVELS - 1, round(rain * (RAIN_LEVELS - 1)))
        if plane.state is FlightState.CRASHED:
            return SoundState(None, 0, None, level)
        engine = (
            round(plane.throttle * (ENGINE_LEVELS - 1))
            if plane.engine_running
            else None  # out of fuel: only the wind is left
        )
        # Wind starts at taxi speeds and is loudest near top speed.
        wind = max(0, min(WIND_LEVELS - 1, int((plane.speed - 20) / 30)))
        return SoundState(engine, wind, plane.alert(agl), level)


SILENT = SoundState(None, 0, None)


class Audio:
    def __init__(self) -> None:
        self.set_engine(6)
        self.wind = [None] + [
            self._wind_sound(level) for level in range(1, WIND_LEVELS)
        ]
        self.rain = [None] + [
            self._rain_sound(level) for level in range(1, RAIN_LEVELS)
        ]
        self.alerts = {
            Alert.STALL: make_sound("c3rc3r", "s", "5", "n", 7),
            Alert.PULL_UP: make_sound("a3e3a3e3rr", "p", "6", "n", 5),
        }
        self.crash = make_sound("c2a1f1d1c1a0f0d0c0", "n", "777654321", "nnnnnnnnf", 9)
        self.blip = make_sound("e3", "p", "4", "f", 6)
        self.confirm = make_sound("c3g3", "p", "4", "nf", 6)
        self.ring = make_sound("e3g3c4", "p", "5", "nnf", 5)
        self.finish = make_sound("c3e3g3c4rc4", "p", "6", "nnnnnf", 8)
        self.full = make_sound("g3c4e4", "p", "5", "nnf", 6)
        self.fanfare = make_sound("c3e3g3c4e4g4", "s", "566677", "nnnnnf", 7)
        self.playing = SILENT
        self.last_state: FlightState | None = None

    def set_engine(self, note: int) -> None:
        """Engine loops starting at Pyxel note ``note``: lower for a bigger plane."""
        self.engine = [
            self._engine_sound(note, level) for level in range(ENGINE_LEVELS)
        ]
        if hasattr(self, "playing"):  # restart the engine loop with the new sound
            self.playing = replace(self.playing, engine=None)

    @staticmethod
    def _engine_sound(note: int, level: int) -> pyxel.Sound:
        base = note + level * 2
        low, high = note_name(base), note_name(base + 7)
        volume = str(3 + level * 3 // ENGINE_LEVELS)
        return make_sound(low + low + high + low, "snsn", volume, "n", 3)

    @staticmethod
    def _wind_sound(level: int) -> pyxel.Sound:
        return make_sound("a3b3a3g3", "n", str(level), "n", 12)

    @staticmethod
    def _rain_sound(level: int) -> pyxel.Sound:
        # Higher and busier than the wind: a steady patter on the canopy.
        return make_sound("g4e4a4f4d4g4", "n", str(level + 1), "n", 2)

    def update(self, plane: Aircraft, agl: float, rain: float = 0.0) -> None:
        """Bring the looping channels in line with the plane; play a crash once."""
        self.set_state(SoundState.choose(plane, agl, rain))
        if (
            plane.state is FlightState.CRASHED
            and self.last_state is not FlightState.CRASHED
        ):
            pyxel.play(CH_EFFECT, self.crash)
        self.last_state = plane.state

    def set_state(self, state: SoundState) -> None:
        old, self.playing = self.playing, state
        if state.engine != old.engine:
            self._loop(
                CH_ENGINE, None if state.engine is None else self.engine[state.engine]
            )
        if (state.wind, state.rain) != (old.wind, old.rain):
            self._loop(
                CH_WIND, self.rain[state.rain] if state.rain else self.wind[state.wind]
            )
        if state.alert != old.alert:
            self._loop(
                CH_ALERT, None if state.alert is None else self.alerts[state.alert]
            )

    def silence(self) -> None:
        """Stop the loops, e.g. while paused. Effects already playing finish."""
        self.set_state(SILENT)

    def menu_move(self) -> None:
        pyxel.play(CH_EFFECT, self.blip)

    def menu_confirm(self) -> None:
        pyxel.play(CH_EFFECT, self.confirm)

    def ring_passed(self) -> None:
        pyxel.play(CH_EFFECT, self.ring)

    def achievement(self) -> None:
        pyxel.play(CH_EFFECT, self.fanfare)

    def tank_full(self) -> None:
        pyxel.play(CH_EFFECT, self.full)

    def race_finished(self) -> None:
        pyxel.play(CH_EFFECT, self.finish)

    @staticmethod
    def _loop(channel: int, sound: pyxel.Sound | None) -> None:
        if sound is None:
            pyxel.stop(channel)
        else:
            pyxel.play(channel, sound, loop=True)
