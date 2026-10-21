"""Weather: wind, gusts, turbulence, clouds and rain, changing slowly over time.

The weather moves between a handful of presets. Every few minutes it picks a
neighbouring one (clear can turn partly cloudy, rain can clear up or become a
storm) and blends over to it, so nothing ever jumps. The player can also lock
it to one preset from the pause menu.

Wind is one vector used for everything that drifts: the plane, the clouds and
smoke. Turbulence is a pair of small, smoothly varying kicks to the plane's
pitch and roll rates.

Nothing here knows about Pyxel; all randomness comes from the world seed, so
the same seed gives the same weather.
"""

import math
from dataclasses import dataclass, fields, replace

import numpy as np


@dataclass(frozen=True)
class Conditions:
    name: str
    wind: float  # mean wind speed, world units per second
    gustiness: float  # 0..1, how much the wind varies around its mean
    cloud_cover: float  # 0..1, share of the sky covered
    cloud_base: float  # altitude of the cloud layer
    rain: float  # 0..1
    turbulence: float  # 0..1, everywhere; more near mountains and in cloud
    overcast: float  # 0..1, how grey and dim the light is
    visibility: float  # 0..1, share of the clear-weather view distance


PRESETS = [
    Conditions("CLEAR", 4.0, 0.1, 0.12, 420.0, 0.0, 0.0, 0.0, 1.0),
    Conditions("PARTLY CLOUDY", 8.0, 0.25, 0.35, 380.0, 0.0, 0.1, 0.1, 1.0),
    Conditions("OVERCAST", 10.0, 0.3, 0.75, 300.0, 0.0, 0.15, 0.5, 0.8),
    Conditions("RAIN", 14.0, 0.4, 0.85, 250.0, 0.7, 0.3, 0.75, 0.55),
    Conditions("STORM", 24.0, 0.7, 0.95, 220.0, 1.0, 0.65, 0.9, 0.4),
]
_NUMBERS = [f.name for f in fields(Conditions) if f.name != "name"]


def blend(a: Conditions, b: Conditions, t: float) -> Conditions:
    """Conditions part way (``t`` in 0..1) from ``a`` to ``b``."""
    values = {n: getattr(a, n) + (getattr(b, n) - getattr(a, n)) * t for n in _NUMBERS}
    return replace(a, name=b.name if t >= 0.5 else a.name, **values)


@dataclass(frozen=True)
class WeatherSettings:
    hold: tuple[float, float] = (120.0, 240.0)  # seconds before the weather turns
    transition: float = 40.0  # seconds to blend into new weather
    locked_transition: float = 8.0  # quicker when the player picks the weather
    direction_drift: float = 0.02  # radians per second the wind slowly turns
    # Turbulence kicks at full strength, radians per second.
    pitch_kick: float = 0.22
    roll_kick: float = 0.45


class Weather:
    def __init__(self, seed: int, settings: WeatherSettings | None = None) -> None:
        self.settings = s = settings or WeatherSettings()
        self.rng = np.random.default_rng([seed & 0xFFFFFFFF, 41])
        self.current = PRESETS[
            int(self.rng.integers(0, 2))
        ]  # start clear or partly cloudy
        self.previous = self.current
        self.target = self.current
        self.blend_time = 0.0  # seconds into the current transition
        self.transition = s.transition
        self.hold = float(self.rng.uniform(*s.hold))
        self.locked: Conditions | None = None
        self.wind_direction = float(self.rng.uniform(0, math.tau))  # blows towards
        self.time = 0.0
        # Phases for the sums of sines behind gusts and turbulence.
        self._phase = self.rng.uniform(0, math.tau, size=12)

    @property
    def mode(self) -> str:
        """``AUTO`` while changing by itself, otherwise the locked preset's name."""
        return "AUTO" if self.locked is None else self.locked.name

    def lock(self, preset: Conditions | None) -> None:
        """Hold the weather at ``preset`` (blending over quickly), or None for auto."""
        self.locked = preset
        if preset is not None:
            self._start(preset, self.settings.locked_transition)

    def update(self, dt: float) -> None:
        s = self.settings
        self.time += dt
        self.wind_direction += (
            s.direction_drift * dt * math.sin(self.time / 97 + self._phase[0])
        )
        if self.target is not self.current or self.blend_time < self.transition:
            self.blend_time = min(self.blend_time + dt, self.transition)
            t = _smooth(self.blend_time / self.transition)
            self.current = blend(self.previous, self.target, t)
        if self.locked is None:
            self.hold -= dt
            if self.hold <= 0:
                self._start(self._neighbour(), s.transition)
                self.hold = float(self.rng.uniform(*s.hold))

    def wind(self) -> tuple[float, float]:
        """Wind velocity (x, y) right now, gusts included."""
        c, p = self.current, self._phase
        t = self.time
        gust = (math.sin(t * 0.7 + p[1]) + 0.6 * math.sin(t * 1.9 + p[2])) / 1.6
        speed = max(0.0, c.wind * (1.0 + c.gustiness * gust))
        return math.sin(self.wind_direction) * speed, -math.cos(
            self.wind_direction
        ) * speed

    def turbulence_kick(self, intensity: float) -> tuple[float, float]:
        """Pitch and roll rate disturbances (radians per second) at ``intensity`` 0..1.

        Smooth sums of sines at unrelated frequencies, so the bumps feel random
        without being jerky from frame to frame.
        """
        s, p, t = self.settings, self._phase, self.time
        pitch = (math.sin(t * 2.3 + p[3]) + math.sin(t * 5.1 + p[4]) * 0.6
                 + math.sin(t * 8.7 + p[5]) * 0.3) / 1.9  # fmt: skip
        roll = (math.sin(t * 1.7 + p[6]) + math.sin(t * 4.3 + p[7]) * 0.6
                + math.sin(t * 9.1 + p[8]) * 0.3) / 1.9  # fmt: skip
        return pitch * s.pitch_kick * intensity, roll * s.roll_kick * intensity

    def _start(self, target: Conditions, transition: float) -> None:
        self.previous = self.current
        self.target = target
        self.blend_time = 0.0
        self.transition = transition

    def _neighbour(self) -> Conditions:
        names = [p.name for p in PRESETS]
        i = names.index(self.target.name)
        # Storms and rain clear up more readily than clear skies turn bad.
        steps = [-1, 1] if 0 < i < len(PRESETS) - 1 else ([1] if i == 0 else [-1])
        weights = np.array([1.4 if step < 0 else 1.0 for step in steps])
        step = int(self.rng.choice(steps, p=weights / weights.sum()))
        return PRESETS[i + step]


def _smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def turbulence_intensity(
    conditions: Conditions,
    agl: float,
    terrain_slope: float,
    in_cloud: bool,
) -> float:
    """How bumpy the air is for a plane ``agl`` units above terrain this steep.

    The weather's own turbulence, more low over steep ground (air tumbling
    over ridges, strongest in strong wind) and more inside cloud.
    """
    low = max(0.0, 1.0 - agl / 200.0)
    ridge = min(1.0, terrain_slope) * low * min(1.0, conditions.wind / 15.0)
    cloud = 0.4 if in_cloud else 0.0
    return min(1.0, conditions.turbulence + 0.6 * ridge + cloud)
