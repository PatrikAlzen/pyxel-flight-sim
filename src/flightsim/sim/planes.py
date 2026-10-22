"""The aircraft the player can fly, each with its own handling and look.

A type is flight-model settings plus what the rest of the game needs to know
about it: how big it is drawn, how deep its engine sounds, how much it is
thrown about by turbulence and its colours. ``TRAINER`` is the original plane.
Aeroplanes come first in ``TYPES``, then the helicopters, whose settings are
``HelicopterSettings``; ``make_vehicle`` builds the right flight model.
"""

import math
from dataclasses import dataclass, replace

from flightsim.sim.aircraft import Aircraft, AircraftSettings
from flightsim.sim.helicopter import Helicopter, HelicopterSettings

BASE = AircraftSettings()


@dataclass(frozen=True)
class AircraftType:
    name: str
    description: str
    settings: AircraftSettings | HelicopterSettings
    span: float = 5.5  # world units across the wings (or rotor), for drawing
    engine_note: int = 6  # lowest engine note (Pyxel note number)
    turbulence: float = 1.0  # how strongly bumps throw it about
    body: str = "w"  # sprite colours (see sprites.PLANE_COLORS)
    trim: str = "g"

    @property
    def helicopter(self) -> bool:
        return isinstance(self.settings, HelicopterSettings)

    @property
    def top_speed(self) -> float:
        """Level flight at sea level and full throttle (or lean)."""
        if self.helicopter:
            return self.settings.top_speed
        return (self.settings.max_thrust / self.settings.drag) ** 0.5


TRAINER = AircraftType("TRAINER", "FORGIVING ALL-ROUNDER", BASE)
STUNT = AircraftType(
    "STUNT",
    "QUICK ROLLS, TIGHT LOOPS, SMALL TANK",
    replace(
        BASE,
        max_thrust=50.0,  # top speed ~158
        stall_speed=42.0,
        takeoff_speed=50.0,
        control_speed=65.0,
        max_pitch_rate=1.8,
        max_roll_rate=3.2,
        roll_leveling=0.4,
        turn_factor=2.4,  # tight banked turns despite its speed
        full_burn=1 / 360,  # 6 minutes at full throttle
    ),
    span=4.5,
    engine_note=10,
    turbulence=1.2,
    body="r",
    trim="y",
)
CARGO = AircraftType(
    "CARGO",
    "HEAVY AND STEADY, BIG TANK, LONG TAKEOFF",
    replace(
        BASE,
        max_thrust=36.0,
        drag=36.0 / 110.0**2,  # top speed 110
        stall_speed=56.0,
        takeoff_speed=70.0,
        control_speed=85.0,
        max_pitch_rate=0.7,
        max_roll_rate=1.0,
        control_lag=0.3,
        throttle_rate=0.35,
        full_burn=1 / 720,  # 12 minutes at full throttle
        cruise_throttle=0.6,
    ),
    span=8.5,
    engine_note=2,
    turbulence=0.5,
    body="g",
    trim="b",
)
SEAPLANE = AircraftType(
    "SEAPLANE",
    "FLOATS: LANDS ON WATER, NOT ON GRASS",
    replace(
        BASE,
        drag=0.0025,  # the floats: top speed ~126
        stall_speed=48.0,
        takeoff_speed=56.0,
        floats=True,
    ),
    span=5.5,
    engine_note=7,
    body="w",
    trim="l",
)
ROTOR = HelicopterSettings()
SCOUT = AircraftType(
    "SCOUT",
    "QUICK AND NIMBLE, SMALL TANK",
    replace(
        ROTOR,
        top_speed=80.0,
        max_tilt=math.radians(30),
        tilt_rate=3.5,
        yaw_rate=1.8,
        max_climb=18.0,
        rotor_radius=3.5,
        full_burn=1 / 360,
    ),
    span=7.0,
    engine_note=9,
    turbulence=1.3,
    body="y",
    trim="r",
)
UTILITY = AircraftType(
    "UTILITY",
    "STEADY ALL-ROUNDER, LANDS ANYWHERE",
    ROTOR,
    span=9.0,
    engine_note=5,
    turbulence=0.8,
    body="w",
    trim="r",
)
TANDEM = AircraftType(
    "TANDEM",
    "TWIN ROTORS: SLOW, STEADY, BIG TANK",
    replace(
        ROTOR,
        top_speed=55.0,
        max_tilt=math.radians(18),
        tilt_rate=1.5,
        yaw_rate=0.6,
        max_climb=10.0,
        rotor_radius=4.5,
        full_burn=1 / 720,
    ),
    span=15.0,
    engine_note=1,
    turbulence=0.4,
    body="b",
    trim="g",
)
PLANES = [TRAINER, STUNT, CARGO, SEAPLANE]
HELICOPTERS = [SCOUT, UTILITY, TANDEM]
TYPES = PLANES + HELICOPTERS  # wrecks and saves store the index into this


def make_vehicle(plane_type: AircraftType, sea_level: float) -> Aircraft:
    """A fresh flight model for ``plane_type`` in a world with this sea level."""
    settings = replace(plane_type.settings, sea_level=sea_level)
    if plane_type.helicopter:
        return Helicopter(settings=settings)
    return Aircraft(settings=settings)


def by_name(name: str) -> AircraftType:
    """The type called ``name``; unknown names (old or edited files) give TRAINER."""
    return next((t for t in TYPES if t.name == name), TRAINER)
