"""The aircraft the player can fly, each with its own handling and look.

A type is flight-model settings plus what the rest of the game needs to know
about it: how big it is drawn, how deep its engine sounds, how much it is
thrown about by turbulence and its colours. ``TRAINER`` is the original plane.
"""

from dataclasses import dataclass, replace

from flightsim.sim.aircraft import AircraftSettings

BASE = AircraftSettings()


@dataclass(frozen=True)
class AircraftType:
    name: str
    description: str
    settings: AircraftSettings
    span: float = 5.5  # world units across the wings, for drawing
    engine_note: int = 6  # lowest engine note (Pyxel note number)
    turbulence: float = 1.0  # how strongly bumps throw it about
    body: str = "w"  # sprite colours (see sprites.PLANE_COLORS)
    trim: str = "g"

    @property
    def top_speed(self) -> float:
        """Level flight at sea level and full throttle."""
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
TYPES = [TRAINER, STUNT, CARGO, SEAPLANE]


def by_name(name: str) -> AircraftType:
    """The type called ``name``; unknown names (old or edited files) give TRAINER."""
    return next((t for t in TYPES if t.name == name), TRAINER)
