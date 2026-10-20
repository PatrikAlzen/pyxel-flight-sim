"""Smoke, fire and debris particles in world space.

All particles live in a few NumPy arrays, so updating hundreds of them is a
handful of vector operations. Nothing here knows about Pyxel; drawing is in
``render/effects.py``.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Kind(IntEnum):
    SMOKE = 0
    FIRE = 1
    SPARK = 2


@dataclass(frozen=True)
class ParticleSettings:
    capacity: int = 700
    gravity: float = 30.0
    # Smoke floats up, slows down and drifts with the wind.
    smoke_lift: float = 9.0  # upward acceleration
    smoke_drag: float = 0.9  # per second, towards the wind velocity
    # Fire rises quickly and dies young.
    fire_lift: float = 16.0
    fire_drag: float = 2.0
    # Continuous emission from a burning wreck, per second.
    smoke_rate: float = 28.0
    fire_rate: float = 22.0
    smoulder_rate: float = 2.5  # per old wreck


@dataclass(frozen=True)
class _Style:
    """How one kind of particle is born: speeds, lifetime and size ranges."""

    speed: tuple[float, float]
    up: tuple[float, float]  # extra upward speed
    life: tuple[float, float]
    size: tuple[float, float]  # starting radius in world units
    growth: tuple[float, float]  # radius growth per second


_BURN = {
    Kind.SMOKE: _Style((0.3, 1.5), (3.0, 6.0), (3.5, 6.0), (0.5, 1.0), (0.5, 1.1)),
    Kind.FIRE: _Style((0.5, 3.0), (4.0, 9.0), (0.4, 0.9), (0.9, 1.6), (-0.6, 0.2)),
}
# A long-dead wreck still smoulders: a thin wisp that rises high.
_SMOULDER = _Style((0.1, 0.5), (2.0, 4.0), (5.0, 8.0), (0.4, 0.7), (0.6, 1.0))
_EXPLOSION = {
    Kind.FIRE: _Style((8.0, 22.0), (4.0, 14.0), (0.4, 1.0), (2.0, 4.0), (1.0, 3.0)),
    Kind.SMOKE: _Style((3.0, 9.0), (3.0, 9.0), (2.0, 4.0), (1.0, 2.0), (0.8, 1.6)),
    Kind.SPARK: _Style((10.0, 24.0), (2.0, 10.0), (1.2, 2.2), (0.25, 0.35), (0.0, 0.0)),
}


class Particles:
    def __init__(self, settings: ParticleSettings | None = None, seed: int = 0) -> None:
        self.settings = settings or ParticleSettings()
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros((0, 3))
        self.velocity = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.life = np.zeros(0)
        self.size = np.zeros(0)
        self.growth = np.zeros(0)
        self.kind = np.zeros(0, dtype=np.int8)
        self._owed = {
            Kind.SMOKE: 0.0,
            Kind.FIRE: 0.0,
        }  # fractional emission carried over

    def __len__(self) -> int:
        return len(self.age)

    def clear(self) -> None:
        keep = np.zeros(len(self), dtype=bool)
        self._keep(keep)

    @property
    def radius(self) -> np.ndarray:
        """Current radius of every particle in world units."""
        return np.maximum(self.size + self.growth * self.age, 0.05)

    @property
    def fade(self) -> np.ndarray:
        """Remaining life of every particle, 1 (new) .. 0 (about to vanish)."""
        return 1.0 - self.age / self.life

    def explode(self, at: np.ndarray) -> None:
        """A fireball with flying sparks and a burst of smoke."""
        for kind, count in ((Kind.SMOKE, 30), (Kind.FIRE, 45), (Kind.SPARK, 40)):
            self._emit(at, count, kind, _EXPLOSION[kind], spread=2.0)

    def burn(self, at: np.ndarray, dt: float) -> None:
        """Keep a wreck burning: a steady stream of flames and smoke."""
        s = self.settings
        for kind, rate in ((Kind.SMOKE, s.smoke_rate), (Kind.FIRE, s.fire_rate)):
            self._owed[kind] += rate * dt
            count = int(self._owed[kind])
            self._owed[kind] -= count
            if count:
                self._emit(at, count, kind, _BURN[kind], spread=1.5)

    def smoulder(self, wrecks: np.ndarray, dt: float) -> None:
        """Thin smoke from old wrecks (rows of ``wrecks``), so they can be spotted from afar.

        Each wreck emits at random at ``smoulder_rate`` on average, which needs
        no bookkeeping per wreck.
        """
        counts = self.rng.poisson(self.settings.smoulder_rate * dt, size=len(wrecks))
        for at, count in zip(wrecks, counts):
            if count:
                self._emit(
                    at + [0.0, 0.0, 1.0], int(count), Kind.SMOKE, _SMOULDER, spread=0.5
                )

    def update(self, dt: float, wind: tuple[float, float]) -> None:
        s = self.settings
        kind = self.kind
        accel = np.zeros_like(self.velocity)
        wind_velocity = np.array([wind[0], wind[1], 0.0])

        smoke, fire, spark = kind == Kind.SMOKE, kind == Kind.FIRE, kind == Kind.SPARK
        for mask, lift, drag in (
            (smoke, s.smoke_lift, s.smoke_drag),
            (fire, s.fire_lift, s.fire_drag),
        ):
            if mask.any():
                towards_wind = (wind_velocity - self.velocity[mask]) * drag
                accel[mask] = towards_wind + np.array([0.0, 0.0, lift])
        accel[spark, 2] -= s.gravity

        self.velocity += accel * dt
        self.position += self.velocity * dt
        self.age += dt
        self._keep(self.age < self.life)

    def _emit(
        self, at: np.ndarray, count: int, kind: Kind, style: _Style, spread: float
    ) -> None:
        count = min(count, self.settings.capacity)
        rng = self.rng
        direction = rng.normal(size=(count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        direction[:, 2] = np.abs(direction[:, 2])  # nothing flies into the ground
        speed = rng.uniform(*style.speed, size=(count, 1))
        velocity = direction * speed
        velocity[:, 2] += rng.uniform(*style.up, size=count)
        position = np.asarray(at, dtype=float) + rng.normal(
            scale=spread, size=(count, 3)
        ) * [1, 1, 0.3]

        self.position = np.concatenate([self.position, position])
        self.velocity = np.concatenate([self.velocity, velocity])
        self.age = np.concatenate([self.age, np.zeros(count)])
        self.life = np.concatenate([self.life, rng.uniform(*style.life, size=count)])
        self.size = np.concatenate([self.size, rng.uniform(*style.size, size=count)])
        self.growth = np.concatenate(
            [self.growth, rng.uniform(*style.growth, size=count)]
        )
        self.kind = np.concatenate([self.kind, np.full(count, kind, dtype=np.int8)])

        # Over capacity: the oldest particles make room.
        excess = len(self) - self.settings.capacity
        if excess > 0:
            keep = np.ones(len(self), dtype=bool)
            keep[np.argsort(-self.age / self.life)[:excess]] = False
            self._keep(keep)

    def _keep(self, keep: np.ndarray) -> None:
        for name in ("position", "velocity", "age", "life", "size", "growth", "kind"):
            setattr(self, name, getattr(self, name)[keep])
