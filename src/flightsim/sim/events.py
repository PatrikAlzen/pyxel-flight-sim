"""Things that happen in the game, announced once to whoever listens.

The game publishes an ``Event`` when the plane crashes, lands or passes a
ring; the replay recorder (and later achievements) subscribe instead of the
game calling each of them. Nothing here knows about Pyxel.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class EventKind(Enum):
    CRASH = "CRASH"
    LANDING = "LANDING"
    RING = "RING"


@dataclass(frozen=True)
class Event:
    kind: EventKind
    position: np.ndarray  # where it happened: x, y, z
    text: str = ""  # what the player was told, e.g. the landing grade


@dataclass
class EventLog:
    listeners: list[Callable[[Event], None]] = field(default_factory=list)

    def subscribe(self, listener: Callable[[Event], None]) -> None:
        self.listeners.append(listener)

    def emit(self, kind: EventKind, position, text: str = "") -> Event:
        event = Event(kind, np.asarray(position, dtype=float).copy(), text)
        for listener in self.listeners:
            listener(event)
        return event
