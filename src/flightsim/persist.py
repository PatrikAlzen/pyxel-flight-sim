"""What the player has left behind in each world, saved between sessions.

One small JSON file per seed holds the wrecks and saved places. Files carry a
version number; fields missing from older files get their defaults and fields
this version does not know are ignored, so the format can grow.

Writes go to a temporary file that then replaces the real one, so a crash or
power cut mid-write never leaves half a file. A file that cannot be read is
moved aside (``.corrupt``) rather than overwritten, and the world starts with
nothing saved.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

VERSION = 1
MAX_PLACES = 20


@dataclass
class Place:
    name: str
    x: float
    y: float
    altitude: float
    heading: float


@dataclass
class WorldSave:
    seed: int
    wrecks: list[list[float]] = field(default_factory=list)  # [x, y, z] each
    places: list[Place] = field(default_factory=list)
    visited: list[int] = field(default_factory=list)  # airfield numbers landed at
    # Best time trial per airfield number: {"splits": [...], "ghost": [[x, y, z], ...]}.
    records: dict[int, dict] = field(default_factory=dict)

    def add_place(self, x: float, y: float, altitude: float, heading: float) -> Place:
        """Save a place under the next free number; the oldest go past MAX_PLACES."""
        used = {p.name for p in self.places}
        number = next(n for n in range(1, len(used) + 2) if f"PLACE {n}" not in used)
        place = Place(f"PLACE {number}", x, y, altitude, heading)
        self.places = (self.places + [place])[-MAX_PLACES:]
        return place

    def to_dict(self) -> dict:
        return {"version": VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict, seed: int) -> WorldSave:
        places = [
            Place(
                name=str(p["name"]),
                x=float(p["x"]),
                y=float(p["y"]),
                altitude=float(p["altitude"]),
                heading=float(p["heading"]),
            )
            for p in data.get("places", [])
        ]
        wrecks = [[float(v) for v in w[:3]] for w in data.get("wrecks", [])]
        visited = [int(n) for n in data.get("visited", [])]
        records = {
            int(number): {
                "splits": [float(t) for t in record["splits"]],
                "ghost": [[float(v) for v in p[:3]] for p in record.get("ghost", [])],
            }
            for number, record in data.get("records", {}).items()
        }
        return cls(seed, wrecks=wrecks, places=places, visited=visited, records=records)


class SaveStore:
    """Loads and saves ``WorldSave`` files in one directory."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def path(self, seed: int) -> Path:
        return self.directory / f"world-{seed}.json"

    def load(self, seed: int) -> WorldSave:
        path = self.path(seed)
        try:
            data = json.loads(path.read_text())
            return WorldSave.from_dict(data, seed)
        except FileNotFoundError:
            return WorldSave(seed)
        except OSError, ValueError, TypeError, KeyError, AttributeError:
            # Unreadable: keep it for inspection instead of overwriting it.
            try:
                path.replace(path.with_suffix(".json.corrupt"))
            except OSError:
                pass
            return WorldSave(seed)

    def save(self, world: WorldSave) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(world.seed)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(world.to_dict(), indent=1))
        os.replace(temporary, path)

    def summary(self, seed: int) -> tuple[int, int]:
        """Number of wrecks and places saved for ``seed``, without side effects."""
        try:
            data = json.loads(self.path(seed).read_text())
            return len(data.get("wrecks", [])), len(data.get("places", []))
        except OSError, ValueError, TypeError, AttributeError:
            return 0, 0
