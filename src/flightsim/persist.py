"""What the player has left behind in each world, saved between sessions.

One small JSON file per seed holds the wrecks and saved places, and one
profile file holds what belongs to the player in every world (settings such
as fuel on or off). Files carry a
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
    landmarks: list[int] = field(default_factory=list)  # lighthouse numbers discovered

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
        landmarks = [int(n) for n in data.get("landmarks", [])]
        return cls(
            seed,
            wrecks=wrecks,
            places=places,
            visited=visited,
            records=records,
            landmarks=landmarks,
        )


@dataclass
class Profile:
    """The player's settings and progress across all worlds."""

    fuel: bool = False  # fuel runs out and must be refilled at airfields

    def to_dict(self) -> dict:
        return {"version": VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> Profile:
        return cls(fuel=bool(data.get("fuel", False)))


class SaveStore:
    """Loads and saves ``WorldSave`` files in one directory."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def path(self, seed: int) -> Path:
        return self.directory / f"world-{seed}.json"

    def load(self, seed: int) -> WorldSave:
        data = self._read(self.path(seed))
        if data is None:
            return WorldSave(seed)
        try:
            return WorldSave.from_dict(data, seed)
        except ValueError, TypeError, KeyError, AttributeError:
            self._set_aside(self.path(seed))
            return WorldSave(seed)

    def save(self, world: WorldSave) -> None:
        self._write(self.path(world.seed), world.to_dict())

    @property
    def profile_path(self) -> Path:
        return self.directory / "profile.json"

    def load_profile(self) -> Profile:
        data = self._read(self.profile_path)
        if data is None:
            return Profile()
        try:
            return Profile.from_dict(data)
        except ValueError, TypeError, KeyError, AttributeError:
            self._set_aside(self.profile_path)
            return Profile()

    def save_profile(self, profile: Profile) -> None:
        self._write(self.profile_path, profile.to_dict())

    def _read(self, path: Path) -> dict | None:
        """The JSON object in ``path``; None if missing or unreadable (and then
        an unreadable file is moved aside for inspection, not overwritten)."""
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except OSError, ValueError:
            self._set_aside(path)
            return None
        if not isinstance(data, dict):
            self._set_aside(path)
            return None
        return data

    @staticmethod
    def _set_aside(path: Path) -> None:
        try:
            path.replace(path.with_suffix(".json.corrupt"))
        except OSError:
            pass

    def _write(self, path: Path, data: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=1))
        os.replace(temporary, path)

    def photo_path(self, seed: int, x: float, y: float, lighting: str) -> Path:
        """Where to save a photo, without the ``.png`` Pyxel adds: named after
        the seed, the position and the light, numbered if that is taken."""
        directory = self.directory / "photos"
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"seed{seed}_x{round(x)}_y{round(y)}_{lighting.lower()}"
        path, number = directory / stem, 2
        while path.with_name(path.name + ".png").exists():
            path, number = directory / f"{stem}_{number}", number + 1
        return path

    def summary(self, seed: int) -> tuple[int, int]:
        """Number of wrecks and places saved for ``seed``, without side effects."""
        try:
            data = json.loads(self.path(seed).read_text())
            return len(data.get("wrecks", [])), len(data.get("places", []))
        except OSError, ValueError, TypeError, AttributeError:
            return 0, 0
