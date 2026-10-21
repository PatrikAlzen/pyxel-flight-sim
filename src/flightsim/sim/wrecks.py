"""Where the plane has crashed in this world, so the wrecks stay behind."""

import numpy as np


class Wrecks:
    """Crash sites in the order they happened, oldest dropped past ``limit``.

    Each wreck remembers which aircraft type it was (an index into
    ``planes.TYPES``), so it keeps its colours.
    """

    def __init__(self, limit: int = 100) -> None:
        self.limit = limit
        self.positions = np.zeros((0, 3))
        self.kinds = np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def latest(self) -> np.ndarray | None:
        return self.positions[-1] if len(self) else None

    def add(self, position: np.ndarray, kind: int = 0) -> None:
        self.positions = np.vstack([self.positions, position[:3]])[-self.limit :]
        self.kinds = np.append(self.kinds, kind)[-self.limit :]

    def clear(self) -> None:
        self.positions = np.zeros((0, 3))
        self.kinds = np.zeros(0, dtype=np.int64)

    def settle(self, ground) -> None:
        """Put every wreck back on the ground (``ground(x, y)`` per array), e.g.
        after the terrain has changed under old saved wrecks."""
        if len(self):
            self.positions[:, 2] = ground(self.positions[:, 0], self.positions[:, 1])

    def _within(self, x: float, y: float, radius: float) -> np.ndarray:
        offset = self.positions[:, :2] - (x, y)
        return np.hypot(offset[:, 0], offset[:, 1]) <= radius

    def near(self, x: float, y: float, radius: float) -> np.ndarray:
        """Wrecks within ``radius`` (horizontally) of (x, y)."""
        return self.positions[self._within(x, y, radius)]

    def near_with_kinds(
        self, x: float, y: float, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Like ``near``, with each wreck's aircraft type."""
        within = self._within(x, y, radius)
        return self.positions[within], self.kinds[within]

    def to_list(self) -> list[list[float]]:
        """[x, y, z, type] per wreck, for saving."""
        return [[*map(float, p), int(k)] for p, k in zip(self.positions, self.kinds)]
