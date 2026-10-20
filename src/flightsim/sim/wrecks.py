"""Where the plane has crashed in this world, so the wrecks stay behind."""

import numpy as np


class Wrecks:
    """Crash sites in the order they happened, oldest dropped past ``limit``."""

    def __init__(self, limit: int = 100) -> None:
        self.limit = limit
        self.positions = np.zeros((0, 3))

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def latest(self) -> np.ndarray | None:
        return self.positions[-1] if len(self) else None

    def add(self, position: np.ndarray) -> None:
        self.positions = np.vstack([self.positions, position])[-self.limit :]

    def clear(self) -> None:
        self.positions = np.zeros((0, 3))

    def near(self, x: float, y: float, radius: float) -> np.ndarray:
        """Wrecks within ``radius`` (horizontally) of (x, y)."""
        offset = self.positions[:, :2] - (x, y)
        return self.positions[np.hypot(offset[:, 0], offset[:, 1]) <= radius]
