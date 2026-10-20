"""Vectorised 2D gradient noise and fractal combinations of it.

Every function takes world-coordinate arrays (any broadcastable shapes) and
evaluates all points in one go, so a whole chunk or screen costs a handful of
NumPy operations per octave instead of a Python loop per sample.

Everything is deterministic: the same seed always produces the same values,
which is what lets the world be regenerated chunk by chunk without seams.
"""

import numpy as np

# Lattice period of a single noise layer. The pattern repeats after this many
# lattice cells, but octaves use different seeds and frequencies, so the sum
# does not visibly repeat.
PERIOD = 1024
_MASK = PERIOD - 1

# Perlin noise in 2D peaks at sqrt(0.5); scale so output covers about [-1, 1].
_NORMALISE = np.sqrt(2.0)


class GradientNoise:
    """Perlin-style gradient noise in roughly [-1, 1]."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(PERIOD)
        self._perm = np.concatenate([perm, perm]).astype(np.int32)
        angles = rng.random(PERIOD) * 2 * np.pi
        # Gradient for hash index h is _grad[h] = gradient[perm[h]], stored as a
        # complex number (x + iy) so each corner needs a single gather.
        gradients = (np.cos(angles) + 1j * np.sin(angles)).astype(np.complex64)
        self._grad = gradients[self._perm]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Lattice coordinates need float64 to stay exact far from the origin;
        # the fractional parts are fine in float32.
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        xi = (x0.astype(np.int64) & _MASK).astype(np.int32)
        yi = (y0.astype(np.int64) & _MASK).astype(np.int32)
        px0 = self._perm[xi]
        px1 = self._perm[xi + 1]
        grad = self._grad

        def corner(
            px: np.ndarray, dy_index: int, dx: np.ndarray, dy: np.ndarray
        ) -> np.ndarray:
            g = grad[px + (yi + dy_index)]
            return g.real * dx + g.imag * dy

        n00 = corner(px0, 0, fx, fy)
        n10 = corner(px1, 0, fx - 1, fy)
        n01 = corner(px0, 1, fx, fy - 1)
        n11 = corner(px1, 1, fx - 1, fy - 1)

        u = _fade(fx)
        v = _fade(fy)
        top = n00 + (n10 - n00) * u
        bottom = n01 + (n11 - n01) * u
        return ((top + (bottom - top) * v) * _NORMALISE).astype(np.float32)


class FractalNoise:
    """Several octaves of gradient noise summed at increasing frequency.

    Each octave has its own seed and a fixed offset, which hides the lattice
    alignment that would otherwise line up at the origin.
    """

    def __init__(
        self, seed: int, octaves: int, lacunarity: float = 2.0, gain: float = 0.5
    ) -> None:
        self.octaves = [GradientNoise(seed * 1009 + i) for i in range(octaves)]
        self.lacunarity = lacunarity
        self.gain = gain
        rng = np.random.default_rng(seed)
        self._offsets = rng.uniform(-PERIOD, PERIOD, size=(octaves, 2))

    def _layers(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        amp, freq = 1.0, 1.0
        for noise, (ox, oy) in zip(self.octaves, self._offsets):
            yield amp, noise(x * freq + ox, y * freq + oy)
            amp *= self.gain
            freq *= self.lacunarity

    def fbm(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fractal Brownian motion in roughly [-1, 1]. Smooth, rolling shapes."""
        total = np.zeros(
            np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=np.float32
        )
        norm = 0.0
        for amp, n in self._layers(x, y):
            total += amp * n
            norm += amp
        return total / norm

    def ridged(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ridged multifractal in [0, 1]. Sharp crests where the noise crosses zero.

        Each octave is weighted by the previous one, so fine detail gathers on
        the ridges and valleys stay smoother, which is how real mountains look.
        """
        total = np.zeros(
            np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=np.float32
        )
        weight = np.ones_like(total)
        norm = 0.0
        for amp, n in self._layers(x, y):
            ridge = (1.0 - np.abs(n)) ** 2
            total += amp * ridge * weight
            weight = np.clip(ridge * 2.0, 0.0, 1.0)
            norm += amp
        return np.clip(total / norm, 0.0, 1.0)


def domain_warp(
    x: np.ndarray,
    y: np.ndarray,
    warp_x: FractalNoise,
    warp_y: FractalNoise,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset coordinates by a noise field to bend straight features into organic ones.

    ``strength`` is in the same units as ``x`` and ``y``.
    """
    return x + warp_x.fbm(x, y) * strength, y + warp_y.fbm(x, y) * strength


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _fade(t: np.ndarray) -> np.ndarray:
    """Quintic easing: zero first and second derivative at the lattice points."""
    return t * t * t * (t * (t * 6 - 15) + 10)
