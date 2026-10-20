"""Top-down debug map of the procedural world, for tuning terrain generation.

    arrows      pan (hold SHIFT for faster)
    Z / X       zoom in / out
    1 / 2 / 3   layer: shaded terrain / flat biomes / moisture
    N / B       next / previous seed
    H           toggle help
    Q / ESC     quit

Hover the mouse over the map to read height, biome and moisture.
"""

import time

import numpy as np
import pyxel

from flightsim import config
from flightsim.render import palette
from flightsim.render.framebuffer import screen_array
from flightsim.terrain.biomes import Biome
from flightsim.terrain.generator import TerrainGenerator, TerrainSample

W, H = config.SCREEN_W, config.SCREEN_H
ZOOMS = [1, 2, 4, 8, 16, 32, 64]
PAN_PIXELS = 4

# Moisture bands from dry to wet, using the default Pyxel colours.
MOISTURE_COLORS = np.array(
    [
        pyxel.COLOR_BROWN,
        pyxel.COLOR_ORANGE,
        pyxel.COLOR_YELLOW,
        pyxel.COLOR_LIME,
        pyxel.COLOR_GREEN,
        pyxel.COLOR_CYAN,
        pyxel.COLOR_LIGHT_BLUE,
        pyxel.COLOR_DARK_BLUE,
    ],
    dtype=np.uint8,
)

LAYERS = ["terrain", "biomes", "moisture"]


class MapView:
    def __init__(self) -> None:
        self.seed = config.SEED
        self.generator = TerrainGenerator(self.seed)
        self.cx, self.cy = 0.0, 0.0
        self.zoom_index = ZOOMS.index(8)
        self.layer = 0
        self.show_help = True
        self.sample: TerrainSample | None = None
        self.image = np.zeros((H, W), dtype=np.uint8)
        self.gen_ms = 0.0
        self.dirty = True

        pyxel.init(
            W,
            H,
            title=f"{config.TITLE} - map",
            fps=config.FPS,
            display_scale=config.DISPLAY_SCALE,
        )
        palette.install()
        pyxel.mouse(True)
        pyxel.run(self.update, self.draw)

    @property
    def zoom(self) -> int:
        return ZOOMS[self.zoom_index]

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.generator = TerrainGenerator(seed)
        self.dirty = True

    def update(self) -> None:
        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()
        if pyxel.btnp(pyxel.KEY_H):
            self.show_help = not self.show_help
        if pyxel.btnp(pyxel.KEY_N):
            self.set_seed(self.seed + 1)
        if pyxel.btnp(pyxel.KEY_B):
            self.set_seed(self.seed - 1)
        for key, layer in [(pyxel.KEY_1, 0), (pyxel.KEY_2, 1), (pyxel.KEY_3, 2)]:
            if pyxel.btnp(key) and self.layer != layer:
                self.layer = layer
                self.recolor()
        if pyxel.btnp(pyxel.KEY_Z) and self.zoom_index > 0:
            self.zoom_index -= 1
            self.dirty = True
        if pyxel.btnp(pyxel.KEY_X) and self.zoom_index < len(ZOOMS) - 1:
            self.zoom_index += 1
            self.dirty = True

        fast = 4 if pyxel.btn(pyxel.KEY_SHIFT) else 1
        dx = pyxel.btn(pyxel.KEY_RIGHT) - pyxel.btn(pyxel.KEY_LEFT)
        dy = pyxel.btn(pyxel.KEY_DOWN) - pyxel.btn(pyxel.KEY_UP)
        if dx or dy:
            self.cx += dx * PAN_PIXELS * fast * self.zoom
            self.cy += dy * PAN_PIXELS * fast * self.zoom
            self.dirty = True

        if self.dirty:
            self.regenerate()

    def regenerate(self) -> None:
        start = time.perf_counter()
        self.sample = self.generator.sample(
            self.cx - W / 2 * self.zoom, self.cy - H / 2 * self.zoom, W, H, self.zoom
        )
        self.gen_ms = (time.perf_counter() - start) * 1000
        self.dirty = False
        self.recolor()

    def recolor(self) -> None:
        s = self.sample
        if s is None:
            return
        if self.layer == 0:
            self.image = s.color
        elif self.layer == 1:
            self.image = palette.biome_color(s.biome, palette.SHADES - 2)
        else:
            bands = np.minimum(
                (s.moisture * len(MOISTURE_COLORS)).astype(int),
                len(MOISTURE_COLORS) - 1,
            )
            self.image = MOISTURE_COLORS[bands]

    def draw(self) -> None:
        screen_array()[:] = self.image
        self.draw_status()
        self.draw_hover()
        if self.show_help:
            self.draw_help()

    def draw_status(self) -> None:
        text = (
            f"SEED {self.seed}  X {self.cx:.0f} Y {self.cy:.0f}  1:{self.zoom}  "
            f"{LAYERS[self.layer].upper()}  {self.gen_ms:.0f}MS"
        )
        pyxel.rect(0, H - 8, W, 8, pyxel.COLOR_BLACK)
        pyxel.text(2, H - 7, text, pyxel.COLOR_WHITE)

    def draw_hover(self) -> None:
        s = self.sample
        mx, my = pyxel.mouse_x, pyxel.mouse_y
        if s is None or not (0 <= mx < W and 0 <= my < H - 8):
            return
        wx = s.x0 + mx * s.step
        wy = s.y0 + my * s.step
        biome = Biome(int(s.biome[my, mx]))
        text = (
            f"({wx:.0f},{wy:.0f})  H {s.ground[my, mx]:.0f}  "
            f"{biome.label.upper()}  MOIST {s.moisture[my, mx]:.2f}"
        )
        pyxel.rect(0, 0, len(text) * 4 + 3, 8, pyxel.COLOR_BLACK)
        pyxel.text(2, 1, text, pyxel.COLOR_WHITE)

    def draw_help(self) -> None:
        lines = [
            "ARROWS PAN (SHIFT FAST)",
            "Z/X    ZOOM IN/OUT",
            "1/2/3  TERRAIN/BIOME/MOIST",
            "N/B    NEXT/PREV SEED",
            "H      HIDE HELP",
        ]
        x, y = W - 4 * 26 - 4, 10
        pyxel.rect(x - 2, y - 2, 4 * 26 + 4, len(lines) * 7 + 3, pyxel.COLOR_BLACK)
        for i, line in enumerate(lines):
            pyxel.text(x, y + i * 7, line, pyxel.COLOR_GRAY)


def main() -> None:
    MapView()


if __name__ == "__main__":
    main()
