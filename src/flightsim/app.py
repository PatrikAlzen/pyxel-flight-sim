"""Main Pyxel application."""

import pyxel

from flightsim import config


class App:
    def __init__(self) -> None:
        pyxel.init(
            config.SCREEN_W,
            config.SCREEN_H,
            title=config.TITLE,
            fps=config.FPS,
            display_scale=config.DISPLAY_SCALE,
        )
        pyxel.run(self.update, self.draw)

    def update(self) -> None:
        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()

    def draw(self) -> None:
        pyxel.cls(pyxel.COLOR_NAVY)
        msg = "PYXEL FLIGHT SIM"
        pyxel.text(
            (config.SCREEN_W - len(msg) * 4) // 2,
            config.SCREEN_H // 2 - 4,
            msg,
            pyxel.COLOR_WHITE,
        )
        hint = "Q/ESC: QUIT"
        pyxel.text(
            (config.SCREEN_W - len(hint) * 4) // 2,
            config.SCREEN_H // 2 + 8,
            hint,
            pyxel.COLOR_GRAY,
        )


def main() -> None:
    App()
