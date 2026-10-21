import pytest


@pytest.fixture(scope="session")
def screen():
    """A headless Pyxel window with the game palette. Pyxel can only be
    initialised once per process, so every test that draws shares this one."""
    import pyxel

    from flightsim.render import palette

    pyxel.init(256, 144, headless=True)
    palette.install()
    return pyxel
