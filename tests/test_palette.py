import pytest

from flightsim.render import palette
from flightsim.terrain.biomes import Biome


def test_palette_fits_pyxel():
    assert len(palette.colors()) <= 256


def test_palette_layout():
    colors = palette.colors()
    last_terrain = palette.biome_color(Biome.SNOW, palette.SHADES - 1)
    assert last_terrain == palette.TERRAIN_BASE + palette.TERRAIN_COLORS - 1
    assert (
        palette.SKY_BASE
        == palette.TERRAIN_BASE + palette.TERRAIN_COLORS * palette.FOG_LEVELS
    )
    assert len(colors) == palette.SKY_BASE + palette.SKY_COLORS


@pytest.mark.parametrize("lighting", palette.LIGHTINGS, ids=lambda lt: lt.name)
def test_fully_fogged_matches_horizon_sky(lighting):
    colors = palette.colors(lighting)
    assert colors[palette.SKY_BASE] == lighting.fog
    assert len(colors) == len(palette.colors())


def test_night_is_darker_than_day():
    def brightness(rgb):
        return sum((rgb >> s) & 0xFF for s in (16, 8, 0))

    grass = palette.biome_color(Biome.GRASS, palette.SHADES - 1)
    day, night = palette.colors(palette.DAY), palette.colors(palette.NIGHT)
    assert brightness(night[grass]) < brightness(day[grass]) / 2
    # HUD colours are untouched.
    assert night[: palette.TERRAIN_BASE] == day[: palette.TERRAIN_BASE]


def test_first_fog_copy_is_clear_terrain():
    colors = palette.colors()
    start = palette.TERRAIN_BASE
    clear = colors[start : start + palette.TERRAIN_COLORS]
    snow = palette.biome_color(Biome.SNOW, palette.SHADES - 1)
    assert clear[snow - start] == palette.RAMPS[Biome.SNOW][1]
