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


def test_fully_fogged_matches_horizon_sky():
    colors = palette.colors()
    assert colors[palette.SKY_BASE] == palette.FOG_COLOR


def test_first_fog_copy_is_clear_terrain():
    colors = palette.colors()
    start = palette.TERRAIN_BASE
    clear = colors[start : start + palette.TERRAIN_COLORS]
    snow = palette.biome_color(Biome.SNOW, palette.SHADES - 1)
    assert clear[snow - start] == palette.RAMPS[Biome.SNOW][1]
