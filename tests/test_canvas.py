from flightsim.render import canvas, palette
from flightsim.render.framebuffer import image_array, scale_onto_screen, screen_array


def test_interface_layer_is_drawn_apart_and_laid_over_the_screen(screen):
    layer = screen.Image(128, 72)
    canvas.use(layer)
    try:
        assert (canvas.width(), canvas.height()) == (128, 72)
        canvas.cls(palette.CLEAR)
        canvas.rect(4, 2, 3, 2, screen.COLOR_RED)
        canvas.text(10, 5, "HI", screen.COLOR_WHITE)
        assert canvas.array()[2, 4] == screen.COLOR_RED
    finally:
        canvas.use(None)
    assert (canvas.width(), canvas.height()) == (256, 144)
    screen_array()[:] = screen.COLOR_NAVY
    # Twice the size: its pixel (x, y) covers screen (x * 2, y * 2).
    scale_onto_screen(image_array(layer), palette.CLEAR)
    pixels = screen_array()
    assert (pixels[4:8, 8:14] == screen.COLOR_RED).all()
    assert pixels[3, 8] == screen.COLOR_NAVY  # CLEAR leaves the screen as it was
    assert (pixels[10:22, 20:36] == screen.COLOR_WHITE).any()
    assert not (pixels[24:] != screen.COLOR_NAVY).any()
    scale_onto_screen(image_array(layer))  # no see-through colour: all of it
    assert (screen_array() == palette.CLEAR).sum() > 0
