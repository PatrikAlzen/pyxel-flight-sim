import numpy as np
import pyxel

from flightsim.render.framebuffer import image_array


def test_image_array_shape_and_dtype():
    img = pyxel.Image(8, 4)
    arr = image_array(img)
    assert arr.shape == (4, 8)
    assert arr.dtype == np.uint8


def test_image_array_writes_reach_image():
    img = pyxel.Image(8, 4)
    arr = image_array(img)
    arr[2, 5] = 9
    arr[0, :] = 3
    assert img.pget(5, 2) == 9
    assert all(img.pget(x, 0) == 3 for x in range(8))


def test_image_array_reads_image():
    img = pyxel.Image(8, 4)
    img.pset(1, 3, 12)
    assert image_array(img)[3, 1] == 12
