"""Global settings for the game."""

import math

TITLE = "Pyxel Flight Sim"

# Internal render resolution (16:9). Pyxel scales the window up to fit the display.
SCREEN_W = 256
SCREEN_H = 144
DISPLAY_SCALE = 4
FPS = 30

SEED = 1337

# Terrain levels of detail kept around the camera, finest first: (cells per
# side, world units per cell). Each must reach far enough for the depth range
# the renderer assigns it.
TERRAIN_LEVELS = [(1024, 1.0), (1024, 2.0), (1024, 8.0), (512, 32.0)]
# Background processes generating terrain tiles.
TERRAIN_WORKERS = 2

# Flying.
COCKPIT_HEIGHT = 1.0  # camera height above the aircraft's reference point
RESPAWN_DELAY = 4.0  # seconds after a crash
CRASH_VIEW_DISTANCE = 40.0  # the crash is watched from this far away
CRASH_VIEW_HEIGHT = 18.0  # ...and this high above the ground or crash site
CRASH_ORBIT_SPEED = math.radians(12)  # the crash camera circles the wreck

# Free camera.
GHOST_SPEED = 2.0  # world units per frame
GHOST_FAST = 5.0  # speed multiplier while SHIFT is held
GHOST_TURN = 0.04  # radians per frame
GHOST_CLEARANCE = 4.0  # minimum height above the ground
