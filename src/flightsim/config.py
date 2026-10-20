"""Global settings for the game."""

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
TERRAIN_LEVELS = [(1024, 1.0), (1024, 2.0), (512, 8.0)]
# Background processes generating terrain tiles.
TERRAIN_WORKERS = 2

# Flying.
COCKPIT_HEIGHT = 1.0  # camera height above the aircraft's reference point
RESPAWN_DELAY = 3.0  # seconds after a crash
CRASH_VIEW_DISTANCE = 60.0  # the crash is watched from this far behind
CRASH_VIEW_HEIGHT = 30.0  # ...and this high above the higher of camera ground and crash site

# Free camera.
GHOST_SPEED = 2.0  # world units per frame
GHOST_FAST = 5.0  # speed multiplier while SHIFT is held
GHOST_TURN = 0.04  # radians per frame
GHOST_CLEARANCE = 4.0  # minimum height above the ground
