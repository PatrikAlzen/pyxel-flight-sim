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
MAX_WRECKS = 100  # old crash sites kept in the world; the oldest go first
WRECK_DRAW_DISTANCE = 600.0  # wrecks further away are too small to see
WRECK_SMOKE_DISTANCE = 1500.0  # old wrecks this close smoulder, visible from afar

# Saved data (wrecks, places) per seed, under Pyxel's user data directory.
SAVE_VENDOR = "pyxel-flight-sim"
PLACE_MIN_CLEARANCE = 20.0  # a saved place never drops the plane closer to the ground

# Airfields.
FIELD_SEARCH_RINGS = 3  # cells searched around the plane for the nearest field
RUNWAY_START_OFFSET = 20.0  # starting on a runway: this far in from its end
PROP_DRAW_DISTANCE = 900.0  # hangars and windsocks further away are not drawn
HANGAR_WIDTH = 36.0  # world units
WINDSOCK_WIDTH = 4.0
OVERCAST_STEPS = 8  # palette rebuilds as the sky greys over
TURBULENCE_PROBE = 20.0  # how far around the plane the ground slope is measured

# Trees, houses and lighthouses.
OBJECT_DRAW_DISTANCE = 300.0  # trees and houses beyond this are not drawn
OBJECT_LIMIT = 800  # at most this many objects per frame, the nearest ones
LIGHTHOUSE_DRAW_DISTANCE = 2000.0  # landmarks, so visible much further out
LAMP_PERIOD = 2.5  # seconds between lighthouse flashes...
LAMP_ON = 0.5  # ...each lasting this long
DISCOVER_DISTANCE = 250.0  # flying this close to a lighthouse discovers it
COLLISION_SEARCH = 10.0  # objects within this of the plane are tested for hits
MAP_SETTLEMENT_ZOOM = 16  # villages and lighthouses on the map up to 1:16

# Free camera.
GHOST_SPEED = 2.0  # world units per frame
GHOST_FAST = 5.0  # speed multiplier while SHIFT is held
GHOST_TURN = 0.04  # radians per frame
GHOST_CLEARANCE = 4.0  # minimum height above the ground

# Photo mode.
PHOTO_FOVS = (90.0, 70.0, 50.0, 35.0, 25.0)  # degrees, Z/X step through them
PHOTO_SCALE = 3  # saved image is this many times the screen size
PHOTO_FLASH = 0.2  # seconds of white after taking a photo

# Replays.
REPLAY_MIN = 1.0  # seconds recorded before there is anything to replay
REPLAY_SEEK = 5.0  # seconds per press of left or right
REPLAY_RELOAD = 2.0  # camera jumps beyond this many times the finest level's reach load

# Bridges.
BRIDGE_DRAW_DISTANCE = 1500.0  # bridges further away are not drawn
BRIDGE_SEARCH = 200.0  # bridges this close are checked for collisions and passes

# Traffic.
MIDAIR_DISTANCE = 0.7  # of the two planes' half-spans added: a collision
NEAR_MISS_DISTANCE = 30.0  # closer than this without colliding: a near miss

# Achievements.
AWARD_SECONDS = 3.0  # each new achievement is announced this long
RIVER_CHECK_FRAMES = 10  # how often the river under the plane is looked up
RIVER_VALLEY = 0.5  # valley depth (0..1) that counts as over a river
PROGRESS_SAVE_FRAMES = 600  # progress (distances, counts) is saved this often

# Fuel.
REFUEL_MAX_SPEED = 1.0  # the plane must stand (nearly) still to refuel
