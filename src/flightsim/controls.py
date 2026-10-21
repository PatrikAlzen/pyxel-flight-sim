"""Keyboard and gamepad input, mapped to flight controls and menu actions.

Every action works from the keyboard and from the first gamepad, so the rest
of the game never asks for a specific key.
"""

from enum import Enum, auto

import pyxel

from flightsim.sim.aircraft import Controls

# Analog sticks report -32768..32767; ignore small movements around centre.
AXIS_MAX = 32767
DEADZONE = 0.15


class Action(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    CONFIRM = auto()
    BACK = auto()
    PAUSE = auto()
    MAP = auto()
    RESPAWN = auto()
    GHOST = auto()
    DEBUG = auto()
    RANDOM = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    CAMERA = auto()
    SAVE_PLACE = auto()
    DELETE = auto()
    TIME_OF_DAY = auto()
    HUD_MODE = auto()


BINDINGS: dict[Action, tuple[int, ...]] = {
    Action.UP: (pyxel.KEY_UP, pyxel.GAMEPAD1_BUTTON_DPAD_UP),
    Action.DOWN: (pyxel.KEY_DOWN, pyxel.GAMEPAD1_BUTTON_DPAD_DOWN),
    Action.LEFT: (pyxel.KEY_LEFT, pyxel.GAMEPAD1_BUTTON_DPAD_LEFT),
    Action.RIGHT: (pyxel.KEY_RIGHT, pyxel.GAMEPAD1_BUTTON_DPAD_RIGHT),
    Action.CONFIRM: (pyxel.KEY_RETURN, pyxel.KEY_SPACE, pyxel.GAMEPAD1_BUTTON_A),
    Action.BACK: (pyxel.KEY_ESCAPE, pyxel.GAMEPAD1_BUTTON_B),
    Action.PAUSE: (pyxel.KEY_ESCAPE, pyxel.KEY_P, pyxel.GAMEPAD1_BUTTON_START),
    Action.MAP: (pyxel.KEY_M, pyxel.GAMEPAD1_BUTTON_X),
    Action.RESPAWN: (pyxel.KEY_R, pyxel.GAMEPAD1_BUTTON_BACK),
    Action.GHOST: (pyxel.KEY_G,),
    Action.DEBUG: (pyxel.KEY_F1,),
    Action.RANDOM: (pyxel.KEY_R, pyxel.GAMEPAD1_BUTTON_Y),
    Action.ZOOM_IN: (pyxel.KEY_Z, pyxel.GAMEPAD1_BUTTON_RIGHTSHOULDER),
    Action.ZOOM_OUT: (pyxel.KEY_X, pyxel.GAMEPAD1_BUTTON_LEFTSHOULDER),
    Action.CAMERA: (pyxel.KEY_C, pyxel.GAMEPAD1_BUTTON_Y),
    Action.SAVE_PLACE: (pyxel.KEY_L,),
    Action.DELETE: (
        pyxel.KEY_DELETE,
        pyxel.KEY_BACKSPACE,
        pyxel.KEY_X,
        pyxel.GAMEPAD1_BUTTON_X,
    ),
    Action.TIME_OF_DAY: (pyxel.KEY_T, pyxel.GAMEPAD1_BUTTON_RIGHTSTICK),
    Action.HUD_MODE: (pyxel.KEY_H, pyxel.GAMEPAD1_BUTTON_LEFTSTICK),
}


# Actions that repeat while held, like moving through a menu. All others fire
# once per press, so holding ESC does not flicker the pause menu on and off.
REPEATING = {
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.ZOOM_IN,
    Action.ZOOM_OUT,
}


def pressed(action: Action) -> bool:
    """True on the frame the action's key or button goes down."""
    if action in REPEATING:
        return any(pyxel.btnp(key, hold=15, repeat=4) for key in BINDINGS[action])
    return any(pyxel.btnp(key) for key in BINDINGS[action])


def axis(code: int) -> float:
    """Gamepad axis in [-1, 1] with the deadzone removed."""
    value = pyxel.btnv(code) / AXIS_MAX
    if abs(value) < DEADZONE:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * min(1.0, (abs(value) - DEADZONE) / (1 - DEADZONE))


def _keys(positive: int, negative: int) -> float:
    return float(pyxel.btn(positive) - pyxel.btn(negative))


def _combine(*values: float) -> float:
    return max(-1.0, min(1.0, sum(values)))


def flight_controls() -> Controls:
    """Current stick, rudder, throttle and brake input.

    Gamepad: left stick flies (pull back to climb), right stick X is the
    rudder, the shoulder buttons or triggers move the throttle and B brakes.
    """
    triggers = axis(pyxel.GAMEPAD1_AXIS_TRIGGERRIGHT) - axis(
        pyxel.GAMEPAD1_AXIS_TRIGGERLEFT
    )
    shoulders = _keys(
        pyxel.GAMEPAD1_BUTTON_RIGHTSHOULDER, pyxel.GAMEPAD1_BUTTON_LEFTSHOULDER
    )
    return Controls(
        pitch=_combine(
            _keys(pyxel.KEY_DOWN, pyxel.KEY_UP), axis(pyxel.GAMEPAD1_AXIS_LEFTY)
        ),
        roll=_combine(
            _keys(pyxel.KEY_RIGHT, pyxel.KEY_LEFT), axis(pyxel.GAMEPAD1_AXIS_LEFTX)
        ),
        yaw=_combine(_keys(pyxel.KEY_D, pyxel.KEY_A), axis(pyxel.GAMEPAD1_AXIS_RIGHTX)),
        throttle=_combine(_keys(pyxel.KEY_W, pyxel.KEY_S), shoulders, triggers),
        brake=bool(pyxel.btn(pyxel.KEY_B) or pyxel.btn(pyxel.GAMEPAD1_BUTTON_B)),
    )
