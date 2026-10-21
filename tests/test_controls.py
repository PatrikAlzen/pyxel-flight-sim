import pytest
import pyxel

from flightsim import controls
from flightsim.controls import AXIS_MAX, Action


@pytest.fixture
def fake_input(monkeypatch):
    """Replace Pyxel's input functions with ones driven by a dict of held keys/axes."""
    state: dict[int, int] = {}
    monkeypatch.setattr(pyxel, "btn", lambda key: bool(state.get(key)))
    monkeypatch.setattr(pyxel, "btnv", lambda key: state.get(key, 0))
    monkeypatch.setattr(pyxel, "btnp", lambda key, **kwargs: bool(state.get(key)))
    return state


def test_no_input_means_neutral_controls(fake_input):
    c = controls.flight_controls()
    assert (c.pitch, c.roll, c.yaw, c.throttle, c.brake) == (0, 0, 0, 0, False)


def test_keyboard_flight_controls(fake_input):
    fake_input.update(
        {pyxel.KEY_DOWN: 1, pyxel.KEY_LEFT: 1, pyxel.KEY_D: 1, pyxel.KEY_W: 1}
    )
    c = controls.flight_controls()
    assert c.pitch == 1  # pull back
    assert c.roll == -1
    assert c.yaw == 1
    assert c.throttle == 1


def test_gamepad_sticks_and_deadzone(fake_input):
    fake_input[pyxel.GAMEPAD1_AXIS_LEFTY] = AXIS_MAX  # stick pulled fully back
    fake_input[pyxel.GAMEPAD1_AXIS_LEFTX] = -AXIS_MAX // 2
    fake_input[pyxel.GAMEPAD1_AXIS_RIGHTX] = int(AXIS_MAX * 0.1)  # inside deadzone
    c = controls.flight_controls()
    assert c.pitch == pytest.approx(1.0)
    assert -0.5 < c.roll < -0.3
    assert c.yaw == 0.0


def test_gamepad_throttle_and_brake(fake_input):
    fake_input[pyxel.GAMEPAD1_AXIS_TRIGGERRIGHT] = AXIS_MAX
    fake_input[pyxel.GAMEPAD1_BUTTON_B] = 1
    c = controls.flight_controls()
    assert c.throttle == 1.0
    assert c.brake

    fake_input.clear()
    fake_input[pyxel.GAMEPAD1_BUTTON_LEFTSHOULDER] = 1
    assert controls.flight_controls().throttle == -1.0


def test_keyboard_and_gamepad_are_combined_and_clamped(fake_input):
    fake_input[pyxel.KEY_RIGHT] = 1
    fake_input[pyxel.GAMEPAD1_AXIS_LEFTX] = AXIS_MAX
    assert controls.flight_controls().roll == 1.0


@pytest.mark.parametrize(
    ("key", "action"),
    [
        (pyxel.KEY_ESCAPE, Action.PAUSE),
        (pyxel.GAMEPAD1_BUTTON_START, Action.PAUSE),
        (pyxel.KEY_M, Action.MAP),
        (pyxel.GAMEPAD1_BUTTON_X, Action.MAP),
        (pyxel.KEY_RETURN, Action.CONFIRM),
        (pyxel.GAMEPAD1_BUTTON_A, Action.CONFIRM),
        (pyxel.GAMEPAD1_BUTTON_DPAD_DOWN, Action.DOWN),
        (pyxel.KEY_H, Action.HUD_MODE),
    ],
)
def test_actions_work_from_keyboard_and_gamepad(fake_input, key, action):
    assert not controls.pressed(action)
    fake_input[key] = 1
    assert controls.pressed(action)


def test_only_navigation_repeats(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pyxel, "btnp", lambda key, **kwargs: calls.append(kwargs) or False
    )
    controls.pressed(Action.DOWN)
    assert all(kw.get("repeat") for kw in calls)
    calls.clear()
    controls.pressed(Action.PAUSE)
    assert not any(kw.get("repeat") for kw in calls)


def test_hud_mode_cycles_full_minimap_none():
    from flightsim.render.hud import HudMode

    seen = [HudMode.FULL]
    for _ in range(3):
        seen.append(seen[-1].next())
    assert seen == [HudMode.FULL, HudMode.MINIMAP, HudMode.NONE, HudMode.FULL]
