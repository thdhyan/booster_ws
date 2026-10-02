"""Tests for the keyboard velocity controller.

The controller is the part of the interactive session that used to be verifiable
only by wiggling a key in a GUI: it read ``carb.input``, which Isaac Sim 6 does
not have, so the whole path died on the first frame and the ramp arithmetic was
never exercised anywhere. It now lives in ``keyboard_cmd.py`` with no Kit and no
X11 dependency, so the behaviour is pinned here instead.

``FakeState`` stands in for a keyboard: the tests set exactly which keys are
down, which is the thing a real keyboard cannot do deterministically.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    _ROOT / "isaac_tasks/k1_velocity/scripts/keyboard_cmd.py"
)
_PLAY = _ROOT / "isaac_tasks/k1_velocity/scripts/play_keyboard_fixed.py"
PLAY = _ROOT / "isaac_tasks/k1_velocity/scripts/play.py"


def _load():
    spec = importlib.util.spec_from_file_location("keyboard_cmd", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


class FakeState:
    """KeyboardState look-alike with explicit key state and no X11."""

    backend = "fake"

    def __init__(self, held=()):
        self.held = {k.lower() for k in held}
        self.queries = 0

    def is_pressed(self, key):
        self.queries += 1
        return key.lower() in self.held

    def close(self):
        pass


def kb(held=(), dt=0.05, **kw):
    return mod.VelocityKeyboard(state=FakeState(held), **kw), dt


def test_forward_key_ramps_up_and_is_capped():
    k, dt = kb(held=("w",), max_lin=1.5, accel_lin=1.0)
    cmd = k.update(dt)
    assert cmd[0] == pytest.approx(0.05)
    for _ in range(200):
        cmd = k.update(dt)
    assert cmd[0] == pytest.approx(1.5), "holding W must stop at max_lin, not run away"


def test_backward_key_drives_negative():
    k, dt = kb(held=("s",))
    for _ in range(50):
        cmd = k.update(dt)
    assert cmd[0] < 0
    assert cmd[0] == pytest.approx(-1.5)


def test_release_decays_to_zero():
    k, dt = kb(held=("w",))
    for _ in range(40):
        k.update(dt)
    state = k.state
    state.held.clear()  # let go
    for _ in range(400):
        cmd = k.update(dt)
    # 0.95^400 is ~1e-9 times the start, so "settled" has to mean "negligible",
    # not "exactly zero": the decay is multiplicative by design and never snaps.
    assert cmd[0] == pytest.approx(0.0, abs=1e-4), (
        "a released key must settle the command, not hold the last speed"
    )


def test_strafe_and_yaw_use_their_own_axes_and_caps():
    k, dt = kb(held=("a", "q"), max_lin=1.0, max_ang=0.5, accel_lin=1.0, accel_ang=1.0)
    for _ in range(200):
        cmd = k.update(dt)
    assert cmd[1] == pytest.approx(1.0), "A drives lateral (axis 1)"
    assert cmd[2] == pytest.approx(0.5), "Q drives yaw (axis 2) and stops at max_ang"
    assert cmd[0] == pytest.approx(0.0), "no forward key, so vx stays zero"


def test_opposite_keys_resolve_to_one_direction():
    """Both keys on an axis: the first match wins, as the original if/elif did."""
    k, dt = kb(held=("w", "s"))
    for _ in range(20):
        cmd = k.update(dt)
    assert cmd[0] > 0, "W wins over S; the command must not oscillate or cancel"


def test_space_zeroes_the_command():
    k, dt = kb(held=("w", " "))
    for _ in range(30):
        cmd = k.update(dt)
    assert cmd == [0.0, 0.0, 0.0], "Space is the stop key and must win over a held direction"


def test_escape_triggers_quit_once_configured():
    fired = []
    k, dt = kb(held=("escape",), on_quit=lambda: fired.append(1))
    k.update(dt)
    assert fired, "Escape must call on_quit so the session can close"


def test_no_quit_callback_means_no_crash():
    k, dt = kb(held=("escape",))
    k.update(dt)  # on_quit is None; must not raise


def test_dead_backend_reports_nothing_pressed():
    """A missing backend must not raise: a still robot beats a crashed window."""

    class Dead:
        backend = "none"

        def is_pressed(self, key):
            return False

        def close(self):
            pass

    k = mod.VelocityKeyboard(state=Dead())
    cmd = k.update(0.05)
    assert cmd == [0.0, 0.0, 0.0]


def test_teacher_play_script_has_keyboard_control():
    """play.py must be able to drive a *teacher* checkpoint from the keyboard.

    The student keyboard script loads a DistillationRunner, so it cannot open a
    teacher (237-dim privileged) checkpoint at all. Without --keyboard here the
    only way to eyeball a teacher policy was a fixed command in the config, which
    is exactly the wrong tool for judging whether it can be steered.
    """
    src = PLAY.read_text()
    assert '"--keyboard"' in src, "play.py needs a --keyboard flag"
    assert "keyboard_cmd.VelocityKeyboard" in src, "it must use the shared X11 controller"

    # Attribute check, not a substring: the explanatory comment names the method
    # that no longer exists, and naming it is the point of the comment.
    import ast

    attrs = {
        node.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute)
    }
    assert "set_command" not in attrs, (
        "CommandManager.set_command was removed in Isaac Lab 2.1+; write into the "
        "tensor from get_command() instead"
    )


def test_play_script_uses_the_shared_controller_not_carb_input():
    """Regression: the scripts died on ``carb.input`` under Isaac Sim 6.

    Checked on the AST rather than the raw text so the explanatory comment that
    names ``carb.input`` does not trip the guard -- only real code counts.
    """
    import ast

    src = _PLAY.read_text()
    assert "import keyboard_cmd" in src, "the play script must use the shared controller"
    assert "keyboard_cmd.VelocityKeyboard" in src

    calls = {
        node.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute)
    }
    assert "acquire_input_interface" not in calls, (
        "carb.input.acquire_input_interface does not exist in Isaac Sim 6; calling "
        "it raises AttributeError on the first frame"
    )