"""Keyboard velocity commands for the interactive play scripts.

WHY THIS EXISTS
---------------
``play_keyboard.py`` and ``play_keyboard_fixed.py`` read the keyboard through
``carb.input.acquire_input_interface()``. That attribute does not exist in Isaac
Sim 6, so both scripts die on their first frame with::

    AttributeError: module 'carb' has no attribute 'input'

which is the exact failure ``play_keyboard_fixed.py`` was already carrying a
``[DEBUG] carb.input available:`` print for. ``omni.*`` is no help either: those
modules only exist once the Kit app is running, and they have changed names
across releases.

So the keyboard is read from **X11** instead (python-xlib). It answers the only
question these scripts ask -- "is this key down right now" -- and it does so
independently of the Omniverse version, because the simulator window is an X11
window either way.

Backends, tried in order:

1. ``carb.input`` -- still correct on older Isaac Sim builds, so it is tried
   first and costs nothing.
2. X11 via python-xlib -- what Isaac Sim 6 needs.

If neither is available the controller reports no keys pressed rather than
raising: a window that stands still is better than a window that crashes.

The mapping and the ramp arithmetic live here, free of both Kit and X11, so they
are unit-testable on a laptop (``tests/test_keyboard_cmd.py``) instead of only
being verifiable by wiggling a key in a GUI.
"""

from __future__ import annotations

# key -> command axis (0 = forward/back, 1 = strafe, 2 = yaw)
KEY_AXIS = {
    "w": (0, +1),
    "s": (0, -1),
    "a": (1, +1),
    "d": (1, -1),
    "q": (2, +1),
    "e": (2, -1),
}


class KeyboardState:
    """Which keys are down right now, across whichever backend works."""

    def __init__(self, display_name: str | None = None):
        self.backend = "none"
        self._carb = None
        self._kb = None
        self._xdisplay = None
        self._codes: dict[str, int] = {}
        self._keymap: list[int] = []
        self._init_carb()
        if self.backend == "none":
            self._init_x11(display_name)

    # -- backends -------------------------------------------------------------
    def _init_carb(self) -> None:
        """Isaac Sim 4.x/5.x path. Absent on 6.x, which is the whole problem."""
        try:
            import carb  # type: ignore
        except Exception:
            return
        if not hasattr(carb, "input"):
            return
        try:
            self._kb = carb.input.acquire_input_interface().get_keyboard()
            self._carb = carb
            self.backend = "carb.input"
        except Exception:
            self._kb = None

    def _init_x11(self, display_name: str | None) -> None:
        try:
            from Xlib import XK, display  # type: ignore
        except Exception:
            return
        try:
            self._xdisplay = display.Display(display_name)
            from Xlib import XK as _XK  # noqa: F811  (local alias for the loop)

            self._codes = {
                name: self._xdisplay.keysym_to_keycode(_XK.string_to_keysym(name))
                for name in KEY_AXIS
            }
            self._keymap = list(self._xdisplay.query_keymap())
            self.backend = "x11"
        except Exception:
            self._xdisplay = None

    # -- queries --------------------------------------------------------------
    def is_pressed(self, key: str) -> bool:
        """True while ``key`` is held. Never raises."""
        key = key.lower()
        if self.backend == "carb.input":
            names = {
                "w": "W", "s": "S", "a": "A", "d": "D", "q": "Q", "e": "E",
                " ": "SPACE", "escape": "ESCAPE", "space": "SPACE",
            }
            code = self._carb.input.KeyboardInput.__getattr__(names.get(key, key.upper()))
            try:
                return bool(self._carb.input.acquire_input_interface().is_key_pressed(self._kb, code))
            except Exception:
                return False
        if self.backend == "x11":
            code = self._codes.get(key)
            if not code:
                return False
            try:
                # query_keymap is 32 bytes of bitmask; refresh every call so a
                # keypress between frames is still seen.
                km = self._xdisplay.query_keymap()
                self._keymap = list(km)
                return bool(km[code // 8] & (1 << (code % 8)))
            except Exception:
                return False
        return False

    def close(self) -> None:
        if self._xdisplay is not None:
            try:
                self._xdisplay.close()
            except Exception:
                pass
            self._xdisplay = None


class VelocityKeyboard:
    """Turns key state into a velocity command, ramping and decaying.

    Args:
        max_lin: cap on commanded linear speed (m/s).
        max_ang: cap on commanded yaw rate (rad/s).
        accel_lin: how fast a held key builds linear speed (m/s^2).
        accel_ang: how fast a held key builds yaw rate (rad/s^2).
        decay: per-step multiplier applied to an axis with no key held, so the
            command settles instead of stepping.
        on_quit: called when Escape is held.

    ``cmd`` is a plain 3-list ``[vx, vy, wz]`` so this class needs neither torch
    nor Kit; the caller wraps it in whatever tensor the env expects.
    """

    AXES = {"w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1),
            "q": (2, +1), "e": (2, -1)}

    def __init__(
        self,
        state: KeyboardState | None = None,
        max_lin: float = 1.5,
        max_ang: float = 2.0,
        accel_lin: float = 1.0,
        accel_ang: float = 1.0,
        decay: float = 0.95,
        on_quit=None,
    ):
        self.state = state if state is not None else KeyboardState()
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.accel_lin = float(accel_lin)
        self.accel_ang = float(accel_ang)
        self.decay = float(decay)
        self.on_quit = on_quit
        self.cmd = [0.0, 0.0, 0.0]

    def update(self, dt: float) -> list[float]:
        """Advance the command by ``dt`` seconds and return it.

        One decision per **axis**, not per key: the two keys on an axis are
        opposites, so handling them separately made the unheld one decay the axis
        the held one had just ramped (W gave 0.0475 m/s instead of 0.05 on the
        first frame, and a held S settled at -0.92 instead of reaching the cap).
        """
        for axis in (0, 1, 2):
            direction = 0
            for key, (key_axis, sign) in self.AXES.items():
                if key_axis == axis and self.state.is_pressed(key):
                    # Both directions held: first match wins, as the original
                    # if/elif chain did.
                    direction = sign
                    break
            if direction == 0:
                self.cmd[axis] *= self.decay
                continue
            cap = self.max_ang if axis == 2 else self.max_lin
            rate = self.accel_ang if axis == 2 else self.accel_lin
            step = rate * dt * direction
            self.cmd[axis] = min(max(self.cmd[axis] + step, -cap), cap)

        if self.state.is_pressed(" "):
            self.cmd = [0.0, 0.0, 0.0]
        if self.state.is_pressed("escape") and self.on_quit is not None:
            self.on_quit()
        return self.cmd

    def close(self) -> None:
        self.state.close()