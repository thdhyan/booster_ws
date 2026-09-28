"""Velocity-range curriculum: expand the commanded speed as tracking improves.

Deliberately free of Isaac imports so the logic is unit-testable on a laptop with
no GPU. It is a plain class taking ``(cfg, env)`` and called as ``term(env,
env_ids)``, which is exactly what Isaac Lab's curriculum manager does, so no
``@configclass`` is needed and none of the tuning has to be verified by eye.

WHY THIS EXISTS
---------------
P2's command range is fixed at +/-0.5 m/s (AGILE's deliberate choice: a biped that
cannot yet stand collapses if asked for more). That makes 0 -> 3 m/s unreachable
no matter how long training runs, because the policy is never asked to go faster
than 0.5 m/s and so never learns to. Only ``terrain_levels`` was curriculum'd.

WHAT THE REAL ROBOT TELLS US
----------------------------
Measured on K1 A2 over a 12-minute, 254 m walk: the factory walker driven by the
handheld remote at full stick peaked at about 1.3 m/s, with a fitted gain of
roughly -0.49 m/s per unit of left-Y stick. So 3 m/s is far beyond anything this
hardware has demonstrated, and 0 -> 1.5 m/s is the range with evidence behind it.
The ceiling here therefore defaults to 1.5 m/s. Raising it is a sim-only stretch
goal, and a sim result at 3 m/s must not be reported as deployable.

HOW IT EXPANDS
--------------
Gated on the running mean of the linear-velocity tracking reward, which is ~1 when
tracking is perfect. Expansion needs the mean to hold above ``success_threshold``
for ``patience`` consecutive checks; contraction is immediate and well below it,
so a policy that regresses gets help straight away instead of after a plateau.
"""

from __future__ import annotations

# Isaac Lab has renamed these attributes across versions, so probe instead of
# assuming one. Guessing wrong here silently does nothing, which is the exact
# failure mode this module exists to prevent.
_RANGE_ATTRS = (
    "lin_vel_x_range",
    "_lin_vel_x_range",
    "lin_vel_x",
    "_lin_vel_x",
)
_Y_ATTRS = ("lin_vel_y_range", "_lin_vel_y_range", "lin_vel_y", "_lin_vel_y")
_ANG_ATTRS = ("ang_vel_z_range", "_ang_vel_z_range", "ang_vel_z", "_ang_vel_z")


def find_range_attr(term):
    """Return the attribute name holding the lin_vel_x range, or raise."""
    for name in _RANGE_ATTRS:
        if getattr(term, name, None) is not None:
            return name
    raise AttributeError(
        f"cannot find the lin_vel_x range attribute on {type(term).__name__}; tried "
        f"{list(_RANGE_ATTRS)}. Update _RANGE_ATTRS for this Isaac Lab version rather "
        "than letting the curriculum silently do nothing."
    )


def _set_range(term, attrs, magnitude):
    for name in attrs:
        if getattr(term, name, None) is not None:
            setattr(term, name, (-magnitude, magnitude))
            return name
    return None


class VelocityRangeCurriculum:
    """Widens the commanded velocity range as the policy's tracking improves."""

    def __init__(
        self,
        env,
        init_lin_vel: float = 0.5,
        init_ang_vel: float = 1.0,
        target_max_lin_vel: float = 1.5,
        target_max_ang_vel: float = 2.0,
        step_lin_vel: float = 0.25,
        step_ang_vel: float = 0.25,
        success_threshold: float = 0.85,
        patience: int = 5,
        interval_steps: int = 50,
        reward_name_contains: str = "track_lin_vel",
    ):
        self._env = env
        self._init_lin = float(init_lin_vel)
        self._init_ang = float(init_ang_vel)
        self._lin = float(init_lin_vel)
        self._ang = float(init_ang_vel)
        self.target_max_lin = float(target_max_lin_vel)
        self.target_max_ang = float(target_max_ang_vel)
        self.step_lin = float(step_lin_vel)
        self.step_ang = float(step_ang_vel)
        self.threshold = float(success_threshold)
        self.patience = int(patience)
        self.interval_steps = int(interval_steps)
        self.reward_name_contains = reward_name_contains

        self._streak = 0
        self._since = 0
        self._applied = False
        self._attr = None
        self.expansions = 0
        self.contractions = 0
        self.peak_lin = float(init_lin_vel)

    # -- introspection --------------------------------------------------------
    def _command_term(self):
        return self._env.command_manager.get_command("base_velocity")

    def _reward_mean(self):
        """Mean of the velocity-tracking reward term, or None if not found."""
        mgr = self._env.reward_manager
        names = [n for n in mgr.active_terms if self.reward_name_contains in n]
        if not names:
            return None
        value = mgr.get_term(names[0]).mean()
        return float(value)

    def current_lin(self) -> float:
        return self._lin

    def current_ang(self) -> float:
        return self._ang

    # -- the actual work ------------------------------------------------------
    def _apply(self, term):
        self._attr = find_range_attr(term)
        _set_range(term, _Y_ATTRS, self._lin)
        _set_range(term, _ANG_ATTRS, self._ang)
        # lin_vel_x last so a failure above leaves the command axis untouched.
        _set_range(term, (self._attr,), self._lin)
        self._applied = True

    def __call__(self, env, env_ids):
        if not self._applied:
            self._apply(self._command_term())
            return

        self._since += 1
        if self._since < self.interval_steps:
            return
        self._since = 0

        mean_reward = self._reward_mean()
        if mean_reward is None:
            # Never widen on a missing signal: an absent term must not be read as
            # success, or the range would climb while nothing is being measured.
            self._streak = 0
            return

        term = self._command_term()
        if mean_reward >= self.threshold:
            self._streak += 1
            if self._streak >= self.patience:
                self._streak = 0
                before = self._lin
                self._lin = min(self._lin + self.step_lin, self.target_max_lin)
                self._ang = min(self._ang + self.step_ang, self.target_max_ang)
                self._apply(term)
                self.expansions += 1
                self.peak_lin = max(self.peak_lin, self._lin)
                _log(
                    env,
                    f"[vel-curriculum] tracking={mean_reward:.3f} >= {self.threshold} for "
                    f"{self.patience} checks: lin {before:.2f} -> {self._lin:.2f} m/s",
                )
        else:
            self._streak = 0
            if self._lin > self._init_lin:
                before = self._lin
                self._lin = max(self._lin - self.step_lin, self._init_lin)
                self._ang = max(self._ang - self.step_ang, self._init_ang)
                self._apply(term)
                self.contractions += 1
                _log(
                    env,
                    f"[vel-curriculum] tracking={mean_reward:.3f} < {self.threshold}: "
                    f"contracted lin {before:.2f} -> {self._lin:.2f} m/s",
                )


def _log(env, msg):
    logger = getattr(env, "logger", None)
    if logger is not None and hasattr(logger, "info"):
        logger.info(msg)
    else:
        print(msg)
