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

# How to reach the object that actually owns the command ranges.
#
# Isaac Lab's UniformVelocityCommand samples from **cfg**, not from instance
# attributes: _resample_command does `r.uniform_(*self.cfg.ranges.lin_vel_x)`.
# So widening means mutating `term.cfg.ranges`, and probing the term for an
# attribute like `lin_vel_x_range` finds nothing -- which is why a launch failed
# with "cannot find the lin_vel_x range attribute on Tensor".
#
# Note also that `command_manager.get_command(name)` returns the command *tensor*,
# not the term. The term comes from `command_manager.get_term(name)`.
def resolve_ranges(term):
    """Return the mutable ranges object of a command term, or raise."""
    cfg = getattr(term, "cfg", None)
    ranges = getattr(cfg, "ranges", None) if cfg is not None else None
    if ranges is None and isinstance(term, dict):
        ranges = term.get("ranges")
    if ranges is None:
        raise AttributeError(
            f"cannot reach the command ranges on {type(term).__name__}: expected "
            "term.cfg.ranges, because UniformVelocityCommand samples from "
            "self.cfg.ranges.lin_vel_x. Refusing to guess -- guessing wrong leaves "
            "the range unchanged and the curriculum silently does nothing."
        )
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        if not hasattr(ranges, axis):
            raise AttributeError(
                f"command ranges object {type(ranges).__name__} has no '{axis}'; "
                "the velocity command layout changed and widening would be a no-op"
            )
    return ranges


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
        self.expansions = 0
        self.contractions = 0
        self.peak_lin = float(init_lin_vel)

    # -- introspection --------------------------------------------------------
    def _command_term(self):
        """The base_velocity CommandTerm (not the command tensor)."""
        mgr = self._env.command_manager
        getter = getattr(mgr, "get_term", None)
        if getter is not None:
            return getter("base_velocity")
        return mgr._terms["base_velocity"]

    def _reward_mean(self):
        """Mean RAW value of the velocity-tracking reward term, or None.

        Two Isaac Lab details matter here and both cost a launch:

        * ``RewardManager`` has **no** ``get_term``. The per-step values live in
          a ``_step_reward`` matrix of shape (num_envs, num_terms), indexed by
          position in ``_term_names``. Calling ``get_term`` raises
          ``AttributeError: 'RewardManager' object has no attribute 'get_term'``.
        * ``_step_reward`` holds ``func(...) * weight``, i.e. the **weighted**
          value. ``track_lin_vel_xy_exp`` has weight 10.0, so the raw exponential
          has to be recovered by dividing by the weight or the 0.85 threshold
          below is meaningless (it would compare against ~8.5).

        The public ``get_active_iterable_terms`` is used as a fallback, but it
        loops per env and still returns weighted values.
        """
        mgr = self._env.reward_manager
        names = list(getattr(mgr, "active_terms", []) or [])
        target = next((n for n in names if self.reward_name_contains in n), None)
        if target is None:
            return None

        weight = 1.0
        cfg_getter = getattr(mgr, "get_term_cfg", None)
        if cfg_getter is None:
            raise AttributeError(
                "RewardManager has no get_term_cfg, so the reward weight is unknown. "
                "The gate threshold is expressed against the RAW term value, so "
                "guessing the weight would compare a weighted value to a raw "
                "threshold and silently mis-gate the curriculum."
            )
        cfg = cfg_getter(target)
        raw_weight = getattr(cfg, "weight", None)
        if not raw_weight:
            raise ValueError(
                f"reward term '{target}' has weight {raw_weight!r}; cannot recover the "
                "raw term value the success_threshold is expressed against"
            )
        weight = float(raw_weight)

        values = None
        step_reward = getattr(mgr, "_step_reward", None)
        term_names = getattr(mgr, "_term_names", None)
        if step_reward is not None and term_names is not None and target in term_names:
            idx = list(term_names).index(target)
            col = step_reward[:, idx]
            values = float(col.mean().item())
        else:
            getter = getattr(mgr, "get_active_iterable_terms", None)
            if getter is not None:
                for name, vals in getter(0):
                    if name == target:
                        values = float(vals[0])
                        break
        if values is None:
            return None
        return values / weight

    def current_lin(self) -> float:
        return self._lin

    def current_ang(self) -> float:
        return self._ang

    # -- the actual work ------------------------------------------------------
    def _apply(self, term):
        """Write the widened range into the term's config, then verify it stuck.

        The read-back matters: if the config object were immutable or the write
        silently missed, the run would look perfectly healthy for 3000 iterations
        while never leaving the starting range.
        """
        ranges = resolve_ranges(term)
        ranges.lin_vel_x = (-self._lin, self._lin)
        ranges.lin_vel_y = (-self._lin, self._lin)
        ranges.ang_vel_z = (-self._ang, self._ang)
        got = ranges.lin_vel_x[1]
        if abs(got - self._lin) > 1e-6:
            raise RuntimeError(
                f"wrote lin_vel_x=+/-{self._lin} but the command term still reports "
                f"{got}; the range is not being applied and the curriculum is inert"
            )
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


# ---------------------------------------------------------------------------
# Isaac Lab adapter
# ---------------------------------------------------------------------------
# isaaclab.managers.CurriculumManager rejects any term whose ``func`` is not a
# ManagerTermBase subclass:
#
#   TypeError: Configuration for the term 'velocity_range' is not of type
#   ManagerTermBase. Received: '<class 'type'>'.
#
# So the config cannot point straight at the logic class. The adapter is a thin
# shell and the import is guarded, which is what lets the logic above stay
# importable -- and therefore testable -- on a machine with no Isaac Sim.
try:  # pragma: no cover - only importable inside the Isaac container
    from isaaclab.managers import ManagerTermBase
except ImportError:  # pragma: no cover
    ManagerTermBase = None


def _as_kwargs(cfg):
    """Normalise a CurrTerm cfg into kwargs for the logic class.

    Isaac Lab hands the term its whole ``ManagerTermBaseCfg``, so the values we
    care about arrive **nested under ``params``** rather than flat. Getting this
    wrong shows up as::

        TypeError: VelocityRangeCurriculum.__init__() got an unexpected
        keyword argument 'params'

    which is only visible after a full Isaac boot, so it is handled explicitly
    and covered by a test.
    """
    if cfg is None:
        return {}
    # Most common: an object (or dict) carrying a nested params mapping.
    nested = getattr(cfg, "params", None)
    if nested is None and isinstance(cfg, dict):
        nested = cfg.get("params")
    if nested is not None and hasattr(nested, "items"):
        return {k: v for k, v in nested.items() if not str(k).startswith("_")}
    if isinstance(cfg, dict):
        return {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    if hasattr(cfg, "items"):
        return {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    return {
        k: v
        for k, v in vars(cfg).items()
        if not str(k).startswith("_") and not callable(v) and not isinstance(v, type)
    }


if ManagerTermBase is not None:  # pragma: no cover - requires Isaac Lab

    # The parameter names below MUST match the keys in the ``velocity_range``
    # CurrTerm params dict exactly, and each must carry a default. Isaac Lab's
    # manager_base._resolve_common_term_cfg does a purely static signature
    # comparison and does not understand **kwargs, so a generic signature is
    # rejected:
    #
    #   ValueError: The term 'velocity_range' expects mandatory parameters:
    #   ['kwargs'] and optional parameters: [], but received: ['init_lin_vel', ...]
    #
    # That is why the list is spelled out instead of using **kwargs. The
    # duplication is forced by the API, so it is pinned by a test that compares
    # this signature against the config's params dict.
    class VelocityRangeCurriculumTerm(ManagerTermBase):
        """ManagerTermBase wrapper so Isaac Lab accepts the curriculum term.

        Isaac Lab's two-stage call convention, matched exactly:

        1. ``manager_base._prepare_terms`` replaces the class with a single
           instance::

               term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

           so ``__init__`` runs once, with keywords, and the instance persists --
           which is what lets the streak counter and widened range survive between
           steps.

        2. ``curriculum_manager.compute`` then calls that instance::

               state = term_cfg.func(self._env, env_ids, **term_cfg.params)

           The params therefore arrive twice. The instance is already configured,
           so they are accepted and ignored here rather than used to re-init.
        """

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._impl = VelocityRangeCurriculum(env, **_as_kwargs(cfg))

        # Exposed so a probe or launch script can read the live range.
        @property
        def current_lin(self) -> float:
            return self._impl.current_lin()

        @property
        def current_ang(self) -> float:
            return self._impl.current_ang()

        @property
        def expansions(self) -> int:
            return self._impl.expansions

        def __call__(
            self,
            env,
            env_ids,
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
            self._impl(env, env_ids)
            return None
