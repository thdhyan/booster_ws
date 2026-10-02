"""Ramp reward weights from light to terminal, gated on the robot actually standing.

WHY THIS EXISTS
---------------
We are lighter on every regularizer than AGILE is only at the *end* of training.
Measured on the 2026-10-01 run, our per-step penalty budget is 35.9 against AGILE
T1's 13.4 (2.68x), and 53% of it is our own gait shaping applied statically from
iteration 0. Meanwhile at iteration 12 the policy falls in 99.3% of episodes.

AGILE answers this with ``update_reward_weight_step``: it starts ``action_rate``
at -0.5 and ramps to -2.0 over steps 50k-150k, i.e. it deliberately runs *light*
while the policy learns to stand, then tightens. G1 carries the same idea as a
static value (``action_rate`` -0.01, 50x lighter than T1's) because G1 is mature.

See ``docs/agile_weight_comparison.md`` for the full extraction.

WHY THIS IS NOT AGILE'S CURRICULUM
-----------------------------------
AGILE ramps on ``env.common_step_counter`` alone. That is wrong for us, and the
reason is the measured failure mode rather than a matter of taste: our policy
falls in 99.3% of episodes, so on a step-based schedule a policy that is still
falling at step 3 000 gets handed the terminal smoothness penalty for not having
been smooth yet. Progress is therefore **credited only while the robot is
upright** -- the same signal ``VelocityRangeCurriculumTerm`` already gates on --
so the ramp self-delays until walking exists and only then charges for it.

AGILE's schedule also does not fit our run length. It is denominated in control
steps: AGILE's reference run reaches 1.2M, ours reach ~1.0M at 42 000 iterations,
but our earlier 3 000-iteration runs reached only 72 000 -- AGILE's 50k-150k ramp
would have been unstarted for the entire run.

Deliberately Isaac-free (mirroring ``velocity_curriculum``) so the arithmetic is
unit-testable on a laptop with no GPU, with a thin ``ManagerTermBase`` adapter
because ``CurriculumManager`` rejects plain functions as terms.
"""

from __future__ import annotations

# Isaac Lab hands the term its whole ManagerTermBaseCfg, so the values arrive
# nested under ``params`` rather than flat. Getting this wrong surfaces only
# after a full Isaac boot, so it is handled explicitly and covered by a test.
def _as_kwargs(cfg):
    if cfg is None:
        return {}
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


def _upright_ratio(env):
    """Fraction of a full episode the robot survives, or None if unmeasurable.

    Same gate as ``VelocityRangeCurriculumTerm``: standing is the precondition for
    everything else, and it rises smoothly as the policy improves.

    A missing measurement returns None and must never be read as success -- an
    absent signal must not advance a ramp, or the weight would climb while nothing
    is being measured.
    """
    buf = getattr(env, "episode_length_buf", None)
    max_len = getattr(env, "max_episode_length", None)
    if buf is None or not max_len:
        return None
    try:
        mean_len = float(buf.float().mean().item())
    except AttributeError:
        return None
    return mean_len / float(max_len)


class RewardWeightRamp:
    """Linearly interpolate one reward weight, crediting progress only when upright.

    The weight sits at ``start_weight`` until ``start_step``, then advances
    linearly toward ``terminal_weight`` over ``num_steps`` of *credited* steps. A
    step is credited only while the upright ratio is at or above
    ``upright_threshold``, so a policy that is still falling does not approach the
    terminal penalty.
    """

    def __init__(
        self,
        env,
        reward_name: str,
        start_weight: float,
        terminal_weight: float,
        start_step: int,
        num_steps: int,
        upright_threshold: float = 0.80,
        log_every: int = 1_000,
    ):
        if num_steps <= 0:
            raise ValueError(
                f"num_steps={num_steps} must be positive; a zero-length ramp never "
                "moves the weight and would look like a working curriculum"
            )
        if start_step < 0:
            raise ValueError(f"start_step={start_step} must be non-negative")
        if upright_threshold > 1.0 or upright_threshold < 0.0:
            raise ValueError(
                f"upright_threshold={upright_threshold} is not a fraction in [0, 1]"
            )
        # A ramp between two weights of opposite sign would pass through zero and
        # silently switch the term off mid-run.
        if (start_weight > 0) != (terminal_weight > 0):
            raise ValueError(
                f"ramp for '{reward_name}' goes from {start_weight} to "
                f"{terminal_weight}, crossing zero; the term would switch off "
                "partway through the run"
            )

        self._env = env
        self.reward_name = str(reward_name)
        self.start_weight = float(start_weight)
        self.terminal_weight = float(terminal_weight)
        self.start_step = int(start_step)
        self.num_steps = int(num_steps)
        self.upright_threshold = float(upright_threshold)
        self.log_every = int(log_every)

        self.credit = 0
        self.stalled_steps = 0
        self.applied = None
        self.last_upright = None

    # -- introspection --------------------------------------------------------
    def current_weight(self) -> float:
        return self.terminal_weight if self.credit >= self.num_steps else self._at(self.credit)

    def progress(self) -> float:
        return min(1.0, self.credit / self.num_steps)

    def _at(self, credited: int) -> float:
        scale = credited / self.num_steps
        return self.start_weight + (self.terminal_weight - self.start_weight) * scale

    # -- the actual work ------------------------------------------------------
    def _term_cfg(self):
        getter = getattr(self._env.reward_manager, "get_term_cfg", None)
        if getter is None:
            raise AttributeError(
                "RewardManager has no get_term_cfg, so the ramp cannot read or write "
                "the weight it is meant to schedule"
            )
        return getter(self.reward_name)

    def _apply(self, weight: float):
        """Write the weight and read it back.

        The read-back matters for the same reason ``VelocityRangeCurriculumTerm``
        does it: if the write silently missed, the run would look healthy for 42
        000 iterations while the term stayed at its static value.
        """
        cfg = self._term_cfg()
        cfg.weight = weight
        got = float(cfg.weight)
        if abs(got - weight) > 1e-9:
            raise RuntimeError(
                f"wrote {self.reward_name} weight {weight} but the reward manager "
                f"reports {got}; the ramp is inert"
            )
        self.applied = got

    def __call__(self, env, env_ids):
        step = int(getattr(env, "common_step_counter", 0))
        if step <= self.start_step:
            self._apply(self.start_weight)
            return None

        upright = _upright_ratio(env)
        self.last_upright = upright
        if upright is None or upright < self.upright_threshold:
            # No credit. Freezing the weight is the whole point: a falling policy
            # must not be charged for not being smooth.
            self.stalled_steps += 1
            return None

        self.credit += 1
        new_weight = self._at(self.credit)
        self._apply(new_weight)

        if self.log_every and self.credit % self.log_every == 0:
            self._log(
                env,
                f"[weight-ramp] {self.reward_name} {new_weight:.4f} "
                f"(step {step}, credit {self.credit}/{self.num_steps}, "
                f"upright {upright:.3f})",
            )
        return None

    def _log(self, env, msg):
        logger = getattr(env, "logger", None)
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg)


def resolve_name(value):
    """Reward names arrive as bare strings; reject a non-string loudly."""
    if not isinstance(value, str):
        raise ValueError(
            f"reward_name must be a string, got {type(value).__name__}. Passing the "
            "RewTerm node or a config object would silently never match a term."
        )
    return value


# ---------------------------------------------------------------------------
# Isaac Lab adapter
# ---------------------------------------------------------------------------
# CurriculumManager rejects any term whose ``func`` is not a ManagerTermBase
# subclass:
#   TypeError: Configuration for the term 'x' is not of type ManagerTermBase.
# So the config cannot point straight at the logic class. The parameter names
# below MUST match the config's params dict exactly and each must carry a
# default: _resolve_common_term_cfg does a static signature comparison and does
# not understand **kwargs.
try:  # pragma: no cover - only importable inside the Isaac container
    from isaaclab.managers import ManagerTermBase
except ImportError:  # pragma: no cover
    ManagerTermBase = None


if ManagerTermBase is not None:  # pragma: no cover - requires Isaac Lab

    class RewardWeightRampTerm(ManagerTermBase):
        """ManagerTermBase wrapper so Isaac Lab accepts the ramp term.

        Matches Isaac Lab's two-stage call convention: ``__init__`` runs once with
        keywords and the instance persists (which is what lets the credit counter
        survive between steps), then ``curriculum_manager.compute`` calls the
        instance with the params a second time -- accepted and ignored.
        """

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            kwargs = _as_kwargs(cfg)
            name = kwargs.get("reward_name")
            if name is None:
                raise ValueError(
                    "a weight ramp must name the reward term it schedules; without "
                    "reward_name there is nothing to write to"
                )
            self._impl = RewardWeightRamp(env, resolve_name(name), **{
                k: v for k, v in kwargs.items() if k != "reward_name"
            })

        @property
        def progress(self) -> float:
            return self._impl.progress()

        @property
        def credit(self) -> int:
            return self._impl.credit

        def __call__(
            self,
            env,
            env_ids,
            reward_name: str,
            start_weight: float,
            terminal_weight: float,
            start_step: int = 0,
            num_steps: int = 100_000,
            upright_threshold: float = 0.80,
            log_every: int = 1_000,
        ):
            self._impl(env, env_ids)
            return None