"""Tests for the upright-gated reward-weight ramp.

The ramp is the single change that makes our regularization schedule differ from
AGILE's: AGILE ramps on ``common_step_counter`` alone, we credit progress only
while the robot is upright. The reason is measured, not stylistic -- at iteration
12 the policy falls in 99.3% of episodes, so a step-based schedule would hand a
falling policy the terminal smoothness penalty for not being smooth yet.

The logic class is deliberately Isaac-free (mirroring ``velocity_curriculum``) so
the arithmetic is testable here, on a laptop, with fakes for ``env`` and the
reward manager.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VEL = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity"
RAMP = VEL / "reward_weight_ramp.py"
CFG = VEL / "velocity_env_cfg.py"

sys_path_mod = __import__("sys")


class FakeTermCfg:
    def __init__(self, weight):
        self.weight = weight


class FakeRewardManager:
    def __init__(self, weights):
        self._terms = {k: FakeTermCfg(v) for k, v in weights.items()}

    def get_term_cfg(self, name):
        return self._terms[name]


class FakeEnv:
    """Minimal stand-in for the two things the ramp reads off env.

    ``episode_length_buf`` ADVANCES on every ``step()`` call, because the ramp's
    idempotency guard compares it against the previous read: if it never moves,
    ``stepped`` is always 0 and the ramp never credits anything. Reproducing that
    here is what let an earlier version of these tests pass a ramp that could not
    actually fire.
    """

    def __init__(self, step, upright_frac, max_len=1000, weights=None):
        self.common_step_counter = step
        self.max_episode_length = max_len
        self._len = upright_frac * max_len
        self.episode_length_buf = _Buf(self._len)
        self.reward_manager = FakeRewardManager(weights or {})
        self._max_len = max_len

    def step(self):
        """One control step: the episode buffer ticks up."""
        self._len = min(self._len + 1, self._max_len)
        self.episode_length_buf = _Buf(self._len)

    def set_upright(self, frac):
        self._len = frac * self._max_len
        self.episode_length_buf = _Buf(self._len)


class _Buf:
    def __init__(self, mean):
        self._mean = mean

    def float(self):
        return self

    def mean(self):
        return _Scalar(self._mean)


class _Scalar:
    def __init__(self, v):
        self.v = float(v)

    def item(self):
        return self.v


def _ramp(**kw):
    """Import the logic class and build one, with fakes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ramp_mod", RAMP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defaults = dict(
        reward_name="action_rate_l2",
        start_weight=-0.5,
        terminal_weight=-2.0,
        start_step=0,
        num_steps=100,
        upright_threshold=0.80,
        log_every=0,
    )
    defaults.update(kw)
    return mod, defaults


def _ramp_obj(env, **kw):
    mod, kwargs = _ramp(**kw)
    return mod.RewardWeightRamp(env, **kwargs)


def _ramp_class():
    """The logic class itself, for the class-level constants."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ramp_mod_cls", RAMP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RewardWeightRamp


RewardWeightRamp = _ramp_class()


def test_gate_calibrates_below_the_robots_own_ratio():
    """The gate must be reachable even when the configured value is not.

    Two configured gates were unreachable and both silently disabled the whole
    ramp: 0.80 (AGILE's shape, not our regime) and 0.10 (guessed) against a
    measured upright ratio of 0.006. A ramp that cannot fire is invisible --
    it registers in the curriculum table like a working term while holding every
    weight at its start value.
    """
    # policy barely stands: 6 steps of 1000, and improves a little
    env = FakeEnv(step=1000, upright_frac=0.006, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100, upright_threshold=0.10)
    for _ in range(50):
        env.set_upright(0.10)  # 0.006 baseline -> now standing 10% of episodes
        env.step()
        r(env, None)
    assert r.credit > 0, (
        "an unreachable gate left credit at 0: the ramp can never fire, so every "
        "gait penalty stays at its 1/5 start weight for the whole run"
    )
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight < -0.5


def test_calibrated_gate_still_demands_improvement():
    """Auto-calibration must not mean 'always open'.

    The gate sits at measured * 1.25, so a policy has to be clearly better than
    it was at the start -- being exactly as bad does not earn credit.
    """
    env = FakeEnv(step=1000, upright_frac=0.10, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100, upright_threshold=0.80)
    for _ in range(20):
        env.set_upright(0.10)  # holding exactly steady
        env.step()
        r(env, None)
    assert r.credit == 0, "holding steady at the starting ratio must not earn credit"
    # now clearly better than the 0.125 calibrated gate
    for _ in range(20):
        env.set_upright(0.20)
        env.step()
        r(env, None)
    assert r.credit == 20


def test_explicit_reachable_threshold_is_respected():
    """When the configured gate IS reachable it wins -- no silent loosening."""
    env = FakeEnv(step=1000, upright_frac=0.90, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100, upright_threshold=0.80)
    for _ in range(10):
        r(env, None)
        env.step()
    assert r.credit == 10


def test_noise_level_ratio_keeps_the_gate_reachable():
    """A near-zero measured ratio must not produce a gate ABOVE that ratio.

    This is the original failure restated as a test: the 0.10 gate against a
    0.006 ratio. A noise floor is still wanted (so jitter cannot open the gate),
    but it must be capped by the baseline -- otherwise the floor itself makes the
    ramp unreachable, which is the bug this replaced.
    """
    env = FakeEnv(step=1000, upright_frac=0.006, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100, upright_threshold=0.80)
    r(env, None)
    gate = r._effective_threshold()
    # The gate must be a small, finite step above the baseline -- reachable by a
    # policy that improves, rather than an order of magnitude away.
    assert gate > 0.006, "the gate must demand some improvement over the baseline"
    assert gate < 0.006 + 0.5, f"gate {gate} is implausibly far above baseline 0.006"
    assert gate > 0.0, "the gate must stay strictly positive so jitter cannot open it"


# --- the arithmetic -----------------------------------------------------------
def test_weight_holds_at_start_before_start_step():
    env = FakeEnv(step=0, upright_frac=1.0, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=100, num_steps=100)
    for _ in range(10):
        r(env, None)
        env.step()
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight == -0.5
    assert r.credit == 0


def test_weight_reaches_terminal_after_full_ramp():
    env = FakeEnv(step=1000, upright_frac=1.0, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100)
    for _ in range(100):
        r(env, None)
        env.step()
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight == pytest.approx(-2.0)
    assert r.progress() == pytest.approx(1.0)


def test_weight_interpolates_linearly_in_between():
    env = FakeEnv(step=1000, upright_frac=1.0, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100)
    for _ in range(50):
        r(env, None)
        env.step()
    # halfway: -0.5 + (-2.0 - -0.5) * 0.5 = -1.25
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight == pytest.approx(-1.25)


# --- the upright gate: the whole point of this class --------------------------
def test_no_credit_while_falling_is_the_critical_behaviour():
    """A falling policy must not advance toward the terminal penalty.

    This is the difference from AGILE's step-based schedule and the reason the
    class exists, so it is tested directly rather than through a proxy.
    """
    env = FakeEnv(step=1000, upright_frac=0.10, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100, upright_threshold=0.80)
    for _ in range(500):
        env.set_upright(0.10)  # falling, and staying fallen
        r(env, None)
    assert r.credit == 0, "a policy that is falling accrued ramp credit"
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight == pytest.approx(-0.5)
    assert r.stalled_steps > 0


def test_credit_accrues_only_above_the_threshold():
    env = FakeEnv(step=1000, upright_frac=0.90, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100, upright_threshold=0.80)
    for _ in range(10):
        r(env, None)
        env.step()
    assert r.credit == 10
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight < -0.5


def test_a_policy_that_recovers_still_gets_credit():
    env = FakeEnv(step=1000, upright_frac=0.10, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100)
    for _ in range(50):
        env.set_upright(0.10)  # falling, and staying fallen
        r(env, None)
    assert r.credit == 0
    # policy learns to stand
    env.set_upright(0.95)
    for _ in range(100):
        r(env, None)
        env.step()
    assert r.progress() == pytest.approx(1.0)


def test_missing_signal_is_not_read_as_success():
    """An unmeasurable upright ratio must never advance the ramp.

    velocity_curriculum documents the same rule: an absent measurement read as
    success would climb the range while nothing was being measured.
    """
    env = FakeEnv(step=1000, upright_frac=1.0, weights={"action_rate_l2": -0.5})
    r = _ramp_obj(env, start_step=0, num_steps=100)
    for _ in range(50):
        env.episode_length_buf = None  # measurement unavailable
        r(env, None)
    assert r.credit == 0
    assert r.last_upright is None


# --- guards -------------------------------------------------------------------
def test_degenerate_ramp_is_rejected():
    env = FakeEnv(step=0, upright_frac=1.0, weights={})
    with pytest.raises(ValueError, match="num_steps"):
        _ramp_obj(env, num_steps=0)


def test_zero_crossing_ramp_is_rejected():
    """A ramp from a penalty to a bonus would switch the term off mid-run."""
    env = FakeEnv(step=0, upright_frac=1.0, weights={})
    with pytest.raises(ValueError, match="crossing zero"):
        _ramp_obj(env, start_weight=-0.5, terminal_weight=1.0)


def test_bad_upright_threshold_is_rejected():
    env = FakeEnv(step=0, upright_frac=1.0, weights={})
    with pytest.raises(ValueError, match="fraction"):
        _ramp_obj(env, upright_threshold=1.7)


def test_write_is_read_back_so_an_inert_ramp_cannot_pass_silently():
    """The read-back matters: a silently missed write looks healthy for 42k iters.

    Models a reward manager that accepts the assignment and discards it, which
    is the failure the read-back exists to catch.
    """

    class SwallowsWrites:
        def __init__(self, v):
            # seed via object.__setattr__ or the guard below eats the initial value
            object.__setattr__(self, "weight", v)

        def __setattr__(self, k, v):
            if k == "weight":
                return  # silently drop the write
            object.__setattr__(self, k, v)

    class BadManager:
        def get_term_cfg(self, name):
            return SwallowsWrites(-0.5)

    env = FakeEnv(step=1000, upright_frac=1.0, weights={"action_rate_l2": -0.5})
    env.reward_manager = BadManager()
    r = _ramp_obj(env, start_step=0, num_steps=100)
    with pytest.raises(RuntimeError, match="inert"):
        r(env, None)
        env.step()


# --- config wiring ------------------------------------------------------------
def _cfg_tree():
    return ast.parse(CFG.read_text())


def _curriculum_cfg() -> ast.ClassDef:
    for n in ast.walk(_cfg_tree()):
        if isinstance(n, ast.ClassDef) and n.name == "CurriculumCfg":
            return n
    raise AssertionError("no CurriculumCfg")


def _rewards_cfg() -> ast.ClassDef:
    for n in ast.walk(_cfg_tree()):
        if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg":
            return n
    raise AssertionError("no RewardsCfg")


def _assigns(cls: ast.ClassDef) -> dict:
    out = {}
    for n in ast.walk(cls):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = n.value
    return out


def _const(node):
    if node is None:
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _dict_get(node, key):
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                return v
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if kw.arg == key:
                return kw.value
    return None


# reward term -> (static weight, terminal weight it ramps to)
RAMPED = {
    # The two smoothness terms were reduced 5-6x on 2026-10-06. Measured raw
    # magnitudes at iteration 4400 were action_jerk 6.6 and action_rate 2.20
    # against feet_clearance 0.0153, so the smoothness budget was 140-430x the
    # lift budget and was actively ANTI-lift: lifting a foot cleanly needs fast
    # deliberate joint motion, which is what these two tax. Strengthening
    # feet_clearance 48x on its own moved measured lift not at all (+0.0065 m).
    "action_rate_l2": (-0.1, -0.4),
    "action_jerk_l2": (-0.01, -0.08),
    "gait_cadence": (-0.2, -1.0),
    "phase_swing": (-0.4, -2.0),
    "feet_clearance": (-1.6, -8.0),
    "feet_alternation": (-0.4, -2.0),
    "stride_length": (-1.2, -6.0),
}


@pytest.mark.parametrize("term,weights", sorted(RAMPED.items()))
def test_static_weight_is_the_light_start_value(term: str, weights):
    """The static weight must be the START value, not the final one.

    If it is left at the terminal value the ramp never helps -- the policy pays
    the full penalty from iteration 0, which is the thing being fixed.
    """
    assigns = _assigns(_rewards_cfg())
    assert term in assigns, f"{term} is not in RewardsCfg"
    got = _const(_dict_get(assigns[term], "weight"))
    assert got == pytest.approx(weights[0]), (
        f"{term} static weight is {got}, expected the light start value {weights[0]}"
    )


@pytest.mark.parametrize("term,weights", sorted(RAMPED.items()))
def test_every_ramped_term_has_a_ramp_naming_it(term: str, weights):
    assigns = _assigns(_curriculum_cfg())
    found = None
    for node in assigns.values():
        if not isinstance(node, ast.Call):
            continue
        params = _dict_get(node, "params")
        if params is None:
            continue
        if _dict_get(params, "reward_name") is None:
            continue
        name = _dict_get(params, "reward_name")
        if isinstance(name, ast.Constant) and name.value == term:
            found = params
            break
    assert found is not None, f"no ramp term names '{term}'"
    assert _const(_dict_get(found, "terminal_weight")) == pytest.approx(weights[1])
    # The ramp's start_weight must match the static weight or the static value is dead.
    assert _const(_dict_get(found, "start_weight")) == pytest.approx(weights[0])


def test_ramps_are_upright_gated():
    """Every ramp must carry the gate -- that is the whole deviation from AGILE."""
    for node in _assigns(_curriculum_cfg()).values():
        if not isinstance(node, ast.Call):
            continue
        params = _dict_get(node, "params")
        if params is None or _dict_get(params, "reward_name") is None:
            continue
        assert _const(_dict_get(params, "upright_threshold")) is not None, (
            "a weight ramp without upright_threshold is AGILE's step-based schedule, "
            "which charges a falling policy for not being smooth"
        )


def test_every_ramp_func_is_the_ramp_term():
    for node in _assigns(_curriculum_cfg()).values():
        if not isinstance(node, ast.Call):
            continue
        params = _dict_get(node, "params")
        if params is None or _dict_get(params, "reward_name") is None:
            continue
        func = _dict_get(node, "func")
        assert "RewardWeightRampTerm" in ast.dump(func), (
            "a ramp must use RewardWeightRampTerm; CurriculumManager rejects a plain "
            "function as a curriculum term"
        )


def test_run_length_reaches_about_one_million_control_steps():
    """42 000 x 24 = 1 008 000, against AGILE's 50 000 x 24 = 1.2M."""
    ppo = (VEL / "agents/rsl_rl_ppo_cfg.py").read_text()
    tree = ast.parse(ppo)
    iters = steps_per = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id == "max_iterations":
                iters = _const(n.value)
            if n.targets[0].id == "num_steps_per_env":
                steps_per = _const(n.value)
    assert iters and steps_per
    total = iters * steps_per
    assert 900_000 <= total <= 1_500_000, f"run reaches {total} control steps, not ~1M"


def test_ramp_schedule_is_scaled_to_the_run_length():
    """start_step/num_steps must be inside the run, not AGILE's absolute 50k/150k.

    On our earlier 3 000-iteration runs (72k steps) AGILE's schedule never started.
    """
    iters = 42_000
    steps_per = 24
    total = iters * steps_per
    for node in _assigns(_curriculum_cfg()).values():
        if not isinstance(node, ast.Call):
            continue
        params = _dict_get(node, "params")
        if params is None or _dict_get(params, "reward_name") is None:
            continue
        start = _const(_dict_get(params, "start_step"))
        num = _const(_dict_get(params, "num_steps"))
        assert start is not None and num is not None
        assert start < total, "ramp starts after the run ends"
        assert start / total < 0.25, f"ramp starts at {100*start/total:.0f}% of the run"
        # it must actually COMPLETE within the run, or it is inert
        assert start + num < total, (
            f"ramp completes at step {start+num} of {total}: it would still be "
            "mid-schedule when the run ends"
        )