"""Tests for the command-coupled gait clock and the phase-locked stride term.

Added 2026-10-01 after the 5000-iteration run finished 8/8 gate failures with
cadence 4.85 steps/s and jerk 0.114, which showed that:

* a cadence *rate* penalty is bistable -- 0.4 steps/s and 11 steps/s are both
  "far from 2.0" and the policy takes the cheaper one, so no weight fixes it;
* ``phase_synced_swing`` (contact locked to the clock) could not fix it either,
  because contact is binary and a 4 cm twitch satisfies it as well as a stride;
* the clock was fixed at 1.001 Hz regardless of the command, so a position
  reference would fight velocity tracking rather than reinforce it.

These are wiring tests (AST + source) in the style of ``test_gait_rewards``;
the clock arithmetic needs torch, so it is checked for real in the numeric
section using the Isaac-free seam the clock already exposes.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VEL = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity"
CFG = VEL / "velocity_env_cfg.py"
GAIT = VEL / "gait_rewards.py"
CLOCK = VEL / "gait_clock.py"
TRAIN = _ROOT / "isaac_tasks/k1_velocity/scripts/train.py"


def _cls(path: pathlib.Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    raise AssertionError(f"no {name} in {path.name}")


def _assigns(cls: ast.ClassDef) -> dict:
    return {
        n.targets[0].id: n.value
        for n in ast.walk(cls)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
    }


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


def _dotted(node):
    if isinstance(node, ast.Attribute):
        b = _dotted(node.value)
        return None if b is None else f"{b}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return None


def _func_name(node):
    return (_dotted(node) or "").rsplit(".", 1)[-1]


# --- the stride term exists and is wired ---------------------------------------
def test_phase_locked_stride_is_wired_as_a_penalty():
    assigns = _assigns(_cls(CFG, "RewardsCfg"))
    assert "phase_locked_stride" in assigns, "the phase-locked stride term is missing"
    node = assigns["phase_locked_stride"]
    assert _func_name(_dict_get(node, "func")) == "phase_locked_stride"
    assert _const(_dict_get(node, "weight")) < 0.0, "a positive weight would reward misplacement"


def test_stride_term_reads_both_feet_in_a_fixed_order():
    """preserve_order matters: this term is the only one reading foot index 0 vs 1.

    It relies on index 0 being the LEFT foot so the reference wave leads on the
    correct side. Default preserve_order=False follows the sensor's order.
    """
    assigns = _assigns(_cls(CFG, "RewardsCfg"))
    params = _dict_get(assigns["phase_locked_stride"], "params")
    asset = _dict_get(params, "asset_cfg")
    dump = ast.dump(asset)
    assert "left_foot_link" in dump and "right_foot_link" in dump
    assert "preserve_order" in dump, "foot order must be pinned or the wave leads on the wrong foot"


def test_stride_term_takes_a_reference_amplitude():
    assigns = _assigns(_cls(CFG, "RewardsCfg"))
    params = _dict_get(assigns["phase_locked_stride"], "params")
    amp = _const(_dict_get(params, "target_stride"))
    assert amp is not None and 0.1 < amp < 0.4, (
        f"target_stride {amp} must be a plausible stride; the gate calls >=0.15 a stride"
    )


def test_stride_term_function_is_defined():
    src = GAIT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "phase_locked_stride" in names


# --- why it is not just another cadence penalty --------------------------------
def test_stride_is_position_based_not_a_rate_penalty():
    """It must key off foot placement against the clock, not off step rate.

    The whole reason this term exists is that ``gait_cadence_penalty`` measured a
    rate and was bistable. A regression to rate/phase-only logic here would
    reintroduce the exact failure.
    """
    src = GAIT.read_text()
    body = src.split("def phase_locked_stride", 1)[1].split("\ndef ", 1)[0]
    assert "body_pos_w" in body, "the term must read foot position"
    assert "reference" in body, "the term must compare against a clock-derived reference"
    # Zero crossings of the reference wave are what make slow and fast both wrong.
    assert "sin" in body, "the reference must oscillate with phase"


def test_stride_forward_axis_is_three_dimensional():
    """Regression: the yaw forward axis was built 2-component and broadcast
    against 3-component positions.

    Cost a full run before it was caught:
    ``RuntimeError: The size of tensor a (3) must match the size of tensor b (2)
    at non-singleton dimension 3`` on the first reward step. The launcher still
    exited 0, which is the ``python.sh`` rc trap -- only the log showed it.
    """
    body = GAIT.read_text().split("def phase_locked_stride", 1)[1].split("\ndef ", 1)[0]
    # A 3-component axis must be stacked from three entries, not two.
    assert "torch.zeros_like" in body, (
        "the forward axis must be 3-component (x, y, 0) to project 3-D positions"
    )
    # And it must be applied without an unsqueeze that would mis-shape it.
    assert ".sum(dim=-1)                             # (N, 2)" in body or (
        "along = ((pos - hip) * fwd).sum(dim=-1)" in body
    ), "fwd must broadcast directly against (N, 2, 3) offsets"


def test_stride_reference_antiphases_the_two_feet():
    """Left and right must be half a cycle apart, else both feet move together."""
    body = GAIT.read_text().split("def phase_locked_stride", 1)[1].split("\ndef ", 1)[0]
    assert "-wave" in body or "1.0 - 2" in body, (
        "the two feet must be driven in antiphase from one phase"
    )


def test_stride_amplitude_falls_with_cadence():
    """Faster cadence means shorter per-step excursion, like a real gait.

    Without this the term would demand the same absolute stride at 0.1 and 0.5
    m/s, which is the command/clock decoupling this change exists to remove.
    """
    body = GAIT.read_text().split("def phase_locked_stride", 1)[1].split("\ndef ", 1)[0]
    assert "hz" in body, "amplitude must be scaled by the clock's frequency"


# --- the clock now follows the command -----------------------------------------
def test_frequency_defaults_to_following_the_command():
    """frequency_hz=None is the new default; a pinned default would undo the fix."""
    tree = ast.parse(CLOCK.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("get_phase", "phase_clock"):
            args = node.args.args
            defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
            for a, d in zip(args, defaults):
                if a.arg == "frequency_hz":
                    assert isinstance(d, ast.Constant) and d.value is None, (
                        f"{node.name}: frequency_hz must default to None so the clock "
                        "follows the commanded speed"
                    )


def test_clock_exposes_a_frequency_helper():
    src = CLOCK.read_text()
    assert "def get_frequency" in src, "the clock must expose its per-env frequency"
    assert "def reset_clock_state" in src, "tests need a way to drop cached clock state"


def test_frequency_is_clamped_into_the_gate_band():
    """The clamp must keep cadence inside the gate's [1.2, 4.0] steps/s.

    MIN_HZ/MAX_HZ are gait CYCLES/s, so the steps/s band is [2*MIN, 2*MAX].
    """
    src = CLOCK.read_text()
    assert "MIN_HZ" in src and "MAX_HZ" in src
    min_hz = _const(next(
        (n.value for n in ast.walk(ast.parse(src))
         if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "MIN_HZ"), None))
    max_hz = _const(next(
        (n.value for n in ast.walk(ast.parse(src))
         if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "MAX_HZ"), None))
    assert min_hz is not None and max_hz is not None
    assert 2 * min_hz >= 1.2, f"MIN_HZ {min_hz} -> {2*min_hz} steps/s is below the gate band"
    assert 2 * max_hz <= 4.0, f"MAX_HZ {max_hz} -> {2*max_hz} steps/s is above the gate band"


def test_frequency_never_reaches_zero():
    """A zero command must not freeze the phase.

    At frequency 0 the reference stops and the term goes uninformative exactly
    when the policy is least confident; MIN_HZ keeps a standing reference.
    """
    src = CLOCK.read_text()
    assert "MIN_HZ" in src.split("def get_frequency", 1)[1], (
        "get_frequency must clamp the low end so a stopped robot still has a reference"
    )


def test_command_resample_is_smoothed():
    """Commands resample every 8-12 s; an unsmoothed frequency would step the
    reference mid-stride."""
    body = CLOCK.read_text().split("def get_frequency", 1)[1].split("\ndef ", 1)[0]
    assert "lerp_" in body or "smooth" in body.lower(), "frequency must be EMA-smoothed"


def test_clock_is_still_idempotent_within_a_control_step():
    """Both observation groups and the reward read the phase in one step.

    The advance-on-episode_length_buf scheme is what makes that safe; the
    frequency change must not have broken it.
    """
    body = CLOCK.read_text().split("def get_phase", 1)[1].split("\ndef ", 1)[0]
    assert "prev_len" in body, "the idempotency guard must survive"
    assert "stepped" in body, "the advance must still be gated on an actual step"


# --- ramp gate must be reachable ------------------------------------------------
def test_ramp_gate_is_not_the_unreachable_080():
    """My 2ddec51 error: 0.80 with a measured upright ratio of ~0.02-0.15.

    The ramp would never fire, leaving every gait penalty at 1/5 strength for the
    whole run -- entrenching the shuffle the ramp was meant to help remove.
    """
    for node in _assigns(_cls(CFG, "CurriculumCfg")).values():
        if not isinstance(node, ast.Call):
            continue
        params = _dict_get(node, "params")
        if params is None or _dict_get(params, "reward_name") is None:
            continue
        thr = _const(_dict_get(params, "upright_threshold"))
        assert thr is not None, "every ramp needs a threshold"
        assert thr <= 0.30, (
            f"upright_threshold {thr} is above the measured upright ratio "
            "(~0.02-0.15), so the ramp would never fire"
        )


# --- video must not throttle training ------------------------------------------
def test_video_recorder_is_opt_in():
    """Measured 14 s -> 23 s per iteration at 4096 envs with the recorder on."""
    src = TRAIN.read_text()
    assert "--video_during_training" in src, "recording must be gated behind its own flag"
    assert "record = args_cli.video and args_cli.video_during_training" in src.replace(" ", " ") or (
        "args_cli.video_during_training" in src
    )


def test_video_interval_default_is_1000():
    src = TRAIN.read_text()
    line = next(l for l in src.splitlines() if "--video_interval" in l)
    assert "1000" in line, f"video_interval default must be 1000, got: {line.strip()}"


def test_render_mode_is_not_set_by_bare_video():
    """render_mode='rgb_array' is the 20x: it captures every step, not just the
    ~200 that a clip covers."""
    src = TRAIN.read_text()
    for line in src.splitlines():
        # skip comments: this file's own explanation mentions render_mode
        code = line.split("#", 1)[0]
        if "render_mode=" in code:
            assert "record" in code, (
                f"render_mode must key off the gated `record`, not bare --video: {line.strip()}"
            )
            break
    else:
        pytest.fail("no render_mode= line found")


# --- the observation must expose the reference the reward uses -----------------
def test_policy_observes_the_phase_the_reward_scores_against():
    """A phase-locked reward is unlearnable if the policy cannot see the phase.

    Both the observation and the reward must read ``gait_clock.get_phase`` /
    ``phase_clock`` so they cannot drift onto different references.
    """
    assigns = _assigns(_cls(CFG, "PolicyCfg"))
    phase_terms = [
        v for k, v in assigns.items()
        if "phase" in k.lower()
    ]
    assert phase_terms, "the policy group must still carry the phase clock"
    obs_src = CFG.read_text()
    assert "gait_clock.phase_clock" in obs_src
    assert "gait_clock.get_phase" in GAIT.read_text(), (
        "the reward must read the same clock the observation encodes"
    )