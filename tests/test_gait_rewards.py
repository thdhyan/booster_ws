"""Tests for the gait-structure rewards and the gait gate.

Two failure modes are covered, and both were real:

* **Wiring** (static, AST): the reward terms must exist, be penalties with
  negative weights, and the tracking reward must actually be sharpened. A reward
  term that is defined but never referenced is inert, which is the same silent
  failure as a ``CurrTerm`` that loads zero terms.

* **Gate logic** (numeric, numpy): ``gait_gate.py`` must FAIL a high-frequency
  shuffle and PASS a human-like gait. A gate that only ever fails is useless, and
  a gate that only ever passes is worse than none -- the displacement gate proved
  that, by passing an 8 Hz chatter.

The reward functions themselves need torch and Isaac Lab, so their numerical
behaviour is checked on the GPU box; what is verified here is that they are
correctly declared and that the gate around them discriminates.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VEL = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity"
CFG = VEL / "velocity_env_cfg.py"
GAIT = VEL / "gait_rewards.py"
GATE = _ROOT / "isaac_tasks/k1_velocity/scripts/gait_gate.py"


# --- helpers -----------------------------------------------------------------
def _rewards_cfg(path: pathlib.Path) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg":
            return node
    raise AssertionError("no RewardsCfg found")


def _terms(cfg: ast.ClassDef) -> dict:
    """Map term name -> RewTerm call node inside RewardsCfg."""
    out = {}
    for node in ast.walk(cfg):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Call):
                out[tgt.id] = node.value
    return out


def _kw(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _const(node) -> float | None:
    """Literal float, handling the unary minus that ``-1.0`` parses into.

    ``-1.0`` is a UnaryOp(USub, Constant(1.0)) in the AST, not a negative
    Constant, so a naive literal read returns None for every penalty weight.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


# --- the new terms exist and are penalties -----------------------------------
NEW_TERMS = ["gait_cadence", "feet_clearance", "feet_alternation", "stride_length", "action_jerk_l2"]


@pytest.mark.parametrize("term", NEW_TERMS)
def test_gait_term_is_declared_in_rewards_cfg(term: str):
    terms = _terms(_rewards_cfg(CFG))
    assert term in terms, f"{term} is not wired into RewardsCfg; a defined-but-unused reward is inert"


@pytest.mark.parametrize("term", NEW_TERMS)
def test_gait_term_has_a_negative_weight(term: str):
    """All five are penalties. A positive weight would reward chattering."""
    terms = _terms(_rewards_cfg(CFG))
    w = _const(_kw(terms[term], "weight"))
    assert w is not None, f"{term} has no literal weight"
    assert w < 0, f"{term} weight is {w}; penalties must be negative"


def test_tracking_reward_was_sharpened():
    """std 0.25 -> 0.15.

    At std=0.25 the exponential is nearly flat near the optimum, so a 0.14 m/s
    tracking error cost almost nothing and the policy settled at 69% of commanded
    speed while being paid for it.
    """
    terms = _terms(_rewards_cfg(CFG))
    params = _kw(terms["track_lin_vel_xy_exp"], "params")
    assert params is not None
    std = None
    for k in params.keys:
        if getattr(k, "value", None) == "std":
            std = _const(params.values[params.keys.index(k)])
    assert std is not None, "track_lin_vel_xy_exp has no std"
    assert std == pytest.approx(0.15), (
        f"tracking std is {std}; it must be 0.15 or the reward is too flat to "
        "pressure the policy to track accurately"
    )


def test_action_rate_was_strengthened():
    """-0.5 could not see 8 Hz chatter; action_rate_l2 only sees 1st differences."""
    terms = _terms(_rewards_cfg(CFG))
    w = _const(_kw(terms["action_rate_l2"], "weight"))
    assert w is not None and w <= -2.0, f"action_rate_l2 weight is {w}, expected <= -2.0"


def test_gait_functions_defined_in_module():
    src = GAIT.read_text()
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in ("gait_cadence_penalty", "feet_clearance", "feet_alternation_penalty",
               "stride_length_penalty", "action_jerk_l2"):
        assert fn in defined, f"{fn} is not defined in gait_rewards.py"


def test_no_dead_placeholder_code_left_in_jerk_term():
    """A placeholder var that is computed then deleted is a smell, not a feature."""
    src = GAIT.read_text()
    assert "placeholder, replaced below" not in src
    assert "del jerk" not in src


# --- gate logic --------------------------------------------------------------
def _synth(cadence_hz: float, speed: float, amp: float, steps: int = 750, envs: int = 8) -> str:
    """Write a synthetic trace: knee oscillation at cadence_hz/2, given speed."""
    t = np.arange(steps) * 0.02
    A = np.zeros((steps, envs, 12), np.float32)
    V = np.zeros((steps, envs, 3), np.float32)
    for e in range(envs):
        for j in range(12):
            phase = 0.0 if j < 6 else np.pi
            A[:, e, j] = amp * np.sin(2 * np.pi * cadence_hz * t + phase + j * 0.1)
        V[:, e, 0] = speed
    p = pathlib.Path("/tmp/opencode/_gt.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, actions=A, root_lin_vel=V)
    return str(p)


def _run_gate(path: str) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(GATE), path], capture_output=True, text=True)
    return r.returncode, r.stdout


def test_gate_fails_a_high_frequency_shuffle():
    """4 Hz knee oscillation = 8 steps/s. This is the failure we actually hit."""
    path = _synth(cadence_hz=4.0, speed=0.35, amp=0.1)
    rc, out = _run_gate(path)
    assert rc == 1, f"gate passed a shuffle:\n{out}"
    assert "GAIT_GATE=FAIL" in out
    assert "cadence" in out


def test_gate_passes_a_human_like_walk():
    """1 Hz knee oscillation = 2 steps/s, human walking cadence."""
    path = _synth(cadence_hz=1.0, speed=0.5, amp=0.35)
    rc, out = _run_gate(path)
    assert rc == 0, f"gate failed a plausible walk:\n{out}"
    assert "GAIT_GATE=PASS" in out


def test_gate_passes_a_slow_deliberate_walk():
    """0.7 Hz knee = 1.4 steps/s: slow but legitimately human."""
    path = _synth(cadence_hz=0.7, speed=0.45, amp=0.35)
    rc, out = _run_gate(path)
    assert rc == 0, f"gate failed a slow walk:\n{out}"


def test_gate_rejects_a_non_trace_cleanly():
    p = pathlib.Path("/tmp/opencode/_notrace.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, something_else=np.zeros(3))
    rc, out = _run_gate(str(p))
    assert rc == 2, f"expected exit 2 for a bad input, got {rc}:\n{out}"


def test_gate_rejects_a_missing_file():
    rc, out = _run_gate("/tmp/opencode/definitely_absent.npz")
    assert rc == 2


def test_gait_phase_buffer_shape_is_requested_exactly():
    """The jerk term needs (num_envs, num_actions), not a 1-D (num_envs,).

    Allocating 1-D produced "size of tensor a (12) must match the size of tensor
    b (8) at non-singleton dimension 1" and killed the preflight. The state helper
    must therefore honour the requested shape rather than assuming 1-D.
    """
    src = GAIT.read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "get"
    )
    # the signature must take a shape, not a num_envs int
    args = [a.arg for a in fn.args.args]
    assert "shape" in args, f"_GaitPhase.get signature is {args}, expected a 'shape' arg"
    assert "num_envs" not in args, "a num_envs int invites the 1-D bug back"
    # ast.dump renders torch.zeros as Name(torch) + Attribute(zeros), not as a
    # literal string, so look for the attribute node instead.
    calls_zeros = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "zeros"
        for n in ast.walk(fn)
    )
    assert calls_zeros, "_GaitPhase.get must allocate with torch.zeros"
    # the jerk term must request the full action shape
    assert "flat.shape" in src, "action_jerk_l2 must pass flat.shape, not an int"
