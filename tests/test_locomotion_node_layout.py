"""Deployment-node input-layout contract.

Regression. `locomotion_node.py` probed the policy with a hardcoded
`torch.zeros(1, OBS_DIM)` (50) while its inference path feeds
`history.T.flatten()` (50 x 10 = 500) whenever `input_mode='stacked'`.

That made the only mode a distilled student can run in the one mode that failed
to load. A student's first Linear is (512, 500) -- it consumes the 10-step
history -- so bringing one up needed `input_mode='stacked'`, and that setting
made the load-time probe feed 50 dims to a 500-input net and raise "policy input
layout mismatch". The message then blamed 48-dim pre-phase-clock exports, which
had nothing to do with it.

These are source/AST checks plus a numeric check of the width arithmetic, so they
run with no ROS, no GPU and no robot.
"""
from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
NODE = _ROOT / "src/k1_locomotion/k1_locomotion/locomotion_node.py"

OBS_DIM = 50
HISTORY_LEN = 10


def _src() -> str:
    return NODE.read_text()


def _cls(name: str) -> ast.ClassDef:
    for n in ast.walk(ast.parse(_src())):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    raise AssertionError(f"no class {name} in {NODE.name}")


def test_input_width_follows_input_mode_not_a_hardcoded_obs_dim():
    """The probe width must be derived from input_mode and history_len."""
    tree = ast.parse(_src())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_policy"
    )
    body = ast.unparse(fn)
    assert "torch.zeros(1, OBS_DIM)" not in body, (
        "the load-time probe is hardcoded to OBS_DIM, so input_mode='stacked' "
        "(500 inputs) can never pass it -- which is the only mode a distilled "
        "student works in"
    )
    assert "self._policy_input_dim" in body, (
        "the probe must use the width the inference path actually feeds"
    )


def test_expected_width_is_history_times_obs_dim_when_stacked():
    """Check the arithmetic itself: stacked -> 500, latest -> 50."""
    stacked = HISTORY_LEN * OBS_DIM
    assert stacked == 500, "a distilled student's first Linear is (512, 500)"
    widths = {"latest": OBS_DIM, "stacked": stacked}
    assert widths["stacked"] != widths["latest"], (
        "if both modes had the same width there would be nothing to regress"
    )


def test_inference_feed_matches_the_probed_width():
    """The assert at the inference site is what stops the two drifting again."""
    src = _src()
    assert "history.T.flatten()" in src, (
        "stacked mode must keep feeding the term-major history stack"
    )
    tree = ast.parse(src)
    asserts = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assert)
        and "_policy_input_dim" in ast.unparse(n.test)
    ]
    assert asserts, (
        "no assert ties the inference feed to the probed width; history_len and "
        "input_mode can drift apart silently again"
    )


def test_error_message_names_both_real_widths_not_the_old_48_dim_boogeyman():
    """The old message blamed 48-dim exports and misdirected the reader."""
    tree = ast.parse(_src())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_policy"
    )
    body = ast.unparse(fn)
    assert "48-dim" not in body, (
        "the phase-clock 48-dim story is not the cause of a 500-vs-50 mismatch "
        "and sends the reader hunting the wrong regression"
    )
    assert "input_mode" in body and "stacked" in body, (
        "the message must say which input_mode was used and what the other "
        "layout wants, since that is the actionable fact"
    )


def test_node_obs_dim_matches_the_phase_clock_layout():
    """OBS_DIM must stay 50 = 48 proprioception + 2 gait-clock, never 48."""
    tree = ast.parse(_src())
    assigns = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "OBS_DIM" for t in n.targets)
    ]
    assert assigns, "OBS_DIM must be a module-level constant"
    value = assigns[0].value
    assert isinstance(value, ast.Constant) and value.value == OBS_DIM, (
        "OBS_DIM drifted from 50; 48 is the pre-phase-clock layout"
    )
