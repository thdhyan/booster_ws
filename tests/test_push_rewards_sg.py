"""Tests for the SG-style arm-contact push reward terms.

Why these run without Isaac
---------------------------
``push_rewards_sg`` is a sibling of ``push_mdp``, which imports Isaac at module
scope. But the arithmetic that decides whether a reward is well-behaved -- whether
``recip`` can blow up at contact, whether ``pull_fraction`` has the sign that makes
dragging the box *cost* something -- lives in the ``_*`` helpers and touches nothing
but tensors. The module is exec'd here from its real source with the sibling import
stripped, so this exercises the shipped code rather than a copy.

The public terms take an Isaac ``env`` and need a live scene; what is checked for
those is wiring (the names exist, they use the arm geometry, and ``parked_bonus``
agrees with the termination that actually ends the episode).
"""
from __future__ import annotations

import ast
import pathlib
import re
import types

import pytest

torch = pytest.importorskip("torch")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
PUSH = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/push"
MOD = PUSH / "push_rewards_sg.py"


def _load_without_isaac():
    """exec the module with the sibling ``push_mdp`` import removed."""
    src = "\n".join(
        l for l in MOD.read_text().splitlines() if not l.startswith("from .push_mdp import")
    )
    mod = types.ModuleType("push_rewards_sg_no_isaac")
    mod.__dict__["__file__"] = str(MOD)
    exec(compile(src, str(MOD), "exec"), mod.__dict__)
    return mod


M = _load_without_isaac()


# --------------------------------------------------------------------------- recip


def test_recip_is_one_at_zero_and_decays_without_going_negative():
    d = torch.tensor([0.0, 0.1, 0.4, 1.0, 10.0])
    r = M.recip(d, 0.40)
    assert r[0].item() == pytest.approx(1.0)
    assert torch.all(r >= 0.0)
    assert torch.all(r[:-1] > r[1:]), "must be strictly decreasing"
    assert r[-1].item() < 0.05, "must actually decay toward zero"


def test_recip_is_bounded_which_is_the_whole_point():
    """A raw 1/d returns 50 at d=0.02 m and swamps every other reward in the manager.

    That is the failure the bounded form exists to prevent, so assert the bound
    rather than only the monotonicity.
    """
    d = torch.tensor([0.0, 0.001, 0.01, 0.02])
    assert M.recip(d, 0.40).max().item() <= 1.0 + 1e-6


def test_recip_clamps_negative_distance():
    """Contact noise can report a tiny negative gap; it must not invert the reward."""
    assert M.recip(torch.tensor([-0.05]), 0.40).item() == pytest.approx(1.0)


# ------------------------------------------------------------------ inside_radius


def test_inside_radius_hard_variant_is_a_step():
    r = M.inside_radius(torch.tensor([0.30, 0.35, 0.36]), 0.35)
    assert r.tolist() == [1.0, 1.0, 0.0]


def test_inside_radius_smooth_ramp_decays_linearly_to_zero():
    d = torch.tensor([0.35, 0.385, 0.42])
    r = M.inside_radius(d, 0.35, smooth=0.07)
    assert r[0].item() == pytest.approx(1.0)
    assert r[1].item() == pytest.approx(0.5, abs=1e-5)
    assert r[2].item() == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------- heading_cos


def test_heading_cos_spans_the_full_range():
    to_goal = torch.tensor([[1.0, 0.0]])
    assert M.heading_cos(torch.tensor([[1.0, 0.0]]), to_goal).item() == pytest.approx(1.0)
    assert M.heading_cos(torch.tensor([[-1.0, 0.0]]), to_goal).item() == pytest.approx(-1.0)
    assert M.heading_cos(torch.tensor([[0.0, 1.0]]), to_goal).item() == pytest.approx(0.0, abs=1e-6)


def test_heading_cos_is_finite_for_a_degenerate_zero_direction():
    """Standing exactly on the goal must not produce NaN."""
    out = M.heading_cos(torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0, 0.0]]))
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------ pull_fraction


def test_pull_fraction_sign_separates_pushing_from_dragging():
    to_goal = torch.tensor([[1.0, 0.0]])  # goal lies at +x of the box
    assert M.pull_fraction(torch.tensor([[0.30, 0.0]]), to_goal).item() > 0.0
    assert M.pull_fraction(torch.tensor([[-0.30, 0.0]]), to_goal).item() < 0.0, (
        "dragging the box must be chargeable, or a distance term will reward it"
    )


def test_pull_fraction_ignores_lateral_slip():
    to_goal = torch.tensor([[1.0, 0.0]])
    assert M.pull_fraction(torch.tensor([[0.0, 0.5]]), to_goal).item() == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------------------- engagement


def test_engagement_gates_on_the_gap():
    """Silent once the hands are on the box -- the boundary is exclusive."""
    g = torch.tensor([0.9, 0.6, 0.2])
    assert M.engagement(g, 0.60).tolist() == [1.0, 0.0, 0.0]


# ------------------------------------------------------- module-level guarantees


def test_module_does_not_import_push_things():
    """The point of the port: this repo must not depend on the other checkout.

    Checked on the AST, not the raw text -- the module docstring legitimately names
    Push-Things when explaining what it was ported from, and a substring check
    would flag that and force the useful explanation out of the file.
    """
    for node in ast.walk(ast.parse(MOD.read_text())):
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("push_things") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("push_things")


def test_all_seven_ported_terms_are_present():
    names = {n.name for n in ast.walk(ast.parse(MOD.read_text()))
             if isinstance(n, ast.FunctionDef)}
    for fn in (
        "approach_object_recip",
        "fast_approach",
        "task_box_to_goal_recip",
        "stop_at_goal_box_still",
        "heading_progress_combo",
        "push_pull_etiquette",
        "parked_bonus",
    ):
        assert fn in names, f"missing ported term {fn}"


def test_terms_measure_the_arms_not_the_base():
    """Regression guard for the semantic difference from the base-push reference.

    An arm-contact task that silently measures base distance reads "arrived" while
    the hands are nowhere near the box.
    """
    src = MOD.read_text()
    body = src.split("def approach_object_recip", 1)[1].split("\ndef ", 1)[0]
    assert "wrist_box_gap" in body, "approach must be measured from the wrists"
    assert "root_pos_w" not in body, "approach must not fall back to base distance"
    assert "wrist_box_gap" in src.split("def push_pull_etiquette", 1)[1].split("\ndef ", 1)[0], (
        "standoff must be wrist withdrawal, which is the arm-specific part"
    )


def test_parked_bonus_agrees_with_the_success_termination():
    """The dense reward must describe the same event the terminator rewards.

    ``parked_bonus`` mirrors ``goal_reached`` (mean corner error under SUCCESS_ERR_M,
    held for SUCCESS_HOLD_S). It must NOT drift back to the base-push
    ``box_inside_and_parked`` shape: v3 gives the box a goal *pose*, so a
    centre-in-radius bonus would pay out for correct position at wrong yaw and would
    disagree with the termination that actually ends the episode.
    """
    body = MOD.read_text().split("def parked_bonus", 1)[1].split("\ndef ", 1)[0]
    assert "SUCCESS_ERR_M" in body and "SUCCESS_HOLD_S" in body, (
        "parked_bonus must take its tolerance from the termination's constants"
    )
    # Assert the geometry that is actually called, not merely the word "corner": a
    # centre-distance rewrite can keep the variable named corner_err and slip through.
    assert "box_corners_base" in body and "goal_corners_base" in body, (
        "success here is the 8-corner pose error; a centre-distance rewrite would pay "
        "out for correct position at wrong yaw"
    )
    assert "root_pos_w" not in body.split('"""', 2)[-1], (
        "the active implementation must not use the base-push centre/radius shape"
    )
    assert "success_hold" in body, "must reuse the counter goal_reached maintains"


def test_success_constants_agree_between_module_and_mdp():
    """Guard the constants themselves against being edited in one place only."""
    mdp = (PUSH / "push_mdp.py").read_text()
    err = re.search(r"SUCCESS_ERR_M\s*=\s*([0-9.]+)", mdp)
    hold = re.search(r"SUCCESS_HOLD_S\s*=\s*([0-9.]+)", mdp)
    assert err and hold, "push_mdp must define SUCCESS_ERR_M / SUCCESS_HOLD_S"
    assert "success_hold" in MOD.read_text(), "parked_bonus must reuse the maintained counter"