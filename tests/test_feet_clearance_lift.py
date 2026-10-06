"""`feet_clearance` must measure LIFT, not absolute foot height.

Regression, 2026-10-06. The term compared each swinging foot's world height
against `target_height = 0.06` m. Measured on the seed-43 student, the foot link
origin rests at ~0.059 m, so at rest `short = 1 mm` and `short**2 = 1e-6`: the
term was pinned near zero in every run and could only charge for *sinking*, never
for failing to lift.

Measured lift is what exposed it -- the feet unload 63-80% of steps yet mean LIFT
is +0.0000 m at 0.1 m/s and only +0.015 m at 0.5 m/s. That is the shuffle.

So the invariant is arithmetic, not stylistic: a penalty whose target sits at or
below the foot's own resting height can never charge anything, and a term that
compares against a *measured* resting height will silently return ~0 forever.
"""
from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VEL = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity"
CFG = VEL / "velocity_env_cfg.py"
GAIT = VEL / "gait_rewards.py"

# Measured on students43_v0.1_trace.npz / v0.5_trace.npz (8 envs, 750 steps).
FOOT_REST_HEIGHT_M = 0.059      # foot link origin world z when loaded
OLD_TARGET_M = 0.06             # what the term used to ask for
MEASURED_LIFT_AT_0_1 = 0.0000   # mean lift, cmd 0.1 m/s
MEASURED_LIFT_AT_0_5 = 0.0153   # mean lift, cmd 0.5 m/s


def _fn(name: str):
    tree = ast.parse(GAIT.read_text())
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _code_only(fn: ast.FunctionDef) -> str:
    """Unparse a function's *statements*, excluding its docstring.

    The docstrings here deliberately name ``target_height`` while explaining what
    it replaced, so a substring check over ``ast.unparse(fn)`` would match the
    prose instead of the code.
    """
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _cls(name: str) -> ast.ClassDef:
    tree = ast.parse(CFG.read_text())
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == name
    )


def _const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const(node.operand)
        return None if inner is None else -inner
    return None


def _term_params(term_name: str) -> dict:
    cls = _cls("RewardsCfg")
    assigns = {
        n.targets[0].id: n.value
        for n in ast.walk(cls)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
    }
    term = assigns[term_name]
    params = next(
        kw.value for kw in term.keywords if kw.arg == "params"
    )
    assert isinstance(params, ast.Dict)
    out = {}
    for k, v in zip(params.keys, params.values):
        if isinstance(k, ast.Constant):
            out[k.value] = v
    return out


def test_old_target_sat_at_or_below_the_foot_resting_height():
    """The arithmetic that made the term inert, kept as a regression."""
    old_short = OLD_TARGET_M - FOOT_REST_HEIGHT_M
    old_penalty = old_short ** 2
    assert old_penalty < 1e-5, (
        f"the old law predicted a penalty of {old_penalty:.2e} at rest, i.e. "
        "nothing -- which is why it read -0.0000 for every run"
    )
    # And the failure it could not see: the shuffle has ~zero lift.
    assert MEASURED_LIFT_AT_0_1 < 0.005, (
        "the premise of the fix: if the measured shuffle had real lift, the old "
        "absolute-height term would not have been the blocker"
    )
    assert MEASURED_LIFT_AT_0_5 < 0.02, (
        "even at the fastest gated speed the mean lift is ~1.5 cm, not a step"
    )


def test_clearance_measures_lift_against_the_other_foot():
    """The scored quantity must be a foot's height RELATIVE to the stance foot."""
    body = _code_only(_fn("feet_clearance"))
    assert "target_height" not in body, (
        "target_height is the inert formulation; the term must ask for a lift"
    )
    assert "target_lift" in body, "the parameter must be a lift target"
    # Reference each foot against the mean of the others.
    assert "others" in body and "lift" in body, (
        "each foot must be scored against the mean height of the OTHER feet, "
        "so the swinging foot is measured against the stance foot"
    )
    assert "body_pos_w[:, asset_cfg.body_ids, 2]" in body, (
        "lift must come from foot-link world z"
    )


def test_clearance_is_wired_with_a_target_lift_and_pinned_foot_order():
    params = _term_params("feet_clearance")
    assert "target_lift" in params, (
        "feet_clearance is wired with target_height; the reward would be inert. "
        "A defined-but-misconfigured term is worse than a missing one, because "
        "it reads as a working penalty in the log"
    )
    assert "target_height" not in params
    lift = _const(params["target_lift"])
    assert lift is not None and 0.03 <= lift <= 0.12, (
        f"target_lift {lift} m is out of range: under ~3 cm is a scrape, over "
        "~12 cm is unreachable for a 0.59 m-tall biped"
    )
    for key in ("contact_cfg", "asset_cfg"):
        assert "preserve_order" in ast.dump(params[key]), (
            f"{key} must pin preserve_order=True; it defaults to False, which "
            "follows the sensor's body order instead of [left, right]"
        )


def test_clearance_still_skips_envs_with_no_swing_foot():
    """The `active` gate stays: an env with both feet down must not be punished.

    This gate is also what let the old bug hide -- a clean 0.0000 read as
    "compliant" rather than "never applicable". It stays because penalising
    standing would fight the upright/base-height terms, but it must not be the
    thing that decides whether the term works.
    """
    body = _code_only(_fn("feet_clearance"))
    assert "swing.any(dim=1)" in body, (
        "envs with no swinging foot must still be skipped"
    )
