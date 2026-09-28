"""Static audit of the K1 reward sets for degenerate optima.

P3 (head tracking) shipped a 2000-iteration "converged" policy that did not
track the ball at all.  Cause: ``ball_centered = exp(-(du^2 + dv^2)/s^2)``
evaluates to its **maximum 1.0** when the detector sees nothing, because the
detector leaves ``du = dv = 0``.  So pointing the head away -- or lying on the
floor so the camera sees no ball -- banked the full reward.  Training looked
healthy (``ball_in_frame 0.13``, W&B reward rising) and the policy was useless.

The same class of bug is cheap to reintroduce, so it is checked here rather than
discovered by burning GPU-hours.  This is a *static* audit: it parses the config
with ``ast`` and needs no Isaac Sim, so it can run on a laptop or in CI.

Run:  python -m pytest tests/test_reward_degeneracy.py -v
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VELOCITY_CFG = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_env_cfg.py"
HEAD_MDP = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/head/head_mdp.py"


def _class_body(path: pathlib.Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {path}")


def _assignments(path: pathlib.Path) -> dict[str, ast.AST]:
    """Top-level ``NAME = <expr>`` assignments, e.g. the joint-name lists."""
    tree = ast.parse(path.read_text())
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
    return out


def _reward_terms(path: pathlib.Path, cls: str) -> dict[str, ast.Assign]:
    """``{term_name: RewTerm(...)}`` assignments inside ``cls``."""
    body = _class_body(path, cls)
    terms = {}
    for node in body.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == "RewTerm":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        terms[target.id] = node
    return terms


def _weight(assign: ast.Assign) -> float | None:
    """The ``weight=`` argument. Negative literals parse as UnaryOp, not Constant."""
    for kw in assign.value.keywords:
        if kw.arg == "weight":
            try:
                return float(ast.literal_eval(kw.value))
            except (ValueError, TypeError):
                return None
    return None


def _inner_func(assign: ast.Assign) -> str:
    """The ``func=`` argument of a RewTerm/DoneTerm call, e.g. ``mdp.bad_orientation``.

    ``ast.unparse(call.func)`` would only give the *constructor* name
    (``RewTerm`` / ``DoneTerm``), which is identical for every term.
    """
    for kw in assign.value.keywords:
        if kw.arg == "func":
            return ast.unparse(kw.value)
    return ""


def _func(assign: ast.Assign) -> str:
    return _inner_func(assign)


def _params(assign: ast.Assign) -> str:
    """Source text of the RewTerm's ``params={...}`` argument."""
    return " ".join(ast.unparse(k.value) for k in assign.value.keywords if k.arg == "params")


# --------------------------------------------------------------------------
# 1. The P3 bug itself must stay fixed
# --------------------------------------------------------------------------
def test_head_centre_reward_is_masked_by_visibility():
    """``ball_centered`` must be gated on the detector's visible flag.

    Unmasked, exp(-0) == 1.0 is the maximum, so "see nothing" outscores
    "see it and centre it".
    """
    src = HEAD_MDP.read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "ball_centered")
    body = ast.unparse(fn)
    assert "st.yolo[:, 0]" in body, (
        "ball_centered no longer multiplies by the visible flag -- a policy that "
        "cannot see the ball will again collect the maximum centring reward"
    )


# --------------------------------------------------------------------------
# 2. Velocity task: no degenerate air-time attraction
# --------------------------------------------------------------------------
def test_velocity_has_no_air_time_reward():
    terms = _reward_terms(VELOCITY_CFG, "RewardsCfg")
    offenders = [n for n, a in terms.items() if "air_time" in _func(a)]
    assert not offenders, (
        f"air-time reward(s) {offenders} reintroduced. Air time is maximised by "
        "hopping or by never loading one foot; AGILE's Booster T1 velocity task "
        "omits it and shapes the feet directly instead."
    )


def test_velocity_has_dense_upright_reward():
    """Uprightness must be a gradient, not only a fall termination."""
    terms = _reward_terms(VELOCITY_CFG, "RewardsCfg")
    upright = [n for n, a in terms.items()
               if "base_height" in _func(a) or "orientation" in _func(a)]
    assert upright, "no base_height / orientation reward: nothing incentivises staying up"


def test_velocity_orientation_weight_is_not_negligible():
    """A weak tilt penalty is how P3 fell over; AGILE uses -5.0."""
    terms = _reward_terms(VELOCITY_CFG, "RewardsCfg")
    orient = [a for n, a in terms.items() if "orientation" in _func(a)]
    assert orient, "no flat_orientation reward"
    for a in orient:
        w = _weight(a)
        assert w is not None and w <= -1.0, (
            f"flat_orientation weight {w} is too weak to keep a biped upright "
            "(AGILE T1 uses -5.0)"
        )


def test_velocity_tracking_dominates_regularisation():
    """The task reward must stay in the same league as posture/regularisation,
    or the policy learns to satisfy the penalties instead of the command.

    The one-off fall penalty (``is_terminated``) is excluded: it is charged once
    at episode end, not per step, so summing it with per-step costs is a
    category error.  Per-step penalties total ~22.9 against 10.0 of tracking --
    the same operating point as AGILE's T1 task (5.0 + 5.0 tracking against ~20
    of penalties) -- so the floor sits below that, not above it.
    """
    terms = _reward_terms(VELOCITY_CFG, "RewardsCfg")
    track = sum(_weight(a) for n, a in terms.items() if "track_" in n and (_weight(a) or 0) > 0)
    reg = -sum(_weight(a) for n, a in terms.items()
               if (_weight(a) or 0) < 0 and "is_terminated" not in _func(a))
    assert track > 0, "no positive velocity-tracking reward"
    assert track >= reg * 0.3, (
        f"tracking ({track}) is dwarfed by per-step penalties ({reg:.1f}); the "
        "policy will satisfy the penalties instead of the task"
    )


# --------------------------------------------------------------------------
# 3. Joint-limit coverage
# --------------------------------------------------------------------------
def test_joint_pos_limits_cover_every_leg_joint():
    """Penalty must cover all 12 leg joints -- a subset lets hips/knees
    hyperextend (a real K1 collapse mode) for free."""
    terms = _reward_terms(VELOCITY_CFG, "RewardsCfg")
    cand = [a for n, a in terms.items() if "joint_pos_limits" in _func(a)]
    assert cand, "no joint_pos_limits reward"
    leg = _assignments(VELOCITY_CFG)["K1_LEG_JOINTS"]
    assert isinstance(leg, ast.List), "K1_LEG_JOINTS should be a literal list"
    n_legs = len(leg.elts)
    for a in cand:
        params = _params(a)
        if "K1_LEG_JOINTS" in params:
            return
        regexes = re.findall(r'"([^"]*)"', params)
        covered = sum(1 for r in regexes if "Ankle" in r or "Hip" in r or "Knee" in r)
        assert covered < n_legs, (
            f"joint_pos_limits covers {covered} of {n_legs} leg joints by regex; "
            "extend it to K1_LEG_JOINTS"
        )


# --------------------------------------------------------------------------
# 4. Fall must terminate, and tightly
# --------------------------------------------------------------------------
def test_velocity_has_fall_termination():
    body = _class_body(VELOCITY_CFG, "TerminationsCfg")
    fns = {_inner_func(n) for n in body.body
           if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)}
    assert any("root_height_below_minimum" in f for f in fns), "no height-based fall termination"
    assert any("bad_orientation" in f for f in fns), "no tilt termination"


def test_head_task_has_fall_termination():
    """P3 fell face-first for 500 steps with only a timeout to end the episode.

    Independent of rewards, that fall had to be *visible to training*.
    """
    body = _class_body(
        VELOCITY_CFG.parents[1] / "head/head_env_cfg.py", "TerminationsCfg")
    fns = {_inner_func(n) for n in body.body
           if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)}
    fall = [f for f in fns if any(k in f for k in
            ("root_height_below_minimum", "bad_orientation", "illegal_contact"))]
    assert fall, (
        "P3 has no fall termination -- the robot can collapse and the episode "
        "runs to timeout, hiding the failure from training"
    )


# --------------------------------------------------------------------------
# 5. No dead terms
# --------------------------------------------------------------------------
def test_no_zero_weight_reward_terms():
    terms = _reward_terms(VELOCITY_CFG, "RewardsCfg")
    dead = [n for n, a in terms.items() if _weight(a) in (0, 0.0)]
    assert not dead, f"reward terms with zero weight (dead config): {dead}"


def test_every_reward_func_is_importable_by_name():
    """Guard against referencing an mdp function that does not exist -- that
    fails at env construction, not at import, and costs a full container start."""
    terms = _reward_terms(VELOCITY_CFG, "RewardsCfg")
    missing = []
    for n, a in terms.items():
        fn = _func(a)
        mod, _, attr = fn.rpartition(".")
        if mod == "gait":
            gp = VELOCITY_CFG.parent / "gait_rewards.py"
            gtree = ast.parse(gp.read_text())
            have = {x.name for x in ast.walk(gtree) if isinstance(x, ast.FunctionDef)}
            if attr not in have:
                missing.append(f"{n} -> gait_rewards.{attr}")
    assert not missing, f"reward functions not defined: {missing}"
