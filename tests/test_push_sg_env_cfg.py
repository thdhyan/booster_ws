"""Wiring checks for the arm-contact SG push env cfgs.

Isaac-free and static, like the rest of this repo's cfg tests: ``push_sg_env_cfg``
inherits from ``push_env_cfg``, which needs a full Isaac boot to instantiate, and the
mistakes worth catching here are all decidable from the source.

The one that matters most is the SIGN CONVENTION. ``push_env_cfg`` documents a real
past bug: functions that already return negative error were given negative weights,
so the product was positive and policies were paid to keep their wrists away from the
box -- 0% contact in the v1 runs. The SG functions are the opposite convention
(non-negative, so positive weights are correct), which means a weight copied across
from the v3 block would silently invert the reward again.
"""
from __future__ import annotations

import ast
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]
PUSH = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/push"
SG_CFG = PUSH / "push_sg_env_cfg.py"
SG_MDP = PUSH / "push_rewards_sg.py"
INIT = PUSH / "__init__.py"
BASE_CFG = PUSH / "push_env_cfg.py"


def _cfg_assignments(path):
    """{class_name: {term_name: ast node}} for RewTerm assignments in a cfg file."""
    out = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ClassDef):
            continue
        terms = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                if getattr(stmt.value.func, "id", "") != "RewTerm":
                    continue
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        terms[t.id] = stmt.value
        out[node.name] = terms
    return out


def _weight(call):
    for kw in call.keywords:
        if kw.arg == "weight":
            return ast.literal_eval(kw.value)
    return None


def _func_name(node):
    """Bare name of a func expression: sg.foo / mdp.foo / foo."""
    if isinstance(node, ast.Attribute):
        return node.attr
    return getattr(node, "id", None)


# ------------------------------------------------------------------- registration


def test_both_sg_tasks_are_registered():
    src = INIT.read_text()
    for tid in ("Isaac-Push-Reach-SG-K1-v0", "Isaac-Push-SG-K1-v0"):
        assert f'id="{tid}"' in src, f"{tid} is not registered"


def test_sg_tasks_point_at_the_sg_cfg_not_the_v3_one():
    """A copy-paste slip here would silently train the v3 rewards under an SG name."""
    src = INIT.read_text()
    for tid, cfg in (
        ("Isaac-Push-Reach-SG-K1-v0", "K1PushReachSGEnvCfg"),
        ("Isaac-Push-SG-K1-v0", "K1PushSGEnvCfg"),
    ):
        block = src.split(f'id="{tid}"', 1)[1].split("gym.register", 1)[0]
        assert f"push_sg_env_cfg:{cfg}" in block, f"{tid} does not point at {cfg}"


# ------------------------------------------------------------------- reward wiring


def test_every_sg_term_resolves_to_a_real_function():
    defined = {n.name for n in ast.walk(ast.parse(SG_MDP.read_text()))
               if isinstance(n, ast.FunctionDef)}
    mdp_defined = {n.name for n in ast.walk(ast.parse((PUSH / "push_mdp.py").read_text()))
                   if isinstance(n, ast.FunctionDef)}
    cfgs = _cfg_assignments(SG_CFG)
    for cls in ("K1PushSGRewardsCfg", "K1PushReachSGRewardsCfg"):
        for term, call in cfgs[cls].items():
            # RewTerm(func=..., weight=...) passes the function as a KEYWORD
            fn_kw = next((kw.value for kw in call.keywords if kw.arg == "func"), None)
            fn = _func_name(fn_kw) if fn_kw is not None else (
                _func_name(call.args[0]) if call.args else None)
            assert fn in defined | mdp_defined, f"{cls}.{term} -> unknown function {fn}"


def test_sign_convention_all_sg_weights_are_positive():
    """SG functions return non-negative (etiquette is signed, positive = clean).

    Every weight must therefore be positive. A negative one here is the v1 sign bug
    wearing a new name: the term would pay the policy for the behaviour it exists to
    suppress.
    """
    cfgs = _cfg_assignments(SG_CFG)
    for cls in ("K1PushSGRewardsCfg", "K1PushReachSGRewardsCfg"):
        for term, call in cfgs[cls].items():
            w = _weight(call)
            assert w is None or w > 0, f"{cls}.{term} has weight {w}; SG terms are non-negative"


def test_v3_corner_block_is_retired_in_the_sg_push_cfg():
    """The SG set replaces the corner-goal block; keeping both double-counts progress."""
    tree = ast.parse(SG_CFG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "K1PushSGRewardsCfg")
    retired = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            retired.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    for name in ("corner_goal_tracking", "centroid_goal_tracking",
                 "box_goal_progress", "box_vel_toward_goal"):
        assert name in retired, f"{name} must be retired (set to None) in the SG push cfg"


def test_reach_stage_has_no_push_shaping():
    """Reaching is walking up and touching the box; paying for goal progress there
    would reward knocking a heavy box around instead of placing the hands on it."""
    tree = ast.parse(SG_CFG.read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "K1PushReachSGRewardsCfg")
    retired = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            retired.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    for name in ("task_box_goal", "heading_progress", "stop_bonus", "etiquette", "parked"):
        assert name in retired, f"{name} must not be active in the reach stage"


# ------------------------------------------------------------------- inheritance


def test_orthogonal_terms_are_inherited_not_re_tuned():
    """Height / velocity tracking / regularization must come from v3 unchanged.

    Re-tuning them here would confound an A/B of the reward SET with an A/B of the
    regularization, and there would be no way to attribute a result.
    """
    src = SG_CFG.read_text()
    sg_terms = set(_cfg_assignments(SG_CFG)["K1PushSGRewardsCfg"])
    inherited = {
        "base_height_command", "success_bonus", "termination_penalty",
        "track_cmd_lin_vel", "track_cmd_ang_vel", "flat_orientation_l2",
        "action_rate_l2", "dof_torques_l2", "joint_pos_limits",
        "wrist_target_tracking",
    }
    base_terms = set(_cfg_assignments(BASE_CFG)["K1PushRewardsCfg"])
    for name in inherited:
        assert name in base_terms, f"{name} vanished from the v3 base cfg"
        assert name not in sg_terms, (
            f"{name} is re-specified in push_sg_env_cfg; it should be inherited so the "
            "only difference from v3 is the push-shaping block"
        )
    assert "K1PushSGRewardsCfg(base.K1PushRewardsCfg)" in src


def test_env_cfgs_subclass_the_v3_ones():
    src = SG_CFG.read_text()
    assert "class K1PushSGEnvCfg(base.K1PushEnvCfg)" in src
    assert "class K1PushReachSGEnvCfg(base.K1PushReachEnvCfg)" in src