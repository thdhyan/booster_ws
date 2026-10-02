"""Tests for AGILE T1 reward parity.

Four rewards were added on 2026-10-01 to close the gap against NVIDIA's AGILE
locomotion task for the Booster T1
(``agile/rl_env/tasks/locomotion/t1/velocity_env_cfg.py``), the closest reference
AGILE ships to a K1:

* ``ankle_torques`` -- torque penalty over all ankle joints
* ``dof_vel`` -- whole-body joint-velocity penalty
* ``jumping`` -- both feet off the ground at once
* the *weighted* velocity trackers, replacing the plain exponential

These are wiring tests (static, AST + source text) in the same style as
``test_gait_rewards.py``: the reward bodies need torch and Isaac Lab, so their
numerics are checked on the GPU box, while what is verified here is that they are
declared, carry AGILE's weights, and are attached to the config.

The weight-ramp maths in ``agile_rewards`` is pure torch and is tested for real
in ``test_agile_parity.py``'s numeric section, because a weight ramp that is
silently pinned to 1.0 looks exactly like a working reward in a training log.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VEL = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity"
CFG = VEL / "velocity_env_cfg.py"
AGILE = VEL / "agile_rewards.py"


def _rewards_cfg(path: pathlib.Path) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg":
            return node
    raise AssertionError("no RewardsCfg found")


def _terms(cfg: ast.ClassDef) -> dict:
    out = {}
    for node in ast.walk(cfg):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Call):
                out[tgt.id] = node.value
    return out


def _kw(call: ast.Call, name: str):
    """Value of keyword ``name`` on a Call, or on a dict literal passed as params."""
    if isinstance(call, ast.Call):
        for kw in call.keywords:
            if kw.arg == name:
                return kw.value
        return None
    if isinstance(call, ast.Dict):
        for k, v in zip(call.keys, call.values):
            if isinstance(k, ast.Constant) and k.value == name:
                return v
    return None


def _const(node) -> float | None:
    """Literal float, tolerating the UnaryOp that ``-1.0`` parses into."""
    if node is None:
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _func_name(node: ast.AST) -> str | None:
    """Dotted source name of a `func=` reference, e.g. ``agile.jumping`` -> 'agile.jumping'."""
    if isinstance(node, ast.Attribute):
        base = _func_name(node.value)
        return None if base is None else f"{base}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return None


# --- the three added H2/T1 terms ----------------------------------------------
# Weights are AGILE T1's verbatim. They are pinned because a weight silently
# drifting off the reference is indistinguishable from a training failure.
AGILE_T1_WEIGHTS = {
    "ankle_torques": -1.0e-4,
    "dof_vel": -2.0e-4,
    "jumping": -0.5,
}


@pytest.mark.parametrize("term,weight", sorted(AGILE_T1_WEIGHTS.items()))
def test_agile_t1_term_is_declared_with_agiles_weight(term: str, weight: float):
    terms = _terms(_rewards_cfg(CFG))
    assert term in terms, f"{term} missing: AGILE T1 declares it"
    got = _const(_kw(terms[term], "weight"))
    assert got == pytest.approx(weight), f"{term} weight {got} != AGILE T1 {weight}"


@pytest.mark.parametrize("term", sorted(AGILE_T1_WEIGHTS))
def test_added_penalties_are_negative(term: str):
    """All three are penalties; a positive weight would reward the behaviour."""
    terms = _terms(_rewards_cfg(CFG))
    assert _const(_kw(terms[term], "weight")) < 0.0


def test_ankle_torques_covers_both_ankle_axes():
    """All four ankle joints, pitch AND roll.

    Matching only ``Ankle_Roll`` (as ``ankle_roll_torques`` does) would leave the
    sagittal push-off axis untaxed, which is the axis the push-off happens on.
    """
    terms = _terms(_rewards_cfg(CFG))
    params = _kw(terms["ankle_torques"], "params")
    assert params is not None, "ankle_torques must pass an explicit joint selection"
    text = ast.dump(params)
    assert "Ankle" in text, "ankle_torques must select ankle joints explicitly"
    assert "Roll" not in text, (
        "ankle_torques must not narrow to Ankle_Roll -- that is ankle_roll_torques' "
        "job at 20x the weight; this term is the broad ankle penalty"
    )


def test_dof_vel_is_whole_body_like_agile_t1():
    """AGILE T1 passes SceneEntityCfg("robot") -- all joints, not just the legs.

    Restricting it to the 12 leg joints would silently be a different (weaker)
    term than the reference it is meant to reproduce.
    """
    terms = _terms(_rewards_cfg(CFG))
    params = _kw(terms["dof_vel"], "params")
    asset = _kw(params, "asset_cfg")
    assert asset is not None, "dof_vel needs an asset_cfg"
    assert not (isinstance(asset, ast.Call) and _kw(asset, "joint_names") is not None), (
        "dof_vel must cover the whole robot (AGILE T1 passes no joint_names); "
        "a joint_names filter makes it a leg-only term"
    )


def test_jumping_uses_the_contact_sensor_and_a_force_threshold():
    terms = _terms(_rewards_cfg(CFG))
    func = _kw(terms["jumping"], "func")
    assert func is not None and "jumping" in (_func_name(func) or "")
    params = _kw(terms["jumping"], "params")
    assert _const(_kw(params, "threshold")) is not None, "jumping needs a force threshold"
    sensor = _kw(params, "sensor_cfg")
    assert sensor is not None, "jumping must read the contact sensor"
    assert "foot" in ast.dump(sensor).lower(), "jumping must watch the feet"


def test_jumping_is_not_a_duplicate_of_phase_swing():
    """They must not resolve to the same predicate.

    ``phase_swing`` scores contact against the gait clock; ``jumping`` is a plain
    both-feet-airborne test. If this regresses, one of the two is redundant and
    the gait structure is paying twice for a single constraint.
    """
    terms = _terms(_rewards_cfg(CFG))
    assert _func_name(_kw(terms["jumping"], "func")) != _func_name(
        _kw(terms["phase_swing"], "func")
    )


# --- weighted tracking --------------------------------------------------------
def test_both_trackers_use_the_weighted_agile_functions():
    terms = _terms(_rewards_cfg(CFG))
    lin = (_func_name(_kw(terms["track_lin_vel_xy_exp"], "func")) or "").rsplit(".", 1)[-1]
    ang = (_func_name(_kw(terms["track_ang_vel_z_exp"], "func")) or "").rsplit(".", 1)[-1]
    assert lin == "track_lin_vel_xy_exp_weighted", f"linear tracker is {lin!r}"
    assert ang == "track_ang_vel_z_world_exp_weighted", f"angular tracker is {ang!r}"
    # The unweighted Isaac Lab terms are what this change replaced; if either
    # comes back, the speed shortfall returns with it. Compared on the exact
    # final name because the weighted names have the plain ones as a prefix.
    assert lin != "track_lin_vel_xy_yaw_frame_exp"
    assert ang != "track_ang_vel_z_world_exp"


def test_weighted_trackers_keep_the_audited_narrow_std():
    """std must stay 0.15 / 0.25, not AGILE's 0.2.

    Those narrowings were the earlier half of the same 'it walks too slowly' fix
    (a wide exponential is nearly flat near the optimum, so a 0.14 m/s error cost
    almost nothing). Taking AGILE's std while keeping AGILE's weighted function
    would undo it.
    """
    terms = _terms(_rewards_cfg(CFG))
    for name, expected in (("track_lin_vel_xy_exp", 0.15), ("track_ang_vel_z_exp", 0.25)):
        params = _kw(terms[name], "params")
        assert _const(_kw(params, "std")) == pytest.approx(expected)


def test_weighted_trackers_pass_the_command_name():
    terms = _terms(_rewards_cfg(CFG))
    for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
        params = _kw(terms[name], "params")
        cmd = _kw(params, "command_name")
        assert isinstance(cmd, ast.Constant) and cmd.value == "base_velocity", (
            f"{name} must name the command term that CommandsCfg registers"
        )


# --- the port module itself ---------------------------------------------------
def test_weight_ramp_reads_the_live_command_range():
    """The ramp must follow ``cfg.ranges`` at call time, not a value captured once.

    ``VelocityRangeCurriculumTerm`` rewrites those ranges during training to widen
    from +/-0.5 to +/-4.0 m/s. A ramp pinned to the initial range goes flat the
    moment the curriculum moves and stops rewarding speed -- the precise failure
    the weighted tracker was added to fix.
    """
    src = AGILE.read_text()
    assert "term.cfg.ranges" in src, (
        "the weight ramp must read the command term's live ranges, not a constant"
    )


def test_weight_ramp_refuses_a_degenerate_range():
    """A zero-width range divides by zero; that must raise, not silently pin to 1.0.

    Silently returning ``weight_min`` looks exactly like a correctly-behaving
    reward in a training log, so it is a guard rather than a nicety.
    """
    src = AGILE.read_text()
    assert "hi > lo" in src or "hi > lo:" in src, "degenerate range must be rejected"


def _load_agile_module():
    """Import ``agile_rewards`` by file path, bypassing the package ``__init__``.

    ``k1_velocity.tasks.velocity.__init__`` registers every gym task, and that
    global registration breaks unrelated tests later in the suite (they then
    resolve a task config through a stale import). Loading the one module we need
    keeps this test's side effects to the module under test.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_agile_under_test", AGILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_degenerate_range_is_only_tolerated_when_the_caller_opts_in():
    """A pinned axis must not crash play, but must still crash training.

    Recording a fixed straight-line command sets ``ang_vel_z = (0, 0)``, so the
    yaw ramp's upper bound is 0 and ``hi > lo`` is false. Before ``require_span``
    existed, every ``--cmd X 0 0`` run of the Distill and Rough play tasks died on
    the first reward step with "weighted tracking needs hi > lo, got lo=0.1,
    hi=0.0" -- which blocked straight-line gait recordings and the keyboard
    forward-walk session, the two things the play cfgs exist for.

    The opt-in must be narrow: with the default the guard still raises, because a
    silent collapse during training is exactly the bug it was written to catch.
    """
    import torch

    agile_rewards = _load_agile_module()
    mag = torch.tensor([0.0, 0.5, 2.0])
    # Opted in (play): a zero-width range yields the floor weight, not a crash.
    out = agile_rewards._weight_from_magnitude(mag, 0.1, 0.0, require_span=False)
    assert torch.allclose(out, torch.ones_like(mag)), (
        f"a pinned axis should hold the floor weight, got {out.tolist()}"
    )
    # Default (training): still refuses, so a curriculum collapse stays loud.
    with pytest.raises(ValueError, match="hi > lo"):
        agile_rewards._weight_from_magnitude(mag, 0.1, 0.0)


def test_play_configs_opt_out_of_the_range_span_requirement():
    """Both play cfgs that keep rewards must set ``require_span=False``.

    The teacher play cfg replaces RewardsCfg wholesale, so it never evaluated a
    tracking term and never hit the bug -- which is exactly why this went
    unnoticed. The Rough and Distill play cfgs keep the training rewards and pin
    the command, so both need the opt-out, and any new play cfg that keeps rewards
    will need it too.

    Checked on the source rather than by instantiating the cfgs: building a cfg
    pulls in the whole task module tree, and this file's tests are AST + source by
    design so they run with no GPU.
    """
    for path in (VEL / "velocity_play_cfg.py", VEL / "velocity_play_distill.py"):
        tree = ast.parse(path.read_text())
        post_inits = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
        ]
        assert post_inits, f"{path.name} has no __post_init__ to configure from"
        body = "\n".join(ast.unparse(n) for n in post_inits)
        for term_name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            assert term_name in body, (
                f"{path.name} no longer references {term_name}; if the weighted "
                "tracking terms moved, re-check whether it still needs the opt-out"
            )
        assert '"require_span"] = False' in body.replace("'", '"'), (
            f"{path.name} must set params[\"require_span\"] = False on the weighted "
            "tracking terms: a fixed command pins the unused axes to zero, so the "
            "ramp has no span and the term raises on the first reward step"
        )


def test_min_vel_norm_is_a_parameter_and_never_zeroes_commands():
    """``min_vel_norm`` must stay a weighting knob, never a command filter.

    In AGILE it doubles as a zeroing threshold on ``UniformNullVelocityCommand``.
    We keep standing environments on purpose (they carry the upright signal
    ``base_height`` depends on), so a port that also zeroed commands would delete
    the standing data the task needs.

    Checked structurally (is anything ASSIGNED to a command buffer?) rather than
    by substring, because these functions legitimately *read* ``vel_command_b``.
    """
    assert "min_vel_norm: float = 0.1" in AGILE.read_text(), (
        "min_vel_norm must be a defaulted parameter"
    )
    tree = ast.parse(AGILE.read_text())
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for tgt in targets:
            name = _func_name(tgt) or ""
            if "vel_command" in name:
                raise AssertionError(
                    f"{name!r} is assigned to in agile_rewards.py; this module must "
                    "be read-only w.r.t. sampled commands or it silently deletes the "
                    "standing environments the upright signal depends on"
                )
        for node2 in ast.walk(node):
            if (
                isinstance(node2, ast.Call)
                and isinstance(node2.func, ast.Attribute)
                and node2.func.attr.startswith("zero_")
            ):
                raise AssertionError(
                    f"zeroing call {node2.func.attr} in agile_rewards.py: this module "
                    "must not clear sampled commands"
                )


def test_agile_module_documents_every_deviation_from_agile():
    """Each port deviation must be written down, or the next reader assumes a bug.

    Three departures from AGILE source are deliberate (the ``min_vel_norm``
    availability, the live-range ramp, the guarded denominator) and a fourth is a
    weight we kept from an earlier audit instead of AGILE's.
    """
    src = AGILE.read_text()
    doc = ast.get_docstring(ast.parse(src)) or ""
    for marker in ("min_vel_norm", "vel_max", "curriculum"):
        assert marker.lower() in doc.lower(), f"deviation {marker!r} is undocumented"


def test_agile_module_does_not_import_isaac_at_module_scope():
    """It must import cleanly on a laptop with no Isaac Sim, so the tests can run.

    ``gait_rewards`` sets the precedent and ``test_gait_rewards`` relies on it.
    """
    tree = ast.parse(AGILE.read_text())
    top = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and "isaaclab.utils.math" in ast.dump(n)
    ]
    assert not top, (
        "quaternion helpers must be imported inside the function; a module-scope "
        "isaaclab.utils.math import makes this file unimportable off-GPU"
    )