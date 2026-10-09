"""The frozen squat base's obs contract is 238 dims, and it must stay that way.

What went wrong
---------------
``FrozenBaseVelocityAction._process_squat`` assembled the frozen base's observation
as 236 dims:

    lin 3 | ang 3 | grav 3 | cmd 4 | leg_pos 12 | leg_vel 12 | last 12 | scan 187

That matched the PRE-gait-fix squat export it was written against. Every squat
teacher trained since the gait work appends a 2-dim phase clock, so the exports are
238, and driving one with the 236 layout dies on the first policy call:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x236 and 238x512)

It surfaced as ``SMOKE_TB=6`` at 4096 envs -- and, because the push launcher's
fallback ladder assumed a failure meant "too many envs", it then also burned 2048,
1024 and 512 before reporting ``NO_WORKING_ENV_COUNT``. The env count was never the
problem; a shape mismatch is env-count independent. So the ladder's diagnosis was
wrong even though the run genuinely could not start.

The second half is the quieter bug. ``gait_clock.command_magnitude`` reads the
``base_velocity`` command term's ``vel_command_b``. In the push task that term is a
wrist-target command and has no such attribute, so the lookup returns ``None`` and
the clock freezes at ``PHASE_FREQUENCY_HZ``. That would not crash -- it would serve
the frozen base a constant cadence it was never trained against, which is the kind of
mismatch that shows up much later as "the base won't track".
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
PUSH = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/push"
PUSH_MDP = PUSH / "push_mdp.py"
CLOCK = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/gait_clock.py"

#: lin 3 + ang 3 + grav 3 + cmd 4 + leg_pos 12 + leg_vel 12 + last 12 + scan 187
BLOCK_DIMS = {"lin": 3, "ang": 3, "grav": 3, "cmd4": 4,
              "leg_pos": 12, "leg_vel": 12, "last": 12, "scan": 187}
CLOCK_DIMS = 2
LEGACY_EXPORT_DIMS = sum(BLOCK_DIMS.values())          # 236, pre-gait-fix
CURRENT_EXPORT_DIMS = LEGACY_EXPORT_DIMS + CLOCK_DIMS  # 238


def test_the_arithmetic_is_what_we_think_it_is():
    """Guard the numbers themselves; the whole test file rests on them."""
    assert LEGACY_EXPORT_DIMS == 236
    assert CURRENT_EXPORT_DIMS == 238


def _squat_method_src() -> str:
    """The method's CODE, with comments stripped.

    Via ast.unparse rather than a text slice: these assertions are about what runs,
    and a text slice cannot tell an explanatory comment from the offending call. The
    first version of this file failed for exactly that reason -- the fix's own
    comment mentions command_magnitude() while explaining why it must not be used.
    """
    tree = ast.parse(PUSH_MDP.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_process_squat":
            return ast.unparse(node)
    raise AssertionError("the squat mode (_process_squat) was renamed or removed")


def test_squat_obs_appends_the_phase_clock():
    """The 236 -> 238 fix. Without the clock the export cannot be loaded at all."""
    body = _squat_method_src()
    assert "phase_clock_from_magnitude" in body, (
        "the frozen base's obs must append the 2-dim gait clock; a post-gait-fix squat "
        f"export is {CURRENT_EXPORT_DIMS}-dim and the {LEGACY_EXPORT_DIMS}-dim layout "
        "raises 'mat1 and mat2 shapes cannot be multiplied' on the first policy call"
    )
    assert "clock]" in body or "clock," in body, "the clock must be concatenated into obs"


def test_the_clock_is_driven_by_the_action_not_the_command_manager():
    """Reading the command manager would silently freeze the cadence in push."""
    body = _squat_method_src()
    assert "command_magnitude" not in body, (
        "push's `base_velocity` command term is a wrist-target command with no "
        "vel_command_b, so command_magnitude returns None and the clock would freeze "
        "at PHASE_FREQUENCY_HZ -- no crash, just a cadence the base never trained on"
    )
    assert "last_vel_cmd" in body, "the clock must follow the action's velocity slice"


def test_gait_clock_exposes_the_magnitude_driven_path():
    tree = ast.parse(CLOCK.read_text())
    fns = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "get_frequency_from_magnitude" in fns
    assert "phase_clock_from_magnitude" in fns


def test_the_ema_and_clamp_exist_in_exactly_one_place():
    """A second copy of the EMA/clamp is how the served cadence drifts from the
    trained one, which is the failure this whole file is about."""
    tree = ast.parse(CLOCK.read_text())
    src = CLOCK.read_text()
    assert src.count("state[\"freq_hz\"].lerp_(") == 1, (
        "the frequency EMA must be written once; get_frequency delegates to "
        "get_frequency_from_magnitude"
    )
    # get_frequency should delegate, not re-derive
    g = next(n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "get_frequency")
    assert "get_frequency_from_magnitude" in ast.unparse(g), (
        "get_frequency must delegate to the shared implementation"
    )


def test_get_phase_accepts_a_per_env_frequency_tensor():
    """Otherwise the shared phase advance cannot be reused for a tensor frequency."""
    tree = ast.parse(CLOCK.read_text())
    gp = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "get_phase")
    assert "isinstance(frequency_hz, torch.Tensor)" in ast.unparse(gp), (
        "get_phase must accept a (num_envs,) tensor frequency, else the push path "
        "needs a second copy of the phase advance"
    )


def test_push_mdp_does_not_reimplement_the_frequency_math():
    """Keep the clock logic in gait_clock; push only supplies a speed."""
    body = _squat_method_src()
    for banned in ("HZ_PER_MPS", "lerp_", "MIN_HZ", "MAX_HZ"):
        assert banned not in body, (
            f"{banned} in push_mdp duplicates gait_clock's frequency logic; supply a "
            "magnitude to phase_clock_from_magnitude instead"
        )


@pytest.mark.parametrize("dims", [LEGACY_EXPORT_DIMS, CURRENT_EXPORT_DIMS])
def test_both_layouts_are_distinguishable(dims):
    """If these ever became equal the whole guard above would be vacuous."""
    assert dims in (236, 238)
    assert dims != CURRENT_EXPORT_DIMS or dims == 238