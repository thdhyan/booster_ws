"""--warm_start must load weights without the optimizer or the iteration counter.

Regression. Fine-tuning under a *changed* reward needs three things a plain
`runner.load()` does not give:

1. **No optimizer state.** `optimizer_state_dict` carries Adam's exp_avg and
   exp_avg_sq -- moment estimates accumulated against the *old* reward. Once the
   reward's scale and shape change they are meaningless, and they are the first
   thing to bias a warm start. This matters here specifically: the reward changed
   from an inert `feet_clearance` (pinned at ~1e-6 by a 1 mm height gap) to a real
   lift penalty, so the old moments describe a reward that no longer exists.
2. **Iteration counter reset.** `OnPolicyRunner.learn` runs
   `range(current_learning_iteration, current_learning_iteration + num_learning_iterations)`.
   Restoring the counter from a `model_4999` means `--max_iterations 5000` starts
   at 4999 and the run is a no-op tail.
3. **A learning rate that can still move.** rsl_rl's PPO adapts LR on measured KL
   (`lr /= 1.5` or `lr *= 1.5`, clamped to [1e-5, 1e-2]) rather than on iteration
   index, so a fresh optimizer needs no LR help. If that ever changes to an
   iteration-indexed schedule, a reset-to-0 counter is what makes it correct.
"""
from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN = _ROOT / "isaac_tasks/k1_velocity/scripts/train.py"
VEL = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity"
GAIT = VEL / "gait_rewards.py"
CFG = VEL / "velocity_env_cfg.py"


def _src() -> str:
    return TRAIN.read_text()


def test_warm_start_flag_exists():
    assert '"--warm_start"' in _src(), (
        "there is no way to fine-tune from a checkpoint without inheriting the "
        "old optimizer state and iteration counter"
    )


def test_warm_start_selectively_loads_and_resets():
    src = _src()
    assert "load_cfg=" in src, "the load must be selective, not a bare runner.load()"
    for key, why in (
        ('"actor": True', "the policy weights are the point of a warm start"),
        ('"critic": True', "the value function must warm-start too"),
        ('"optimizer": False', (
            "Adam moments from the old reward would bias the fine-tune, which is "
            "the whole reason this flag exists")),
        ('"iteration": False', (
            "restoring the counter makes --max_iterations buy no budget, since "
            "learn() starts at current_learning_iteration")),
    ):
        assert key in src, f"warm start must pass {key} -- {why}"


def test_plain_resume_path_still_exists():
    """--warm_start is opt-in; default resume behaviour must be unchanged."""
    src = _src()
    assert "runner.load(resume_path)\n" in src or "runner.load(resume_path)" in src, (
        "the default resume path must remain a plain load"
    )


def test_clearance_fix_is_in_place_for_this_run():
    """The run this flag exists for changes feet_clearance to a lift measure."""
    gait = GAIT.read_text()
    cfg = CFG.read_text()
    fn = next(
        n for n in ast.walk(ast.parse(gait))
        if isinstance(n, ast.FunctionDef) and n.name == "feet_clearance"
    )
    body = [ast.unparse(s) for s in fn.body]
    docstring = body[0] if body and "target_height" in body[0] else ""
    code = "\n".join(body[1:] if docstring else body)
    assert "target_lift" in code, "feet_clearance must score a lift"
    assert "target_height" not in code, (
        "the inert absolute-height formulation must be gone from the code"
    )
    assert '"target_lift"' in cfg, (
        "the reward must be wired with target_lift; a term still asking for "
        "target_height would be silently inert in exactly the way this run is "
        "trying to fix"
    )
