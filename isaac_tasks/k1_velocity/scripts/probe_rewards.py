"""Reward probe: construct the P2 teacher env and exercise every reward term.

Catches the failure mode that a static audit cannot -- a reward function that
raises, returns a wrong shape, is identically zero (dead term), or is NaN.  The
P3 lesson was that a reward set can be *wired* correctly and still be wrong;
this checks the numbers.

Prints a table and exits non-zero if any term is missing, non-finite, or dead.
"""
from __future__ import annotations

import argparse
import math
import os
import sys  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../booster_train_ref/scripts/rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Rough-K1-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--min-stand-z", type=float, default=0.45,
                    help="required mean trunk height at the end of the probe")
# Terms that are legitimately zero for a constant all-zero action, so the probe
# must not report them as broken. The probe holds the action at zero, which means
# no foot is swinging and the action history is constant, so:
#   feet_clearance - no swing foot exists, so there is nothing to score
#   action_jerk_l2 - a constant action has a zero third difference
#   gait_cadence   - a still robot produces no zero crossings
# The first two are structural, not bugs. Listing them here keeps the probe's
# "dead term" check meaningful for everything else.
ZERO_ACTION_INERT = {"action_rate_l2", "dof_vel_limits", "torque_limits",
                     "undesired_contacts", "termination_penalty",
                     "feet_clearance", "action_jerk_l2", "gait_cadence",
                     "feet_alternation", "stride_length"}
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# The hydra_task_config decorator below re-parses sys.argv, so hand it only the
# leftover overrides (same pattern as train.py / play_record.py).
import sys as _sys

_sys.argv = [_sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import booster_train.tasks  # noqa: F401
import k1_velocity.tasks.velocity  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab.envs import ManagerBasedRLEnvCfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    obs, _ = wrapped.reset()

    zero = torch.zeros(args_cli.num_envs, wrapped.num_actions, device=wrapped.device)
    tm = env.unwrapped.termination_manager
    print(f"\nTERMINATION MANAGER: active_terms={list(tm.active_terms)}  "
          f"n_cfg={len(getattr(tm, '_term_cfgs', []))}")
    height_log = []
    for step in range(args_cli.steps):
        out = wrapped.step(zero)
        obs = out[0]  # (obs, rew, terminated, ...) — length varies by Lab version
        if step % 10 == 0 or step == args_cli.steps - 1:
            z = float(env.unwrapped.scene["robot"].data.root_pos_w[:, 2].mean())
            height_log.append((step, z))
            print(f"  step {step:4d}  mean trunk z = {z:.3f} m")

    mgr = env.unwrapped.reward_manager
    print("\n" + "=" * 78)
    print(f"REWARD PROBE  task={args_cli.task}  envs={args_cli.num_envs} steps={args_cli.steps}")
    print("=" * 78)
    print(f"{'term':28s} {'weight':>9s} {'mean':>11s} {'max':>11s}  status")

    dead, nonfinite, missing = [], [], []
    unexpected_dead = []
    # RewardManager keeps the per-term, per-step, already-weight-scaled values in
    # _step_reward[:, i].  (compute() applies *weight * dt, then divides by dt for
    # this buffer, so it reads as the weighted per-step reward.)
    names = list(mgr._term_names)
    cfgs = list(mgr._term_cfgs)
    step_reward = mgr._step_reward
    total = torch.zeros(args_cli.num_envs, device=wrapped.device)
    for i, (name, cfg) in enumerate(zip(names, cfgs)):
        val = step_reward[:, i]
        if val is None:
            missing.append(name)
            continue
        total += val
        finite = bool(torch.isfinite(val).all())
        nonzero = bool(val.abs().max() > 1e-9)
        if not finite:
            nonfinite.append(name)
        if not nonzero:
            dead.append(name)
            if name not in ZERO_ACTION_INERT:
                unexpected_dead.append(name)
        status = "ok"
        if not finite:
            status = "NON-FINITE"
        elif not nonzero:
            status = "DEAD (always 0)"
        print(f"{name:28s} {float(cfg.weight):9.4f} {float(val.mean()):11.5f} "
              f"{float(val.max()):11.5f}  {status}")

    upright = float(env.unwrapped.scene["robot"].data.root_pos_w[:, 2].mean())
    print("-" * 78)
    print(f"mean trunk height after {args_cli.steps} zero-action steps: {upright:.3f} m "
          f"(K1 stands at 0.57; <0.35 means it collapsed under zero action)")
    print(f"total weighted reward/step: {float(total.mean()):.4f}")
    print(f"term count: {len(names)}")

    tail = height_log[len(height_log) // 2:]
    min_tail_z = min(z for _, z in tail)
    end_z = height_log[-1][1]
    stands = end_z >= args_cli.min_stand_z and min_tail_z >= args_cli.min_stand_z - 0.08
    print(f"standing check: end z={end_z:.3f} m, min over last half={min_tail_z:.3f} m, "
          f"required >= {args_cli.min_stand_z:.2f} m -> {'STANDS' if stands else 'COLLAPSED'}")
    if dead:
        print(f"(inert under zero action, expected: {sorted(set(dead))})")
    ok = not (unexpected_dead or nonfinite or missing) and stands
    print("=" * 78)
    if missing:
        print(f"REWARD_PROBE_MISSING={missing}")
    if nonfinite:
        print(f"REWARD_PROBE_NONFINITE={nonfinite}")
    if unexpected_dead:
        print(f"REWARD_PROBE_DEAD={unexpected_dead}")
    if not stands:
        print(f"REWARD_PROBE_NOT_STANDING end_z={end_z:.3f} min_tail_z={min_tail_z:.3f}")
    print(f"REWARD_PROBE_MARKER={'OK' if ok else 'FAIL'}")
    wrapped.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
