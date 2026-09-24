#!/usr/bin/env python3
"""ZERO-AGENT single-step verifier: build an env, reset, run N steps with
zero actions, and prove one full step completes (obs -> act -> physics ->
rewards). No runner, no wandb, no video — the fast gate before a smoke run.

Usage (inside the Isaac image, from the repo root):
    /isaac-sim/python.sh scripts/zero_step_check.py --task Isaac-Push-K1-v0
    /isaac-sim/python.sh scripts/zero_step_check.py --task Isaac-HeadTrack-K1-v0 --cameras

Prints ZERO_STEP_RESULT=OK on success (the rc of python.sh is unreliable —
always grep for this marker).
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--cameras", action="store_true", help="enable cameras (head task)")
parser.add_argument("--seed", type=int, default=42)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=args.cameras)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import booster_train.tasks  # noqa: F401
import k1_velocity.tasks  # noqa: F401 — registers velocity/basic/kick/head/push


def main() -> int:
    env = gym.make(args.task, num_envs=args.num_envs)
    obs, _ = env.reset(seed=args.seed)
    base = env.unwrapped
    action_dim = base.action_manager.total_action_dim
    print(f"[zero] task={args.task} envs={args.num_envs} action_dim={action_dim}")

    actions = torch.zeros((args.num_envs, action_dim), device=base.device)
    for step in range(args.steps):
        obs, rew, terminated, truncated, info = env.step(actions)
        rew_t = torch.as_tensor(rew)
        shapes = {k: tuple(v.shape) for k, v in obs.items()} if isinstance(obs, dict) else tuple(obs.shape)
        print(f"[zero] step {step}: obs={shapes} rew sum={rew_t.sum().item():.4f} "
              f"mean={rew_t.mean().item():.4f} term={int(torch.as_tensor(terminated).sum())} "
              f"trunc={int(torch.as_tensor(truncated).sum())}")
        # task-specific state probes
        st = getattr(base, "push_state", None)
        if st is not None:
            box = base.scene["box"].data.root_pos_w.torch
            print(f"[zero]   push: box_z mean={box[:, 2].mean().item():.3f} "
                  f"half_z mean={st.half_extents[:, 2].mean().item():.3f} "
                  f"mass mean={st.mass.mean().item():.1f}kg goal_off mean={st.goal_offset.norm(dim=-1).mean().item():.3f}m")
        ht = getattr(base, "head_track", None)
        if ht is not None:
            print(f"[zero]   head: yolo_active={ht.yolo_active} visible={int(ht.yolo[:, 0].sum())}"
                  f"/{args.num_envs} ball_speed={ht.ball_speed:.3f}")

    env.close()
    print("ZERO_STEP_RESULT=OK")
    return 0


try:
    rc = main()
except Exception:
    import traceback

    traceback.print_exc()
    print("ZERO_STEP_RESULT=FAIL")
    rc = 1
finally:
    simulation_app.close()
