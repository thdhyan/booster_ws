# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""Save an UNTRAINED actor for the push task, so the environment can be run and rendered now.

WHY
---
Verifying the box-push scene does not require a trained policy. What it requires is a
policy object of the right shape: the scene, the frozen squat base, the per-env ground
patches, the 8 m cell pitch and the 10-dim action layout all live in the env cfg, not in
the weights. Building the runner and calling ``runner.save()`` before any training step
emits a checkpoint with the exact architecture and default-initialised weights, which
``play_record.py`` then loads unchanged.

That separates two questions that were previously tangled:
  * does the ENVIRONMENT work?  -> answered by this, in minutes, no training;
  * does the POLICY work?        -> answered by the training run, hours.

An untrained actor will not push. It will shuffle, drift and fall over, and the video
should be read as evidence about the scene and the plumbing, not about gait quality.

Usage mirrors play_record.py, including the argv reset Hydra needs:

    python make_zero_actor.py --task Isaac-Push-SG-K1-Play-v0 \
        --out models/k1_push_zero_actor.pt --export models/k1_push_zero_actor.ts.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source", "k1_velocity"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Save an untrained push actor.")
parser.add_argument("--task", default="Isaac-Push-SG-K1-Play-v0")
parser.add_argument("--out", required=True, help="where to write the .pt checkpoint")
parser.add_argument("--export", default="", help="also write a TorchScript copy here")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_tasks.utils import register_task  # noqa: E402

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env, agent_cfg = register_task(args_cli.task, {})
    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    agent_cfg.max_iterations = 1
    agent_cfg_dict = agent_cfg.to_dict()
    for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
        for mk in ("actor", "critic"):
            agent_cfg_dict[mk].pop(key, None)

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    wrapped = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = OnPolicyRunner(wrapped, agent_cfg_dict, log_dir="/tmp/k1_zero_actor",
                            device=agent_cfg.device)
    policy = runner.alg.get_policy()
    policy.eval()

    n_act = int(getattr(policy, "obs_dim", -1)) or None
    print(f"[INFO] actor built: obs_groups={agent_cfg.obs_groups.get('actor')}")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
    runner.save(os.path.abspath(args_cli.out))
    print(f"[ZERO_ACTOR_SAVED] {args_cli.out} (untrained, default init)")

    if args_cli.export:
        try:
            jit = policy.as_jit()
            jit = getattr(jit, "cpu", lambda: jit)()
            torch.save(jit, args_cli.export)
            print(f"[ZERO_ACTOR_TS_SAVED] {args_cli.export}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] TorchScript export skipped: {type(exc).__name__}: {exc}")

    print("ZERO_ACTOR_DONE")


main()
simulation_app.close()