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

That separates two questions that were otherwise tangled:
  * does the ENVIRONMENT work?  -> answered by this, in minutes, no training;
  * does the POLICY work?        -> answered by the training run, hours.

An untrained actor will not push. It will shuffle, drift and fall over, and the video
should be read as evidence about the scene and the plumbing, not about gait quality.

Config plumbing follows play_record.py exactly: ``@hydra_task_config`` supplies BOTH
env_cfg and agent_cfg, so nothing has to be pulled out of the gym registry by hand. An
earlier version imported a ``register_task`` helper that does not exist, and
``isaaclab_tasks.utils`` only lazy-exports -- it has no ``register_task`` at all.

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

# ORDERING IS LOAD-BEARING. AppLauncher must run before ANY other isaaclab import.
#
# Importing isaaclab_tasks / rsl_rl first splits module identity: interactive_scene then
# tests the scene's cfgs against a SensorBaseCfg loaded after app start, while the cfgs
# themselves were built from the pre-app import graph. The classes are nominally the same
# but are distinct objects, so isinstance() is False and the scene resolver falls all the
# way through its dispatch chain to
#
#     ValueError: Unknown asset config type for height_scanner: RayCasterCfg(...)
#
# which is the single most confusing error this task has produced: height_scanner IS a
# RayCasterCfg, RayCasterCfg IS a SensorBaseCfg subclass, and SensorBaseCfg is checked at
# line 947, well before the raise at 994. Nothing about that message points at import
# order, which is why it survived three separate wrong diagnoses (class identity, branch
# order, GPU contention, a stale clone) before a probe that instantiated the launcher
# first -- and therefore imported second -- passed the identical build.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import booster_train.tasks  # noqa: F401,E402
import k1_velocity.tasks.velocity  # noqa: F401,E402
import k1_velocity.tasks.push  # noqa: F401,E402 — P6 push family


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    import gymnasium as gym

    env_cfg.scene.num_envs = args_cli.num_envs
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))

    agent_cfg.max_iterations = 1
    agent_cfg_dict = agent_cfg.to_dict()
    for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
        for mk in ("actor", "critic"):
            agent_cfg_dict[mk].pop(key, None)

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir="/tmp/k1_zero_actor",
                            device=agent_cfg.device)
    policy = runner.alg.get_policy()
    policy.eval()
    print(f"[INFO] actor built: obs_groups={agent_cfg.obs_groups.get('actor')} "
          f"action_dim={env.num_actions}")

    out = os.path.abspath(args_cli.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    runner.save(out)
    print(f"[ZERO_ACTOR_SAVED] {out} (untrained, default init)")

    if args_cli.export:
        try:
            jit = policy.as_jit()
            jit = getattr(jit, "cpu", lambda: jit)()
            torch.save(jit, args_cli.export)
            print(f"[ZERO_ACTOR_TS_SAVED] {args_cli.export}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] TorchScript export skipped: {type(exc).__name__}: {exc}")

    env.close()
    print("ZERO_ACTOR_DONE")


if __name__ == "__main__":
    main()
    simulation_app.close()