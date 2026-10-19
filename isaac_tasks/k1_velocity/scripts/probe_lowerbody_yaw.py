"""Yaw-spin probe for the Stage-0 lower-body env (PLAN_HRL_CONTROLLER.md 2b audit).

Lower-body teachers v1/v2 logged error_vel_yaw ~1.0 from iteration 0, while the squat
teacher on the same command/scene logged 0.05-0.2. This runs the env under a ZERO action
(legs held at the default pose) for each ablation variant and prints the trunk yaw rate,
so the term that makes the trunk spin can be found without a training run.

Variants (--variant):
  full     upper pose scale 1.0, deltas 1.0, stub force +-20 N  (end-of-curriculum env)
  start    upper pose scale 0.15, no deltas, no stub force       (iteration-0 env)
  noupper  upper randomization off, stub force off
  nohead   full, but head joints excluded from randomization
  noforce  full, stub force off
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../booster_train_ref/scripts/rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-LowerBody-K1-v0")
parser.add_argument("--variant", default="full")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import booster_train.tasks  # noqa: F401
import k1_velocity.tasks.velocity  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

ARMS = ["ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.events.push_robot = None
    ev, v = env_cfg.events, args_cli.variant
    if hasattr(ev, "upper_pose_random"):
        # the curriculum rewrites these each reset; drop it so the variant holds
        env_cfg.curriculum.upper_pose = None
        env_cfg.curriculum.upper_delta = None
        env_cfg.curriculum.stub_force = None
        scale, delta, force = {"full": (1.0, 1.0, 20.0), "start": (0.15, 0.0, 0.0),
                               "noupper": (0.0, 0.0, 0.0), "nohead": (1.0, 1.0, 20.0),
                               "noforce": (1.0, 1.0, 0.0)}[v]
        ev.upper_pose_random.params["curriculum_scale"] = scale
        ev.upper_pose_delta.params["curriculum_scale"] = delta
        ev.upper_pose_delta.interval_range_s = (0.5, 1.0)
        ev.stub_force.params["force_range"] = (-force, force)
        if v == "nohead":
            ev.upper_pose_random.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARMS)
            ev.upper_pose_delta.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARMS)
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    wrapped.reset()
    robot = env.unwrapped.scene["robot"]
    zero = torch.zeros(args_cli.num_envs, wrapped.num_actions, device=wrapped.device)
    wz_abs, wxy_abs, resets = [], [], 0
    for step in range(args_cli.steps):
        out = wrapped.step(zero)
        resets += int(out[2].sum())
        w = robot.data.root_ang_vel_w.torch if hasattr(robot.data.root_ang_vel_w, "torch") else robot.data.root_ang_vel_w
        wz_abs.append(w[:, 2].abs())
        wxy_abs.append(w[:, :2].norm(dim=-1))
    wz = torch.stack(wz_abs)
    wxy = torch.stack(wxy_abs)
    print(f"YAW_PROBE task={args_cli.task} variant={v} envs={args_cli.num_envs} steps={args_cli.steps} "
          f"wz_mean={float(wz.mean()):.3f} wz_p99={float(wz.quantile(0.99)):.3f} wz_max={float(wz.max()):.3f} "
          f"first10_wz_mean={float(wz[:10].mean()):.3f} wxy_mean={float(wxy.mean()):.3f} resets={resets}")
    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
