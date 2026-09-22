#!/usr/bin/env python3
"""Keyboard-controlled play script for K1 student policy.

Controls:
  W/S: forward/backward linear velocity (x)
  A/D: left/right linear velocity (y)
  Q/E: yaw angular velocity (z)
  Space: stop (zero velocity)
  Esc: quit

Usage:
  python play_keyboard.py --checkpoint models/k1_velocity_student.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../booster_train_ref/scripts/rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard-controlled K1 student policy play.")
parser.add_argument("--task", default="Isaac-Velocity-Distill-K1-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--export", default="", help="export TorchScript .pt path")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-app imports ----
import carb
import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner

from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks
import booster_train.tasks
import k1_velocity.tasks.velocity


# Keyboard input handler using carb
class KeyboardController:
    def __init__(self):
        self.input = carb.input.acquire_input_interface()
        self.keyboard = self.input.get_keyboard()
        self.cmd = torch.zeros(3)  # lin_x, lin_y, ang_z
        self.speed_lin = 1.0
        self.speed_ang = 1.0
        self.max_lin = 1.5
        self.max_ang = 2.0

    def update(self, dt):
        if not self.keyboard:
            return self.cmd
        # W/S - forward/back
        if self.input.is_key_pressed(self.keyboard, carb.input.KeyboardInput.W):
            self.cmd[0] = min(self.cmd[0] + self.speed_lin * dt, self.max_lin)
        elif self.input.is_key_pressed(self.keyboard, carb.input.KeyboardInput.S):
            self.cmd[0] = max(self.cmd[0] - self.speed_lin * dt, -self.max_lin)
        else:
            self.cmd[0] *= 0.95  # decay to zero

        # A/D - left/right strafe
        if self.input.is_key_pressed(self.keyboard, carb.input.KeyboardInput.A):
            self.cmd[1] = min(self.cmd[1] + self.speed_lin * dt, self.max_lin)
        elif self.input.is_key_pressed(self.keyboard, carb.input.KeyboardInput.D):
            self.cmd[1] = max(self.cmd[1] - self.speed_lin * dt, -self.max_lin)
        else:
            self.cmd[1] *= 0.95

        # Q/E - yaw
        if self.input.is_key_pressed(self.keyboard, carb.input.KeyboardInput.Q):
            self.cmd[2] = min(self.cmd[2] + self.speed_ang * dt, self.max_ang)
        elif self.input.is_key_pressed(self.keyboard, carb.input.KeyboardInput.E):
            self.cmd[2] = max(self.cmd[2] - self.speed_ang * dt, -self.max_ang)
        else:
            self.cmd[2] *= 0.95

        # Space - stop
        if self.input.is_key_pressed(self.keyboard, carb.input.KeyboardInput.SPACE):
            self.cmd = torch.zeros(3)

        # Esc - quit
        if self.input.is_key_pressed(self.keyboard, carb.input.KeyboardInput.ESCAPE):
            carb.log_info("ESC pressed - quitting")
            simulation_app.close()

        return self.cmd


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    # Ensure viewer is enabled for keyboard control
    env_cfg.viewer = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg.max_iterations = 1
    agent_cfg_dict = agent_cfg.to_dict()
    for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
        for mk in ("student", "teacher"):
            agent_cfg_dict[mk].pop(key, None)

    runner = DistillationRunner(env, agent_cfg_dict,
                                log_dir="/tmp/k1_play_keyboard",
                                device=agent_cfg.device)
    print(f"[INFO] loading student checkpoint: {args_cli.checkpoint}")
    runner.load(os.path.abspath(args_cli.checkpoint),
                load_cfg={"student": True, "optimizer": True, "iteration": True})
    policy = runner.alg.get_policy()
    policy.eval()

    # Keyboard controller
    kb = KeyboardController()
    print("Keyboard controls:")
    print("  W/S - forward/backward")
    print("  A/D - left/right strafe")
    print("  Q/E - yaw left/right")
    print("  SPACE - stop")
    print("  ESC - quit")

    obs, _extras = env.reset()
    dt = env.unwrapped.physics_dt * env.unwrapped.decimation  # control dt

    # Fix: Distillation env uses "policy" obs group
    with torch.inference_mode():
        while simulation_app.is_running():
            cmd = kb.update(dt)
            # Inject command into observation's "policy" group
            # The env expects commands in obs["policy"] or similar
            # For distill env, the command is part of the observation
            # We need to set it via the command manager
            if hasattr(env.unwrapped, "command_manager"):
                env.unwrapped.command_manager.set_command(cmd.unsqueeze(0))

            # Policy expects {"policy": obs_tensor} for distill env
            actions = policy({"policy": obs["policy"]})
            obs, rew, dones, _ = env.step(actions)

            if dones.any():
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
    simulation_app.close()