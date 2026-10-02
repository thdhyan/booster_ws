#!/usr/bin/env python3
"""Keyboard-controlled play script for K1 student policy."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../booster_train_ref/scripts/rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard-controlled K1 student policy play.")
parser.add_argument("--task", default="Isaac-Velocity-Distill-K1-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--export", default="", help="export TorchScript .pt path")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-app imports ----
import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner

from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks
import booster_train.tasks
import k1_velocity.tasks.velocity

import keyboard_cmd  # same directory; X11 keyboard backend


# Keyboard input handler.
#
# Was: ``carb.input.acquire_input_interface()``, which Isaac Sim 6 no longer has
# -- it raised ``AttributeError: module 'carb' has no attribute 'input'`` on the
# first frame, which is what the [DEBUG] prints below used to chase. The reading
# and the ramp now live in keyboard_cmd, which falls back to X11.
class KeyboardController:
    def __init__(self):
        self._impl = keyboard_cmd.VelocityKeyboard(
            max_lin=1.5, max_ang=2.0, on_quit=simulation_app.close
        )
        print(f"[DEBUG] keyboard backend: {self._impl.state.backend}")
        self.cmd = torch.zeros(3)  # lin_x, lin_y, ang_z
        self.max_lin = 1.5
        self.max_ang = 2.0

    def update(self, dt):
        values = self._impl.update(dt)
        self.cmd = torch.tensor(values, dtype=self.cmd.dtype, device=self.cmd.device)
        return self.cmd

    # The ramp, decay, Space-to-stop and Esc-to-quit logic used to live here,
    # reading carb.input directly. It is now keyboard_cmd.VelocityKeyboard; see
    # that module for why carb.input cannot be used on Isaac Sim 6.


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    from isaaclab.envs.common import ViewerCfg
    env_cfg.viewer = ViewerCfg(eye=(2.5, 0.0, 1.5), lookat=(0.0, 0.0, 0.5))

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print("[DEBUG] keyboard backend resolved by keyboard_cmd, not carb.input")

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
    # Control dt. ``decimation`` moved from the env to the cfg in Isaac Lab 2.1+,
    # so reading it off the env raised
    # ``AttributeError: 'ManagerBasedRLEnv' object has no attribute 'decimation'``.
    # ``step_dt`` is the composed control timestep and is the honest thing to feed
    # the key ramp anyway; fall back to physics_dt * cfg decimation if absent.
    dt = getattr(env.unwrapped, "step_dt", None)
    if dt is None:
        dt = env.unwrapped.physics_dt * env_cfg.decimation

    # ``CommandManager`` has no ``set_command`` in Isaac Lab 2.1+ (only
    # ``get_command``/``get_term``), so the old call raised
    # ``AttributeError: 'CommandManager' object has no attribute 'set_command'``.
    # ``get_command`` returns the live tensor the term samples from, so writing
    # into it in place is both the supported path and the one that also stops the
    # 8-12 s resample timer from overwriting the keyboard command between frames.
    cmd_buf = env.unwrapped.command_manager.get_command("base_velocity")

    with torch.inference_mode():
        while simulation_app.is_running():
            cmd = kb.update(dt)
            cmd_buf[:] = cmd.to(cmd_buf.device, cmd_buf.dtype)

            actions = policy({"policy": obs["policy"]})
            obs, rew, dones, _ = env.step(actions)

            if dones.any():
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
    simulation_app.close()
