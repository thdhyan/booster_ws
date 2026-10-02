"""Keyboard teleop for a TEACHER (PPO) velocity checkpoint.

Added 2026-10-02. The two existing keyboard scripts --
``play_keyboard.py`` and ``play_keyboard_fixed.py`` -- both build a
``DistillationRunner`` and load with ``load_cfg={"student": True}``, so they only
accept distilled STUDENT checkpoints. A PPO teacher checkpoint has a different
runner, a different model class, and a different observation group, so both
scripts reject it::

    size mismatch for mlp.0.weight: copying a param with shape
    torch.Size([512, 237]) from checkpoint, the shape in current model is
    torch.Size([512, 50])

This script drives the teacher with the same keyboard backend
(``keyboard_cmd.VelocityKeyboard``, X11 via python-xlib with a carb.input
fallback) so W/S/A/D/Q/E/SPACE behave identically.

EVALUATION ONLY -- the teacher actor reads the 237-dim privileged group, which
includes a 187-ray terrain height scan. No real robot has that, so this policy is
NOT deployable; a deployable policy comes from the distilled 50-dim student and
uses ``play_keyboard_fixed.py``.

Usage:
    python play_keyboard_teacher.py \\
        --task Isaac-Velocity-Rough-K1-Teacher-Play-v0 \\
        --checkpoint models/k1_teacher_6321_gains.pt --num_envs 1
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Keyboard teleop for a K1 teacher checkpoint.")
# The zz-bw container wrapper (`run_k1_train.sh`) always appends --headless.
# Strip it here so one command line works both headless (smoke / CI) and with a
# window (real keyboard use).
if "--headless" in sys.argv:
    sys.argv.remove("--headless")
parser.add_argument("--task", default="Isaac-Velocity-Rough-K1-Teacher-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
# Isaac Lab 3.0-EA REQUIRES a visualizer selection or it silently forces headless:
#   "No visualizer was selected, so running in headless mode. To launch a
#    visualizer app, pass '--viz <names>' (for example '--viz kit')."
# This script builds its own parser, so without add_app_launcher_args() there is
# no --viz at all and no window ever opens. Default to kit so the keyboard lab
# is a GUI by default; pass --viz none for a headless smoke.
# Isaac Lab 3.0-EA forces headless unless a Kit visualizer is selected:
#   "No visualizer was selected, so running in headless mode. To launch a
#    visualizer app, pass '--viz <names>' (for example '--viz kit')."
#
# --viz CANNOT be defaulted in code. It is registered as
#     arg_group.add_argument("--visualizer", "--viz", action=ExplicitAction, ...)
# and ExplicitAction sets "<dest>_explicit" only when the option is genuinely
# typed on the command line. _resolve_headless_settings then takes the
# "no CLI visualizer selection" branch and forces headless anyway.
#
# So the launcher script must pass --viz kit on the command line. That is safe
# with parse_known_args: --viz is a known option, so it lands in args_cli and is
# absent from hydra_args -- Hydra never sees it, unlike putting it straight into
# sys.argv.
AppLauncher.add_app_launcher_args(parser)
# parse_known_args + argv reset, matching play_record.py / play_keyboard_fixed.py.
# Hydra re-parses sys.argv when @hydra_task_config is applied, and its parser
# knows nothing about --task/--checkpoint/--num_envs, so leaving them in argv
# makes Hydra exit with
#   "unrecognized arguments: --task --checkpoint ... --num_envs 1".
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args



app_launcher = AppLauncher(args_cli=args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import booster_train.tasks  # noqa: F401,E402
import k1_velocity.tasks.velocity  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keyboard_cmd  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
    from isaaclab.envs.common import ViewerCfg

    env_cfg.viewer = ViewerCfg(eye=(2.5, 0.0, 1.4), lookat=(0.0, 0.0, 0.55))

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg.max_iterations = 1
    agent_cfg_dict = agent_cfg.to_dict()
    # rsl_rl 5.x dropped these from MLPModel but isaaclab_rl 3.0.0b2 still
    # serializes them, so strip them under the model sub-dicts the runner reads.
    # Must be ("actor", "critic"): stripping "student"/"teacher" leaves them in
    # place and MLPModel.__init__ rejects `stochastic`.
    legacy_keys = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")
    for model_key in ("actor", "critic"):
        if isinstance(agent_cfg_dict.get(model_key), dict):
            for key in legacy_keys:
                agent_cfg_dict[model_key].pop(key, None)

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir="/tmp/k1_play_keyboard_teacher",
                            device=agent_cfg.device)
    print(f"[INFO] loading TEACHER checkpoint: {args_cli.checkpoint}")
    runner.load(os.path.abspath(args_cli.checkpoint),
                load_cfg={"optimizer": False, "iteration": True})
    policy = runner.alg.get_policy()
    policy.eval()

    kb = keyboard_cmd.VelocityKeyboard()
    print(f"[DEBUG] keyboard backend: {kb.state.backend}")
    print("Keyboard controls:")
    print("  W/S - forward/backward")
    print("  A/D - left/right strafe")
    print("  Q/E - yaw left/right")
    print("  SPACE - stop")
    print("  ESC - quit")
    print("NOTE: teacher policy = privileged obs (height scan). EVALUATION ONLY.")

    obs, _ = env.reset()
    dt = getattr(env.unwrapped, "step_dt", None)
    if dt is None:
        dt = env.unwrapped.physics_dt * env_cfg.decimation

    # Write the command in place: CommandManager has no set_command in Isaac Lab
    # 2.1+, and writing the live tensor also stops the 8-12 s resample timer from
    # overwriting the keyboard command between frames.
    cmd_buf = env.unwrapped.command_manager.get_command("base_velocity")

    with torch.inference_mode():
        last_print = 0.0
        clock = 0.0
        while simulation_app.is_running():
            # keyboard_cmd.VelocityKeyboard.update() returns list[float]; the
            # distilled scripts wrap it in a controller that hands back a tensor.
            # Coerce here instead, once, rather than at each use site.
            cmd = torch.as_tensor(kb.update(dt), dtype=cmd_buf.dtype,
                                  device=cmd_buf.device).reshape(1, -1)[:, :3]
            cmd_buf[:] = cmd
            # Pass the WHOLE obs dict, not {"policy": ...}. This is a PPO policy
            # whose obs_groups are {"actor": ["teacher"]}, so the model indexes
            # obs["teacher"] itself -- handing it a dict keyed only by "policy"
            # raises KeyError: 'teacher'. play_record.py does the same thing for
            # the non-distillation branch (`return policy(obs)`).
            actions = policy(obs)
            obs, rew, dones, _ = env.step(actions)
            clock += dt
            # Live telemetry: without it you cannot tell whether the policy is
            # tracking the keys you are pressing, or drifting, or falling.
            if clock - last_print >= 0.5:
                last_print = clock
                base = env.unwrapped.scene["robot"]
                lin = base.data.root_lin_vel_w[0, :2]
                ang = base.data.root_ang_vel_w[0, 2]
                z = base.data.root_pos_w[0, 2]
                want = cmd[0].cpu()
                err = (lin.cpu() - want[:2]).norm().item()
                print(
                    f"[cmd] vx={want[0]:+.2f} vy={want[1]:+.2f} wz={want[2]:+.2f} | "
                    f"[meas] vx={lin[0]:+.2f} vy={lin[1]:+.2f} wz={ang:+.2f} | "
                    f"err={err:.2f} m/s  trunk_z={z:.3f} m",
                    flush=True,
                )
            if dones.any():
                print("[teleop] episode reset", flush=True)
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
    simulation_app.close()