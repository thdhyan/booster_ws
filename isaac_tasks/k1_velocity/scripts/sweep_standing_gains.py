"""Standing sweep: what leg stiffness lets the K1 hold itself up?

The reward probe showed the K1 sinking monotonically from 0.589 m to 0.076 m
under *zero* action (default joint targets) in ~2.4 s.  No reward tuning fixes a
body that cannot stand, so this measures the actuator gains directly: sweep a
stiffness multiplier on the leg/foot actuators, hold zero action, and report how
high the trunk stays.

AGILE's Booster T1 runs legs at stiffness 100 (hip/knee) and 20 (ankle); K1's
hardware-derived gains are far softer (hip 17.8-30.2, knee 60.4, ankle 35.7).
This finds the standing regime empirically instead of guessing.
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
parser.add_argument("--task", default="Isaac-Velocity-Rough-K1-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--mults", default="1,2,3,4,6,8")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import booster_train.tasks  # noqa: F401
import k1_velocity.tasks.velocity  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402

STAND_Z = 0.57  # K1 nominal trunk height


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    base = env.unwrapped
    robot = base.scene["robot"]

    print("\nACTUATORS:", list(robot.actuators.keys()))
    for name, act in robot.actuators.items():
        try:
            print(f"  {name:6s} stiffness={act.stiffness.detach().cpu().numpy().round(2)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:6s} <no stiffness attr: {exc}>")

    zero = torch.zeros(args_cli.num_envs, wrapped.num_actions, device=wrapped.device)
    mults = [float(m) for m in args_cli.mults.split(",")]

    print("\n" + "=" * 78)
    print(f"STANDING SWEEP  steps={args_cli.steps}  K1 nominal trunk z={STAND_Z} m")
    print("=" * 78)
    print(f"{'mult':>6s} {'stiff@start':>12s} {'z@0':>7s} {'z@min':>7s} "
          f"{'z@end':>7s}  verdict")

    # Snapshot the authored stiffness so each run starts from the same baseline.
    base_stiff = {}
    base_damp = {}
    for name, act in robot.actuators.items():
        try:
            base_stiff[name] = act.stiffness.detach().clone()
            base_damp[name] = act.damping.detach().clone()
        except Exception:  # noqa: BLE001, PERF203
            pass

    results = []
    for mult in mults:
        for name, act in robot.actuators.items():
            if name in base_stiff:
                act.stiffness = base_stiff[name] * mult
                act.damping = base_damp[name] * mult
        base.reset()
        wrapped.reset()
        zs = []
        for _ in range(args_cli.steps):
            wrapped.step(zero)
            zs.append(float(robot.data.root_pos_w[:, 2].mean()))
        z = torch.tensor(zs)
        held = float(z[-1]) > 0.45
        verdict = "STANDS" if held else ("sags" if float(z[-1]) > 0.30 else "COLLAPSES")
        s0 = float((base_stiff["legs"] * mult).mean()) if "legs" in base_stiff else float("nan")
        print(f"{mult:6.1f} {s0:12.1f} {float(z[0]):7.3f} {float(z.min()):7.3f} "
              f"{float(z[-1]):7.3f}  {verdict}")
        results.append((mult, float(z[-1]), verdict))

    print("-" * 78)
    good = [r for r in results if r[2] == "STANDS"]
    if good:
        print(f"STANDING_SWEEP_BEST_MULT={min(good, key=lambda r: r[0])[0]} (lowest that stands)")
    else:
        print("STANDING_SWEEP_BEST_MULT=NONE (no multiplier kept the trunk above 0.45 m)")
    print("STANDING_SWEEP_MARKER=OK")
    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
