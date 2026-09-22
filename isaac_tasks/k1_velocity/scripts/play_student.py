"""Play (eval) the distilled K1 velocity STUDENT policy and export it.

Usage:
  python isaac_tasks/k1_velocity/scripts/play_student.py \
      --task Isaac-Velocity-Distill-K1-v0 \
      --checkpoint logs/rsl_rl/p2_move_student/<run>/model_*.pt \
      --num_envs 20 --steps 500 --headless \
      [--export models/p2_move_student.pt] [--export_onnx models/p2_move_student.onnx]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../booster_train_ref/scripts/rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play K1 velocity student.")
parser.add_argument("--task", default="Isaac-Velocity-Distill-K1-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--export", default="", help="export TorchScript .pt path")
parser.add_argument("--export_onnx", default="", help="export ONNX path")
parser.add_argument("--disable_jit", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-app imports ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import DistillationRunner  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import booster_train.tasks  # noqa: F401,E402
import k1_velocity.tasks.velocity  # noqa: F401,E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg.max_iterations = 1
    agent_cfg_dict = agent_cfg.to_dict()
    for key in ("stochastic", "init_noise_std", "noise_std_type",
                "state_dependent_std"):
        for mk in ("student", "teacher"):
            agent_cfg_dict[mk].pop(key, None)

    runner = DistillationRunner(env, agent_cfg_dict,
                                log_dir="/tmp/k1_play_student",
                                device=agent_cfg.device)
    print(f"[INFO] loading student checkpoint: {args_cli.checkpoint}")
    # load student + optimizer state from a distillation checkpoint
    runner.load(os.path.abspath(args_cli.checkpoint),
                load_cfg={"student": True, "optimizer": True,
                          "iteration": True})
    policy = runner.alg.get_policy()  # the student
    policy.eval()

    # roll (student only — no teacher needed at eval)
    obs, _extras = env.reset()
    torch.backends.cuda.matmul.allow_tf32 = True
    total_rew = torch.zeros(args_cli.num_envs, device=env.device)
    ep_len = torch.zeros(args_cli.num_envs, device=env.device)
    rew_sum, n_steps = 0, 0
    with torch.inference_mode():
        while simulation_app.is_running() and n_steps < args_cli.steps:
            actions = policy({"policy": obs["policy"]})
            obs, rew, dones, _ = env.step(actions)
            total_rew += rew
            ep_len += 1
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                print(f"[step {n_steps:5d}] {len(done_ids):3d} episodes done | "
                      f"reward={total_rew[done_ids].mean().item():7.2f} | "
                      f"len={ep_len[done_ids].mean().item():6.1f}")
                total_rew[done_ids] = 0
                ep_len[done_ids] = 0
            rew_sum += rew.mean().item()
            n_steps += 1

    print("=" * 60)
    print(f"Played {n_steps} steps x {args_cli.num_envs} envs "
          f"(student, history-stacked obs)")
    print(f"Mean step reward: {rew_sum / max(n_steps, 1):.3f}")
    print("=" * 60)

    if args_cli.export:
        jit = policy.as_jit()
        jit = getattr(jit, "cpu", lambda: jit)()
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.export)),
                    exist_ok=True)
        scripted = torch.jit.script(jit)
        scripted.save(args_cli.export)
        print(f"[INFO] TorchScript saved -> {args_cli.export}")
        flat_obs = obs["policy"].cpu()  # already concatenated by the env group
        ref = policy({"policy": flat_obs})
        got = scripted(flat_obs)
        err = (ref.cpu() - got).abs().max().item()
        print(f"[INFO] JIT parity max|Δa| = {err:.2e}")

    if args_cli.export_onnx:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.export_onnx)),
                    exist_ok=True)
        onnx_model = policy.as_onnx(verbose=False).cpu()
        flat_obs = obs["policy"].cpu()
        torch.onnx.export(onnx_model, flat_obs[:1], args_cli.export_onnx)
        print(f"[INFO] ONNX saved -> {args_cli.export_onnx}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
