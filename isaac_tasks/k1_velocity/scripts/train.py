"""Train K1 velocity locomotion policy with RSL-RL.

Wrapper around booster_train's train.py that registers k1_velocity tasks
before Isaac Lab's hydra task resolver runs.

Usage (from booster_ws root, with venv-isaac active):
    python isaac_tasks/k1_velocity/scripts/train.py \
        --task Isaac-Velocity-Rough-K1-v0 \
        --num_envs 1024 \
        --headless \
        --max_iterations 5000

    # Quick smoke test
    python isaac_tasks/k1_velocity/scripts/train.py \
        --task Isaac-Velocity-Rough-K1-v0 \
        --num_envs 64 \
        --headless \
        --max_iterations 10
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

# Add booster_train scripts dir to path for cli_args import
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "../../../isaac_tasks/booster_train_ref/scripts/rsl_rl"))
# Resolve relative path robustly
_bt_scripts = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../booster_train_ref/scripts/rsl_rl")
if os.path.isdir(_bt_scripts):
    sys.path.insert(0, _bt_scripts)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip  (from booster_train scripts)

# ---------- argparse (identical to booster_train train.py) ----------
parser = argparse.ArgumentParser(description="Train K1 velocity with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-K1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------- Everything below runs AFTER Isaac Sim is up --------

import gymnasium as gym
import os
import torch
from datetime import datetime

import omni
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401 — triggers isaaclab built-in task discovery
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_train.tasks  # noqa: F401 — registers booster_train tasks

# *** Register K1 velocity tasks ***
import k1_velocity.tasks.velocity  # noqa: F401
import k1_velocity.tasks.partial  # noqa: F401 — hierarchical partial-control (legs+head) family

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run,
                                          agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # rsl_rl 5.x MLPModel dropped the deprecated noise kwargs that
    # isaaclab_rl 3.0.0b2's RslRlMLPModelCfg still serializes — strip them.
    agent_cfg_dict = agent_cfg.to_dict()
    legacy_keys = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")
    for model_key in ("actor", "critic"):
        for key in legacy_keys:
            agent_cfg_dict[model_key].pop(key, None)
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir,
                            device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
