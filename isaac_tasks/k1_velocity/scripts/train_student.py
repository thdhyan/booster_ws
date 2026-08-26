"""Train the K1 velocity STUDENT policy via teacher-student distillation.

Stage 2 of the teacher-student regime (run AFTER the privileged teacher has
been trained):

  1. Train teacher (privileged: height scan + noise-free obs):
       python isaac_tasks/k1_velocity/scripts/train.py \
           --task Isaac-Velocity-Rough-K1-Teacher-v0 \
           --num_envs 4096 --headless --max_iterations 5000

  2. Train student (this script — distills teacher into 480-dim proprio
     history):
       python isaac_tasks/k1_velocity/scripts/train_student.py \
           --task Isaac-Velocity-Distill-K1-v0 \
           --teacher_checkpoint logs/rsl_rl/k1_velocity_teacher/<run>/model_4999.pt \
           --num_envs 4096 --headless

  3. Export the student for deployment:
       python isaac_tasks/k1_velocity/scripts/play_student.py \
           --task Isaac-Velocity-Distill-K1-v0 \
           --checkpoint logs/rsl_rl/k1_velocity_student/<run>/model_*.pt \
           --num_envs 20 --headless --export models/k1_velocity_student.pt

The student consumes the "policy" obs group history-stacked to 480 dims
(term-major, oldest->newest) — the same layout k1_locomotion's node produces
with `input_mode:=stacked`.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../booster_train_ref/scripts/rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distill K1 velocity student.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Distill-K1-v0")
parser.add_argument("--teacher_checkpoint", type=str, required=True,
                    help="path to the trained teacher PPO checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
cli_args_unused = None  # parity with train.py; extend if needed
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-app imports ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from datetime import datetime  # noqa: E402

from rsl_rl.runners import DistillationRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import booster_train.tasks  # noqa: F401,E402
import k1_velocity.tasks.velocity  # noqa: F401,E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.seed += app_launcher.local_rank
        env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl",
                                                 agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # rsl_rl 5.x dropped the deprecated noise kwargs that isaaclab_rl 3.0.0b2
    # still serializes — strip them from BOTH model configs (same as train.py).
    agent_cfg_dict = agent_cfg.to_dict()
    legacy_keys = ("stochastic", "init_noise_std", "noise_std_type",
                   "state_dependent_std")
    for model_key in ("student", "teacher"):
        for key in legacy_keys:
            agent_cfg_dict[model_key].pop(key, None)

    runner = DistillationRunner(env, agent_cfg_dict, log_dir=log_dir,
                                device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # Load the frozen teacher (auto-detected: PPO checkpoint with
    # actor_state_dict -> loaded into alg.teacher, teacher_loaded=True).
    # strict=False: PPO checkpoints carry a Gaussian "distribution.std_param"
    # key the deterministic teacher MLPModel doesn't have — the MLP weights
    # (what we distill) load fine; the teacher then acts on its mean.
    teacher_path = os.path.abspath(args_cli.teacher_checkpoint)
    print(f"[INFO] loading teacher from: {teacher_path}")
    runner.load(teacher_path, strict=False)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # rsl_rl 5.0.1 upstream bug: OnPolicyRunner.learn() (reused by
    # DistillationRunner) unconditionally logs action_std=policy.output_std.
    # The distilled student is deterministic -> distribution=None -> the
    # output_std property raises AttributeError internally, which falls
    # through to nn.Module.__getattr__ ("has no attribute 'output_std'").
    # Shadow it with a dummy instance attribute so logging is a no-op.
    student_policy = runner.alg.get_policy()
    if getattr(student_policy, "distribution", None) is None:
        student_policy.output_std = torch.zeros(
            env.unwrapped.num_actions, device=agent_cfg.device
        )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
