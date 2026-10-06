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
# Every 1000 iterations, not 2000: at 42 000 iterations (1M steps) a 2000
# interval yields 21 clips and a 1000 interval yields 42, which is what the
# gate review actually needs to see a stride change.
parser.add_argument("--video_interval", type=int, default=1000)
# Recording is opt-IN during training. Measured on zz-bw at 4096 envs:
# collection time ran 14 s -> 23 s over ~30 iterations WITH --video, against
# 1.14 s/iter for the same run without it. `--video` sets render_mode below,
# which makes the renderer capture a frame every control step whether or not a
# clip is being written -- the 20x is paid on the other 990 steps of every
# interval. Off by default so a long run is not throttled by the recorder;
# --video still forces it when asked, for short runs where the cost is moot.
parser.add_argument(
    "--video_during_training",
    action="store_true",
    default=False,
    help="allow --video to attach a recorder (see --video for the measured cost)",
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-K1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument(
    "--safe_resume",
    action="store_true",
    default=False,
    help="use a fixed low LR and tighter gradient clip for a short checkpoint recovery",
)
parser.add_argument(
    "--warm_start",
    action="store_true",
    default=False,
    help=(
        "with --checkpoint, load actor+critic only: fresh optimizer and the "
        "iteration counter reset to 0. Use this to fine-tune an existing policy "
        "under a CHANGED reward -- a plain resume restores Adam moments from the "
        "old reward and starts the iteration counter at the checkpoint's, so "
        "--max_iterations buys no budget."
    ),
)
parser.add_argument(
    "--vel_success_threshold",
    type=float,
    default=None,
    help=(
        "Override the velocity-curriculum gate: the fraction of an episode the robot "
        "must survive before the commanded velocity range widens. The default (0.80) is "
        "the legged_gym standard; 0.60 was needed here because the first full run "
        "reached only ~0.53 in 3000 iterations and the gate never opened."
    ),
)
# NOTE: --checkpoint comes from AppLauncher (do not re-add: duplicate option)
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
import ppo_guard
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
import k1_velocity.tasks.kick  # noqa: F401 — P4 kick family (teacher/student)
import k1_velocity.tasks.head  # noqa: F401 — P3 head-tracking family (detection-based)
import k1_velocity.tasks.push  # noqa: F401 — P6 box-push family

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Long PPO runs died at the tail when one inf value loss wrote NaN into every
# weight; the guard turns that update into a no-op instead of a dead run.
ppo_guard.install()
print("[INFO]: PPO_GRAD_GUARD=ON (non-finite gradients are zeroed, never applied)")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.safe_resume:
        if not args_cli.checkpoint:
            raise ValueError("--safe_resume requires --checkpoint")
        # Conservative tail recovery: the source run is already trained and only
        # needs a short, stable finish.  Fixed LR prevents the adaptive schedule
        # from amplifying a late critic spike; tighter clipping protects both nets.
        agent_cfg.algorithm.learning_rate = 1.0e-4
        agent_cfg.algorithm.schedule = "fixed"
        agent_cfg.algorithm.max_grad_norm = 0.5
        agent_cfg.algorithm.value_loss_coef = 0.5
        agent_cfg.algorithm.entropy_coef = 0.001
        print("[INFO]: SAFE_RESUME lr=1e-4 schedule=fixed max_grad_norm=0.5 value_loss_coef=0.5")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Velocity-curriculum gate override.
    #
    # The gate is the fraction of an episode the robot must survive, and it is
    # what decides when the commanded velocity range widens. It needs to be
    # settable per-run because the right value is an empirical question: the
    # first full run reached ~0.53 after 3000 iterations and was still climbing,
    # so the legged_gym-standard 0.80 never opened and that run produced a
    # 0.5 m/s policy rather than a velocity result.
    #
    # Overridable rather than edited into the config so a run's gate is visible
    # in its command line and its log, instead of being a silent property of
    # whatever the config happened to say that day.
    vel_curr = getattr(getattr(env_cfg, "curriculum", None), "velocity_range", None)
    if vel_curr is not None:
        default_thr = vel_curr.params.get("success_threshold")
        if args_cli.vel_success_threshold is not None:
            vel_curr.params["success_threshold"] = args_cli.vel_success_threshold
            print(
                f"[INFO]: VEL_CURRICULUM success_threshold {default_thr} -> "
                f"{args_cli.vel_success_threshold} (episode-length fraction)"
            )
        print(
            f"[INFO]: VEL_CURRICULUM gate={vel_curr.params.get('success_threshold')} "
            f"patience={vel_curr.params.get('patience')} "
            f"interval_steps={vel_curr.params.get('interval_steps')} "
            f"max_lin={vel_curr.params.get('target_max_lin_vel')}"
        )
    elif args_cli.vel_success_threshold is not None:
        print(
            "[WARN]: --vel_success_threshold given but this task has no "
            "curriculum.velocity_range term; ignoring"
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
    if args_cli.checkpoint:
        ckpt = os.path.abspath(args_cli.checkpoint)
        assert os.path.isfile(ckpt), f"checkpoint not found: {ckpt}"
        assert os.path.basename(ckpt).startswith("model_"), f"unexpected ckpt name: {ckpt}"
        agent_cfg.resume = True
        agent_cfg.load_run = os.path.relpath(os.path.dirname(ckpt), log_root_path)
        agent_cfg.load_checkpoint = os.path.splitext(os.path.basename(ckpt))[0]
        print(f"[INFO]: RESUME from {ckpt} (run={agent_cfg.load_run}, ckpt={agent_cfg.load_checkpoint})")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir

    # render_mode="rgb_array" is what makes RecordVideo capture, but it also
    # makes the renderer capture EVERY step -- including the ~990 of every 1000
    # that no clip covers. Measured 14 s -> 23 s collection time per iteration at
    # 4096 envs, versus 1.14 s/iter without --video. So the recorder is opt-in.
    record = args_cli.video and args_cli.video_during_training
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if record else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run,
                                          agent_cfg.load_checkpoint)

    if record:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    elif args_cli.video:
        print(
            "[INFO] --video given without --video_during_training: NOT recording. "
            "The recorder makes the renderer capture every step and cost ~20x "
            "collection time at 4096 envs. Record from play_record.py instead."
        )

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
        if getattr(args_cli, "warm_start", False):
            # Weights only: actor + critic, NO optimizer, and the iteration
            # counter left at 0.
            #
            # Why not a plain resume, when fine-tuning under a CHANGED reward:
            #   * optimizer_state_dict carries Adam's exp_avg/exp_avg_sq, moment
            #     estimates accumulated against the old reward. Those moments are
            #     meaningless once the reward's scale and shape change, and they
            #     are the first thing to bias a warm start.
            #   * `iteration` restores current_learning_iteration, so
            #     --max_iterations buys nothing: learn() runs
            #     range(current_learning_iteration, +num_learning_iterations),
            #     and resuming a model_4999 at max_iterations=5000 starts at 4999.
            #     Resetting to 0 gives the full budget and a clean log.
            #
            # The learning rate needs no help: rsl_rl's PPO adapts it on measured
            # KL (lr /= 1.5 or *= 1.5, clamped to [1e-5, 1e-2]) rather than on
            # iteration index, so a fresh optimizer simply starts at the
            # configured LR and adapts from the first update.
            runner.load(resume_path, load_cfg={
                "actor": True, "critic": True,
                "optimizer": False, "rnd": False, "iteration": False,
            })
            print("[INFO]: WARM_START weights only (actor+critic), fresh optimizer, "
                  "iteration counter reset to 0")
        else:
            runner.load(resume_path)
        if args_cli.safe_resume:
            # runner.load() restores the checkpoint's param_groups, so the
            # conservative LR is re-applied after loading or it is a no-op.
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = agent_cfg.algorithm.learning_rate
            print(f"[INFO]: SAFE_RESUME_LR_APPLIED lr={agent_cfg.algorithm.learning_rate} "
                  f"schedule={agent_cfg.algorithm.schedule}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
