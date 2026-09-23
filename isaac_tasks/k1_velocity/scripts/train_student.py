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
           --teacher_checkpoint logs/rsl_rl/p2_move_teacher/<run>/model_4999.pt \
           --num_envs 4096 --headless

  3. Export the student for deployment:
       python isaac_tasks/k1_velocity/scripts/play_student.py \
           --task Isaac-Velocity-Distill-K1-v0 \
           --checkpoint logs/rsl_rl/p2_move_student/<run>/model_*.pt \
           --num_envs 20 --headless --export models/p2_move_student.pt

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
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1500)
parser.add_argument("--video_interval", type=int, default=4800)
# IL version drift on these two flags: 3.0-EA's AppLauncher DROPPED them (headless
# comes from the HEADLESS env) but our launch recipes still pass them — so on EA we
# must pre-add them (else they fall through to hydra, which rejects unknown args —
# the GUARD_RC=2 failure of the first Path-B smoke). Newer images (beta2/GA) brought
# the flags BACK and now RAISE if the parser already has them, so only add ours when
# the installed AppLauncher doesn't provide them.
_appl_keys = set(getattr(AppLauncher, "_APPLAUNCHER_CFG_INFO", {}))
if "headless" not in _appl_keys:
    parser.add_argument("--headless", action="store_true", default=False,
                        help="IL 3.0-EA reads HEADLESS env, not a flag; accepted + translated")
if "enable_cameras" not in _appl_keys:
    parser.add_argument("--enable_cameras", action="store_true", default=False,
                        help="forwarded to AppLauncher (camera-capable render path)")
cli_args_unused = None  # parity with train.py; extend if needed
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.headless:
    os.environ.setdefault("HEADLESS", "1")

if args_cli.video:
    args_cli.enable_cameras = True

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
# Video wiring differs across Isaac Lab versions: dl's 3.0-EA records config-level
# via isaaclab_rl.entrypoints.common (env_cfg.video_recorders), but the isaac-lab
# image removed entrypoints and records through a gymnasium RecordVideo wrapper
# instead (scripts/reinforcement_learning/common.py:wrap_record_video). Prefer the
# EA helpers when present; the call site falls back to an identical-trigger wrap.
try:
    from isaaclab_rl.entrypoints.common import (  # noqa: E402
        apply_video_recording,
        pre_launch_video_config,
    )
    _EA_VIDEO = True
except (ImportError, ModuleNotFoundError):
    apply_video_recording = pre_launch_video_config = None
    _EA_VIDEO = False
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

    # Same video wiring as isaaclab_rl's train_rsl_rl backend (pre-launch Kit
    # visualizer injection + recorder declaration), both config-level and
    # evaluated before gym.make builds the env. Clip cadence: first clip at
    # step 0, then every video_interval control steps (4800 = 200 iters x 24).
    if args_cli.video and _EA_VIDEO:
        pre_launch_video_config(env_cfg, args_cli=args_cli)
        apply_video_recording(env_cfg, log_dir, args_cli)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if (args_cli.video and not _EA_VIDEO) else None,
    )
    if args_cli.video and not _EA_VIDEO:
        # Mirrors train_rsl_rl.py:wrap_record_video — wrapper applied BEFORE
        # RslRlVecEnvWrapper, same trigger semantics as the EA config-level path.
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

    # rsl_rl 5.x dropped the deprecated noise kwargs that isaaclab_rl 3.0.0b2
    # still serializes — strip them from BOTH model configs (same as train.py).
    agent_cfg_dict = agent_cfg.to_dict()
    legacy_keys = ("stochastic", "init_noise_std", "noise_std_type",
                   "state_dependent_std")
    for model_key in ("student", "teacher"):
        for key in legacy_keys:
            agent_cfg_dict[model_key].pop(key, None)

    # rsl_rl 5.0.1 upstream bug: OnPolicyRunner.learn() (reused by
    # DistillationRunner) unconditionally logs action_std=policy.output_std.
    # The distilled student is deterministic -> distribution=None -> reading
    # the std raises AttributeError deep in the logging path. Override the
    # CLASS property (instance shadowing fails: property has no setter) with
    # a None-guarded version; PPO/stochastic models are unaffected.
    from rsl_rl.models.mlp_model import MLPModel

    def _safe_output_std(self):
        if self.distribution is None:
            return torch.zeros(1, device=self.mlp[0].weight.device)
        return self.distribution.std

    MLPModel.output_std = property(_safe_output_std)

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

    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
