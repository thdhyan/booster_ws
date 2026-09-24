"""Play + RECORD a trained K1 policy headlessly: mp4 with a live HUD + .npz trace.

The HUD shows the policy's *inputs and outputs* on every frame:
  * velocity commands (or "stand" for the command-less P1 task),
  * all observation groups as value bars (policy / teacher, dim-labelled),
  * the action vector as signed bars,
  * per-step reward + episode reward,
  * task id, checkpoint name, step count / sim time.

A full-fidelity sidecar trace (obs groups, actions, rewards, dones per step,
all envs) is saved as ``*_trace.npz`` for offline analysis, and the policy can
be exported to TorchScript (``--export``) — mirrors ``play.py`` / ``play_student.py`
` loading (PPO vs distillation auto-detected from the agent cfg).

Usage (from booster_ws root, inside the isaac-lab container):
  python isaac_tasks/k1_velocity/scripts/play_record.py \
      --task Isaac-Basic-Teacher-K1-v0 \
      --checkpoint logs/rsl_rl/p1_basic_teacher/<run>/model_6498.pt \
      --num_envs 4 --steps 750 --headless \
      --video_out isaac_tasks/k1_velocity/videos/p1_teacher.mp4 \
      --trace_out isaac_tasks/k1_velocity/videos/p1_teacher_trace.npz \
      --export models/p1_basic_teacher.pt \
      --label "P1 teacher" --eye 3.0,-3.0,1.7 --lookat 0,0,0.55 \
      [--cmd 0.6 0.0 0.5]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../booster_train_ref/scripts/rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play + record a K1 policy with HUD overlay.")
parser.add_argument("--task", required=True, help="registered gym task id")
parser.add_argument("--checkpoint", required=True, help="rsl_rl model_*.pt path")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=750, help="control steps (~50 Hz -> 15 s @ 750)")
parser.add_argument("--video_out", default="", help="output mp4 path")
parser.add_argument("--trace_out", default="", help="output .npz trace path")
parser.add_argument("--export", default="", help="export TorchScript .pt path")
parser.add_argument("--export_onnx", default="", help="export ONNX path")
parser.add_argument("--label", default="", help="HUD title label")
parser.add_argument("--cmd", type=float, nargs=3, default=None,
                    metavar=("VX", "VY", "WZ"),
                    help="pin the velocity command (vx vy wz); also disables heading mode")
parser.add_argument("--eye", default="", help="camera eye 'x,y,z'")
parser.add_argument("--lookat", default="", help="camera lookat 'x,y,z'")
parser.add_argument("--focal", type=float, default=17.0, help="camera focal length (mm)")
parser.add_argument("--width", type=int, default=1024, help="video width px")
parser.add_argument("--height", type=int, default=576, help="video height px")
parser.add_argument("--disable_jit", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# RGB frames need the renderer active (mirrors train.py --video wiring).
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-app imports ----
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import booster_train.tasks  # noqa: F401,E402
import k1_velocity.tasks.velocity  # noqa: F401,E402
import k1_velocity.tasks.basic  # noqa: F401,E402 — P1 BASIC (+P1f) family
import k1_velocity.tasks.partial  # noqa: F401,E402 — partial-control (legs+head) family
import k1_velocity.tasks.kick  # noqa: F401,E402 — kick family (harmless if unused)
import k1_velocity.tasks.head  # noqa: F401,E402 — P3 head-tracking family (harmless if unused)


def _parse_vec3(text: str) -> tuple[float, float, float] | None:
    if not text:
        return None
    x, y, z = (float(v) for v in text.split(","))
    return (x, y, z)


def _load_fonts():
    try:  # Pillow >= 10 accepts a pixel size
        return ImageFont.load_default(size=15), ImageFont.load_default(size=12)
    except TypeError:
        f = ImageFont.load_default()
        return f, f


class Hud:
    """Draws commands / obs / actions / reward over a rendered frame."""

    def __init__(self, label: str, task: str, ckpt: str, num_envs: int,
                 cmd_text: str, dt: float, width: int, height: int):
        self.label = label or task
        self.task = task
        self.ckpt = os.path.basename(ckpt)
        self.num_envs = num_envs
        self.cmd_text = cmd_text
        self.dt = dt
        self.width, self.height = width, height
        self.big, self.small = _load_fonts()

    @staticmethod
    def _bar_row(draw, x0, x1, y0, y1, values, clip=5.0, label=None,
                 font=None, label_w=0):
        """One horizontal row of per-dim signed bars (values clipped to ±clip)."""
        mid = (y0 + y1) // 2
        half = max((y1 - y0) // 2 - 1, 2)
        if label is not None:
            draw.text((x0, y0 - 1), label, fill=(230, 230, 230), font=font)
            x0 += label_w
        vals = np.clip(np.asarray(values, dtype=np.float64), -clip, clip) / clip
        n = len(vals)
        area = max(x1 - x0, 8)
        # center line
        draw.line((x0, mid, x1, mid), fill=(70, 70, 70), width=1)
        for i, v in enumerate(vals):
            bx0 = x0 + int(i * area / n)
            bx1 = max(x0 + int((i + 1) * area / n), bx0 + 1)
            bh = int(abs(v) * half)
            if bh < 1:
                bh = 1
            color = (90, 200, 255) if v >= 0 else (255, 160, 60)
            if v >= 0:
                draw.rectangle((bx0, mid - bh, bx1 - 1, mid), fill=color)
            else:
                draw.rectangle((bx0, mid, bx1 - 1, mid + bh), fill=color)

    def draw(self, frame: np.ndarray, step: int, actions, obs: dict,
             step_reward: float, ep_reward: float) -> np.ndarray:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, "RGBA")
        W, H = img.size
        y0 = int(H * 0.60)
        draw.rectangle((0, y0, W, H), fill=(0, 0, 0, 175))
        # badge (top-left)
        badge = f"{self.ckpt}  |  {self.task}"
        tw = draw.textlength(badge, font=self.small)
        draw.rectangle((4, 4, 8 + tw, 22), fill=(0, 0, 0, 170))
        draw.text((6, 6), badge, fill=(255, 255, 120), font=self.small)

        t = step * self.dt
        draw.text((8, y0 + 4),
                  f"{self.label}   step {step}   t={t:.1f}s   envs={self.num_envs}",
                  fill=(255, 255, 255), font=self.big)
        draw.text((8, y0 + 24),
                  f"cmd: {self.cmd_text}   step reward: {step_reward:+.4f}   "
                  f"episode reward: {ep_reward:+.2f}",
                  fill=(200, 255, 200), font=self.small)

        # actions row (policy outputs, env 0)
        a = np.asarray(actions, dtype=np.float64).ravel()
        avals = " ".join(f"{v:+.2f}" for v in a[:16])
        draw.text((8, y0 + 42),
                  f"actions (env0, {len(a)}-dim): {avals}", fill=(180, 255, 180),
                  font=self.small)
        self._bar_row(draw, 8, W - 8, y0 + 58, y0 + 96, a, clip=1.0,
                      font=self.small)

        # observation rows (one per group, env 0)
        yy = y0 + 104
        for name, vals in obs.items():
            vals = np.asarray(vals, dtype=np.float64).ravel()
            self._bar_row(draw, 8, W - 8, yy, yy + 24, vals, clip=5.0,
                          label=f"obs['{name}'] ({len(vals)}):",
                          font=self.small, label_w=140)
            yy += 30
        return np.asarray(img)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    import inspect

    from isaaclab_visualizers.kit import KitVisualizerCfg

    # -- env sizing --------------------------------------------------------
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

    # -- camera: explicit viewer + Kit visualizer pose (env-relative on EA,
    #    world-origin framing on the isaac-lab image; env 0 sits at the origin)
    eye = _parse_vec3(args_cli.eye)
    lookat = _parse_vec3(args_cli.lookat)
    if eye is not None:
        env_cfg.viewer.eye = eye
    if lookat is not None:
        env_cfg.viewer.lookat = lookat
    vkw = dict(headless=True, focal_length=args_cli.focal,
               window_width=args_cli.width, window_height=args_cli.height)
    if eye is not None:
        vkw["eye"] = eye
    if lookat is not None:
        vkw["lookat"] = lookat
    params = inspect.signature(KitVisualizerCfg).parameters
    vkw = {k: v for k, v in vkw.items() if k in params}
    # EA-only fields: env-relative framing so a moving robot stays centered.
    if "origin_type" in params:
        vkw["origin_type"] = "env"
    if "origin_env_index" in params:
        vkw["origin_env_index"] = 0
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(**vkw)]

    # -- optional command pin (circle path keeps the robot in frame) -------
    cmd_text = "none (stand task)"
    if args_cli.cmd is not None and hasattr(env_cfg, "commands") and \
            hasattr(env_cfg.commands, "base_velocity"):
        vx, vy, wz = args_cli.cmd
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (wz, wz)
        env_cfg.commands.base_velocity.heading_command = False
        cmd_text = f"vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}"

    # -- env + rgb render --------------------------------------------------
    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(gym_env)

    # -- runner: PPO vs distillation (mirrors play.py / play_student.py) ---
    is_distill = getattr(agent_cfg.algorithm, "class_name", "PPO") == "Distillation"
    agent_cfg.max_iterations = 1
    agent_cfg_dict = agent_cfg.to_dict()
    if is_distill:
        for key in ("stochastic", "init_noise_std", "noise_std_type",
                    "state_dependent_std"):
            for mk in ("student", "teacher"):
                agent_cfg_dict[mk].pop(key, None)
        runner = DistillationRunner(env, agent_cfg_dict,
                                    log_dir="/tmp/k1_record_student",
                                    device=agent_cfg.device)
        print(f"[INFO] loading student checkpoint: {args_cli.checkpoint}")
        runner.load(os.path.abspath(args_cli.checkpoint),
                    load_cfg={"student": True, "optimizer": True,
                              "iteration": True})
        # the deployable student consumes only the blind "policy" group
        export_groups = list(agent_cfg.obs_groups.get("student", ["policy"]))
    else:
        for key in ("stochastic", "init_noise_std", "noise_std_type",
                    "state_dependent_std"):
            for mk in ("actor", "critic"):
                agent_cfg_dict[mk].pop(key, None)
        runner = OnPolicyRunner(env, agent_cfg_dict,
                                log_dir="/tmp/k1_record",
                                device=agent_cfg.device)
        print(f"[INFO] loading checkpoint: {args_cli.checkpoint}")
        runner.load(os.path.abspath(args_cli.checkpoint))
        export_groups = list(agent_cfg.obs_groups.get("actor", ["policy"]))
    policy = runner.alg.get_policy()
    policy.eval()

    def infer(obs):
        if is_distill:
            return policy({"policy": obs["policy"]})
        return policy(obs)

    # -- HUD + writers -----------------------------------------------------
    hud = Hud(label=args_cli.label, task=args_cli.task, ckpt=args_cli.checkpoint,
              num_envs=args_cli.num_envs, cmd_text=cmd_text,
              dt=float(getattr(env, "step_dt", 0.02)),
              width=args_cli.width, height=args_cli.height)
    writer = None
    if args_cli.video_out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.video_out)),
                    exist_ok=True)
        fps = int(getattr(env, "metadata", {}).get("render_fps", 50)) or 50
        writer = imageio.get_writer(args_cli.video_out, fps=fps)
        print(f"[INFO] recording -> {args_cli.video_out} ({fps} fps)")

    trace = {k: [] for k in ("actions", "rewards", "dones")}
    obs, _extras = env.reset()
    # obs is a TensorDict (RslRlVecEnvWrapper.reset/step): plain `for k in obs`
    # falls back to the SEQUENCE protocol (obs[0], obs[1]... = batch slices) and
    # never yields the group keys — iterate .keys() explicitly (plain dicts
    # support it too).
    for k in list(obs.keys()):
        trace[f"obs_{k}"] = []
    torch.backends.cuda.matmul.allow_tf32 = True

    ep_reward = torch.zeros(args_cli.num_envs, device=env.device)
    frames_written = 0
    with torch.inference_mode():
        for n in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            actions = infer(obs)
            obs, rew, dones, _ = env.step(actions)
            ep_reward += rew
            ep_reward[dones.bool()] = 0.0

            # ---- capture + overlay ------------------------------------
            frame = gym_env.render()
            if frame is None:
                sys.exit("[ERROR] env.render() returned None — rgb_array "
                         "video_recorder/visualizer not active; cannot record.")
            hud_frame = hud.draw(
                frame, step=n, actions=actions[0].cpu().numpy(),
                obs={k: v[0].cpu().numpy() for k, v in obs.items()},
                step_reward=float(rew[0]), ep_reward=float(ep_reward[0]))
            if writer is not None:
                writer.append_data(hud_frame)
                frames_written += 1

            # ---- trace ------------------------------------------------
            trace["actions"].append(actions.cpu().numpy())
            trace["rewards"].append(rew.cpu().numpy())
            trace["dones"].append(dones.cpu().numpy())
            for k, v in obs.items():
                trace.setdefault(f"obs_{k}", []).append(v.cpu().numpy())

            if n % 100 == 0:
                print(f"[rec {n:5d}/{args_cli.steps}] step_rew={float(rew.mean()):+.4f} "
                      f"ep_rew={float(ep_reward.mean()):+.2f}")

    if writer is not None:
        writer.close()
        print(f"[INFO] video saved -> {args_cli.video_out} "
              f"({frames_written} frames)")
    if args_cli.trace_out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.trace_out)),
                    exist_ok=True)
        # skip empty lists (junk keys from the TensorDict reset-iteration era)
        out = {k: np.stack(v).astype(np.float16) for k, v in trace.items() if v}
        np.savez_compressed(args_cli.trace_out, **out)
        print(f"[INFO] trace saved -> {args_cli.trace_out}")

    # -- TorchScript / ONNX export (same parity checks as play.py) ---------
    if args_cli.export:
        jit = policy.as_jit()
        jit = getattr(jit, "cpu", lambda: jit)()
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.export)),
                    exist_ok=True)
        if args_cli.disable_jit:
            torch.save(jit.state_dict(), args_cli.export)
            print(f"[INFO] raw state-dict saved -> {args_cli.export}")
        else:
            scripted = torch.jit.script(jit)
            scripted.save(args_cli.export)
            print(f"[INFO] TorchScript saved -> {args_cli.export}")
            # Parity: scripted runs on CPU; the LIVE policy may still be on
            # CUDA (as_jit() can return a copy whose .cpu() doesn't move it),
            # so send the ref input to the policy's own device.
            flat_obs = torch.cat([obs[k] for k in export_groups], dim=-1).cpu()
            dev = next(policy.parameters()).device
            ref = policy({k: obs[k].to(dev) for k in export_groups})
            got = scripted(flat_obs)
            err = (ref.cpu() - got).abs().max().item()
            print(f"[INFO] JIT parity max|Δa| = {err:.2e}")

    if args_cli.export_onnx:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.export_onnx)),
                    exist_ok=True)
        onnx_model = policy.as_onnx(verbose=False).cpu()
        flat_obs = torch.cat([obs[k] for k in export_groups], dim=-1).cpu()
        torch.onnx.export(onnx_model, flat_obs[:1], args_cli.export_onnx)
        print(f"[INFO] ONNX saved -> {args_cli.export_onnx}")

    print("=" * 60)
    print(f"Recorded {args_cli.steps} steps x {args_cli.num_envs} envs "
          f"({'distill student' if is_distill else 'ppo'})")
    print("=" * 60)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
