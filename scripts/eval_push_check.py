#!/usr/bin/env python3
"""P6 checkpoint metrics evaluator: run a trained push/reach policy headlessly
and report box position error, contact success rate, wrist tracking, box
displacement and fall rate. No renderer, no wandb, no video.

Usage (inside the Isaac image, from the repo root):
    /isaac-sim/python.sh scripts/eval_push_check.py \
        --task Isaac-Push-Reach-K1-v0 \
        --checkpoint logs/rsl_rl/p6_push_reach/<run>/model_1499.pt

Metric definitions (all averaged over envs x steps unless noted):
    wrist_tgt_err   ||wrist_base - commanded contact target|| (both wrists)
    wrist_gap       wrist -> box surface gap, box frame (0 = touching)
    contact_both    % steps where BOTH wrists gap < 3 cm  (reach success)
    contact_any     % steps where >=1 wrist gap < 3 cm
    goal_err        ||goal_offset|| — the env's box->goal error by construction
                    (goal = box pose + cumulative offset)
    goal_err_anchor ||spawn + offset - box|| — the anchored reading (how far
                    the box is behind where the walking goal reached)
    box_disp        box travel from its episode spawn (how much got pushed)
    done_rate       episodes ended by fall (20 s timeout > eval length)
    mean_reward     mean per-step extrinsic reward

Prints EVAL_METRIC lines then EVAL_PUSH_RESULT=OK (grep the marker; the rc of
python.sh is unreliable).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../isaac_tasks/booster_train_ref/scripts/rsl_rl"))

parser = argparse.ArgumentParser(description="P6 push metrics evaluator.")
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", default=None, help="rsl_rl model_*.pt path")
parser.add_argument("--zero_actions", action="store_true",
                    help="no policy: output zeros (stand test for the frozen base)")
parser.add_argument("--zero_vel", action="store_true",
                    help="policy actions but velocity slice forced to 0 (arms only)")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=400, help="control steps (~0.02 s each)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--contact_thresh", type=float, default=0.03, help="contact gap [m]")
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab.utils.math import quat_inv, quat_mul  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import booster_train.tasks  # noqa: F401,E402
import k1_velocity.tasks  # noqa: F401,E402 — registers velocity/basic/kick/push
import k1_velocity.tasks.push  # noqa: F401,E402 — be explicit: push family
from k1_velocity.tasks.push import push_mdp as pm  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env)
    base = gym_env.unwrapped

    policy = None
    if not args_cli.zero_actions:
        assert args_cli.checkpoint, "--checkpoint required unless --zero_actions"
        agent_cfg.max_iterations = 1
        agent_cfg_dict = agent_cfg.to_dict()
        for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            for mk in ("actor", "critic"):
                agent_cfg_dict[mk].pop(key, None)
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir="/tmp/k1_eval_push",
                                device=agent_cfg.device)
        print(f"[eval] loading checkpoint: {args_cli.checkpoint}")
        runner.load(os.path.abspath(args_cli.checkpoint))
        policy = runner.alg.get_policy()
        policy.eval()

    n, dev, thr = args_cli.num_envs, env.device, args_cli.contact_thresh
    act_dim = base.action_manager.total_action_dim
    print(f"[eval] mode: "
          + ("ZERO_ACTIONS (frozen base stand test)" if args_cli.zero_actions else
             f"policy ckpt={os.path.basename(args_cli.checkpoint)}"
             + (" + ZERO_VEL slice" if args_cli.zero_vel else "")))

    def wrist_gaps():
        """(n,2) wrist -> box surface gap in the box frame."""
        st = pm._state(base)
        wrist_bf = pm.wrist_positions_base(base)             # (n,2,3) base frame
        corners_bf = pm.box_corners_base(base)               # (n,8,3) base frame
        center = corners_bf.mean(1, keepdim=True)            # (n,1,3)
        half = st.half_extents[:, None, :]                   # (n,1,3)
        box_q_bf = quat_mul(quat_inv(base.scene["robot"].data.root_quat_w.torch),
                            base.scene["box"].data.root_quat_w.torch)
        wrist_box = pm._rot_batch(box_q_bf, wrist_bf - center, inverse=True)
        inside = torch.maximum(torch.minimum(wrist_box, half), -half)
        return (wrist_box - inside).norm(dim=-1)             # (n,2)

    sums, counts = {}, 0
    done_count = 0
    peak = {}

    def accumulate(tag, vals: dict):
        for k, v in vals.items():
            key = f"{tag}.{k}"
            sums[key] = sums.get(key, 0.0) + float(v)
            if tag == "max":
                peak[k] = max(peak.get(k, float("-inf")), float(v))

    obs, _ = env.reset()
    st = pm._state(base)
    box = base.scene["box"].data
    spawn = box.root_pos_w.torch.clone()                     # episode spawn (n,3)

    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.inference_mode():
        for step in range(args_cli.steps):
            if policy is None:
                actions = torch.zeros((n, act_dim), device=dev)
            else:
                actions = policy(obs)
                if args_cli.zero_vel:
                    actions[:, :3] = 0.0
            obs, rew, dones, _ = env.step(actions)
            counts += 1

            st = pm._state(base)
            gap = wrist_gaps()                               # (n,2)
            tgt_err = -pm.wrist_target_tracking(base)        # (n,)
            goal_err = st.goal_offset.norm(dim=-1)           # (n,)
            bpos = base.scene["box"].data.root_pos_w.torch
            disp = (bpos[:, :2] - spawn[:, :2]).norm(dim=-1)
            anchor = (spawn[:, :2] + st.goal_offset[:, :2] - bpos[:, :2]).norm(dim=-1)
            contact_both = (gap < thr).all(-1).float().mean()
            contact_any = (gap < thr).any(-1).float().mean()
            root_z = base.scene["robot"].data.root_pos_w.torch[:, 2]
            stand = (root_z > 0.35).float().mean()

            accumulate("m", {
                "wrist_tgt_err": tgt_err.mean(),
                "wrist_gap": gap.mean(),
                "goal_err": goal_err.mean(),
                "goal_err_anchor": anchor.mean(),
                "box_disp": disp.mean(),
                "reward": rew.mean(),
                "contact_both": contact_both,
                "contact_any": contact_any,
                "root_z": root_z.mean(),
                "stand_frac": stand,
            })
            accumulate("max", {"root_z": root_z.max()})
            accumulate("max", {"box_disp": disp.max(), "goal_err": goal_err.max()})

            d = dones.bool()
            done_count += int(d.sum())
            # done envs reset within the step: re-anchor their spawn
            if d.any():
                bpos = base.scene["box"].data.root_pos_w.torch
                spawn[d] = bpos[d]

            if step % 100 == 0:
                print(f"[eval] step {step:4d}: gap={gap.mean():.3f}m "
                      f"tgt_err={tgt_err.mean():.3f}m contact_both={contact_both:.2f} "
                      f"goal_err={goal_err.mean():.3f}m box_disp={disp.mean():.3f}m "
                      f"rew={rew.mean():+.3f}")

    m = {k[2:]: v / counts for k, v in sums.items() if k.startswith("m.")}
    print(f"[eval] ckpt={os.path.basename(args_cli.checkpoint) if args_cli.checkpoint else 'ZERO_ACTIONS'} task={args_cli.task} "
          f"envs={n} steps={counts} contact_thr={thr}m")
    print(f"[eval] wrist_tgt_err mean={m['wrist_tgt_err']:.4f} m")
    print(f"[eval] wrist_gap      mean={m['wrist_gap']:.4f} m")
    print(f"[eval] contact_both   rate={m['contact_both']:.4f}  (BOTH wrists < {thr} m = success)")
    print(f"[eval] contact_any    rate={m['contact_any']:.4f}")
    print(f"[eval] goal_err       mean={m['goal_err']:.4f} m (env: ||goal_offset||) "
          f"peak={peak.get('goal_err', float('nan')):.4f} m")
    print(f"[eval] goal_err_anchor mean={m['goal_err_anchor']:.4f} m (spawn+offset-box)")
    print(f"[eval] box_disp       mean={m['box_disp']:.4f} m peak={peak.get('box_disp', float('nan')):.4f} m")
    print(f"[eval] reward         mean={m['reward']:.4f} /step")
    print(f"[eval] root_z         mean={m['root_z']:.4f} m peak={peak.get('root_z', float('nan')):.4f} m")
    print(f"[eval] stand_frac     {m['stand_frac']:.4f}  (root_z > 0.35)")
    print(f"[eval] done_rate      {done_count / (n * counts):.4f} (falls; 20 s timeout > eval)")
    print("EVAL_METRIC "
          f"ckpt={os.path.basename(args_cli.checkpoint) if args_cli.checkpoint else 'ZERO_ACTIONS'} "
          f"wrist_tgt_err={m['wrist_tgt_err']:.4f} wrist_gap={m['wrist_gap']:.4f} "
          f"contact_both={m['contact_both']:.4f} contact_any={m['contact_any']:.4f} "
          f"goal_err={m['goal_err']:.4f} goal_err_anchor={m['goal_err_anchor']:.4f} "
          f"box_disp={m['box_disp']:.4f} reward={m['reward']:.4f} "
          f"root_z={m['root_z']:.4f} stand_frac={m['stand_frac']:.4f} "
          f"done_rate={done_count / (n * counts):.4f}")
    print("EVAL_PUSH_RESULT=OK")
    env.close()


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    print("EVAL_PUSH_RESULT=FAIL")
finally:
    simulation_app.close()
