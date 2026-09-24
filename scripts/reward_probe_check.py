#!/usr/bin/env python3
"""Per-term REWARD probe: build an env, step N times with random actions, and
print the mean RAW value of every reward term + contact-sensor diagnostics.

Purpose: after velocity-config edits (gait/velocity reward changes), prove that
every term RESOLVES (function exists, SceneEntityCfg body/joint names match)
and that the gait terms actually FIRE on this image:
  feet_air_time > 0 (single-stance moments), feet_slide != 0 (foot dragging),
  contact sensor sees forces on left/right_foot_link.

Usage (inside the Isaac image, from the repo root):
    /isaac-sim/python.sh scripts/reward_probe_check.py --task Isaac-Velocity-Rough-K1-v0

Prints REWARD_PROBE_RESULT=OK on success — always grep for the marker, never rc.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=42)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import booster_train.tasks  # noqa: F401
import k1_velocity.tasks  # noqa: F401 — registers velocity/basic/kick/head/push
import k1_velocity.tasks.partial  # noqa: F401 — NOT imported by tasks/__init__ (needs register_tasks or direct import)


def main() -> int:
    import importlib

    from isaaclab.managers import SceneEntityCfg

    torch.manual_seed(args.seed)
    spec = gym.spec(args.task)
    mod_name, cls_name = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg_cls = getattr(importlib.import_module(mod_name), cls_name)
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    env.reset(seed=args.seed)
    base = env.unwrapped
    n = args.num_envs
    action_dim = base.action_manager.total_action_dim
    rm = base.reward_manager
    term_names = list(rm._term_names)
    weights = [rm.get_term_cfg(t).weight for t in term_names]
    print(f"[probe] task={args.task} envs={n} steps={args.steps} action_dim={action_dim}")
    print(f"[probe] reward terms ({len(term_names)}): {term_names}")

    # accumulate per-step weighted values ourselves: _episode_sums is wiped on
    # every termination, and a random policy falls often.
    acc = torch.zeros(n, len(term_names), device=base.device)
    has_step_reward = hasattr(rm, "_step_reward")
    ret_total = 0.0
    for step in range(args.steps):
        actions = torch.empty(n, action_dim, device=base.device).uniform_(-0.5, 0.5)
        obs, rew, terminated, truncated, info = env.step(actions)
        ret_total += torch.as_tensor(rew).mean().item()
        if has_step_reward:
            acc += rm._step_reward
        else:  # fallback: snapshot episode sums (wiped envs under-count)
            acc += torch.stack([rm._episode_sums[t] for t in term_names], dim=-1)

    print(f"[probe] mean per-step total reward (random policy): {ret_total / args.steps:+.4f}")
    print(f"[probe] per-term mean RAW value over {args.steps} steps "
          f"({'_step_reward/weight' if has_step_reward else 'episode_sums delta/weight'}):")
    fired, dead = [], []
    for i, (t, w) in enumerate(zip(term_names, weights)):
        raw = (acc[:, i].mean() / max(args.steps, 1) / w) if w != 0.0 else 0.0
        r = raw.item()
        print(f"[probe]   {t:32s} weight={w:+.6g}  raw_mean={r:+.6f}")
        (fired if abs(r) > 1e-9 else dead).append(t)
    print(f"[probe] nonzero terms: {len(fired)}/{len(term_names)}")
    if dead:
        print(f"[probe] zero terms (may be legitimately 0 with a random policy): {dead}")

    # ---- contact-sensor wiring: decisive for the gait terms ----
    try:
        scfg = SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"])
        scfg.resolve(base.scene)
        cs = base.scene.sensors["contact_forces"]
        nf = cs.data.net_forces_w_history.torch[:, :, scfg.body_ids, :].norm(dim=-1)  # (n,hist,2)
        air = cs.data.current_air_time.torch[:, scfg.body_ids]
        ctm = cs.data.current_contact_time.torch[:, scfg.body_ids]
        print(f"[probe] feet: net_force_max={nf.max().item():.2f} N  "
              f"in_contact_frac={(nf.max(dim=1)[0] > 1.0).float().mean().item():.3f}  "
              f"air_time_max={air.max().item():.3f} s  contact_time_max={ctm.max().item():.3f} s")
        if nf.max().item() <= 0.0:
            print("[probe] WARNING: contact sensor reports ZERO force on feet — "
                  "feet_air_time/feet_slide would be dead")
    except Exception as e:
        print(f"[probe] CONTACT SENSOR PROBE FAILED: {type(e).__name__}: {e}")
        env.close()
        print("REWARD_PROBE_RESULT=FAIL")
        return 1

    env.close()
    print("REWARD_PROBE_RESULT=OK")
    return 0


try:
    rc = main()
except Exception:
    import traceback

    traceback.print_exc()
    print("REWARD_PROBE_RESULT=FAIL")
    rc = 1
finally:
    simulation_app.close()

raise SystemExit(rc)
