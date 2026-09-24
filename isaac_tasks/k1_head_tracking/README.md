# K1 Head-Tracking Task (scaffolding)

**Status: 📝 cfg only — not registered, not trained, no checkpoints.**

> Planned as **PLAN P3** (`Isaac-HeadTrack-K1-v0` tentative id): turn the head
> toward the ball so the perception stack keeps the target in view. Gates the
> P4 kick runs (see [`TRAINING.md`](../../TRAINING.md) campaign table, run 5).

## Contents

| File | Purpose |
|---|---|
| `source/k1_head_tracking/tasks/head_tracking/head_tracking_env_cfg.py` | full env cfg (scene, obs, actions, rewards, terminations) |

There is **no `__init__.py`**, no `gym.register` entry, and no agent runner
cfg — nothing here is importable as a task yet.

## Authored spec (per cfg docstring)

| Aspect | Value |
|---|---|
| Observation | 11-dim: `ball_angle_to_head` 2 + `head_joint_pos` 2 + `head_joint_vel` 2 + `base_ang_vel` 3 + `last_action` 2 |
| Action | 2-dim: `AAHead_yaw`, `Head_pitch` joint position targets |
| Rewards | track ball angle (exponential kernel) − action-rate − joint-limit penalties |
| Notes | head-only policy; legs frozen (locomotion handled by the base policy) |

## To make it trainable

1. Add `source/k1_head_tracking/tasks/head_tracking/__init__.py` + package
   `__init__.py` with `gym.register(...)` train/play ids, mirroring
   `k1_velocity/tasks/partial/__init__.py`.
2. Author `agents/rsl_rl_ppo_cfg.py` (PPO; frozen legs = P1 teacher final per
   PLAN).
3. 16×3 smoke run first, then the planned 512×2000 on the GPU freed by P2
   (per `TRAINING.md`).
