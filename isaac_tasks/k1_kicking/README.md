# K1 Kicking Task (scaffolding)

**Status: 📝 cfg only — not registered, not trained, no checkpoints.**

> The *active* kicking line for the soccer HRL campaign is
> [`k1_velocity`'s Kick family](../k1_velocity/README.md#kick-ball)
> (`Isaac-Kick-Ball-K1-{v0,Teacher-v0,Distill-v0}` — ball + kinematic goal,
> OmniReset placement families). This package holds an earlier ball-relative
> kicking attempt kept for reference until it's either registered or retired.

## Contents

| File | Purpose |
|---|---|
| `source/k1_kicking/tasks/kicking/kicking_env_cfg.py` | full env cfg (scene, obs, actions, rewards, terminations) |

There is **no `__init__.py`**, no `gym.register` entry, and no agent runner
cfg — nothing here is importable as a task yet.

## Authored spec (per cfg docstring)

| Aspect | Value |
|---|---|
| Observation | ~48-dim: `ball_distance` 1 + `ball_angle` 2 + `base_lin_vel` 3 + `base_ang_vel` 3 + `projected_gravity` 3 + `joint_pos` 12 + `joint_vel` 12 + `last_action` 12 |
| Action | 12 leg joint positions (kick motion) |
| Rewards | ball forward velocity (primary) + ball contact force + stable approach stance + fall/energy/action-rate penalties |
| Notes | blind proprioception + relative ball features (no camera) |

## To make it trainable

1. Add `source/k1_kicking/tasks/kicking/__init__.py` + package `__init__.py`
   with `gym.register(...)` entries (train/play ids), mirroring
   `k1_velocity/tasks/basic/__init__.py`.
2. Author the agent runner cfg (`agents/rsl_rl_ppo_cfg.py`).
3. Add a `train_kicking.py` (or route through `k1_velocity`'s train script) and
   a 16×3 smoke before any full run — per `TRAINING.md` operating rules.
