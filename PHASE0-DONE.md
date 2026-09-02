# Phase 0 — What Was Built

**Branch:** `dev/phase-0`  
**Commits:** `faefa3e` (USD flatten) · `89b13a9` (kick task)  
**Date:** 2026-08-28  

---

## Task A — USD Flatten + Contact Rewards (`faefa3e`)

### Problem
Isaac Sim 6.0 URDF importer nests child link prims under parents
(`/Robot/Trunk/Arm_1/foot_link`). Isaac Lab 3.0.0b2 `ContactSensor`
assumes a **flat** hierarchy — it only matched the root `Trunk` body,
never the foot links. Upstream fix (PR #6378) landed on `develop`, not
in pinned 3.0.0b2. We do not patch the wheel.

### Solution
Flatten the K1 USD at the prim level so all rigid bodies sit as siblings
under the robot root. Contact sensors then address them correctly.

### Files

| File | What it does |
|------|-------------|
| `scripts/flatten_k1_usd.py` | Traverses the URDF-converted USD, finds all `UsdPhysics.RigidBodyAPI` prims, reparents them to robot root preserving world pose, applies `PhysxContactReportAPI` on all 23 bodies, writes `src/k1_description/assets/robots/K1/K1_flat.usd` (4.2 MB). Idempotent. |
| `isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_env_cfg.py` | Re-enabled `ContactSensorCfg` (prim_path `{ENV_REGEX_NS}/Robot/.*`, history_length=3, track_air_time=True, filter `/World/ground`). Re-added `feet_air_time` + `feet_slide` rewards matching G1 weights. Added `foot_contact` obs to teacher group only (student stays contact-free). Removed all "contact sensors unavailable" stub comments. |

### To train with contact rewards on zz-bw
```bash
# Clear cached clone so container picks up the new USD + env changes
ssh zz-bw 'rm -rf /export/scratch/thakk100/k1/tmp/booster_ws'

# Teacher (velocity + contact rewards)
ssh zz-bw 'tmux new -d -s k1vel_contact "K1_GPU=1 ~/run_k1_train.sh \
  --task Isaac-Velocity-Flat-K1-v0 \
  --num_envs 4096 --headless --max_iterations 5000"'

# Student (distillation, contact-free obs)
ssh zz-bw 'tmux new -d -s k1vel_student "K1_GPU=1 ~/run_k1_train.sh \
  --task Isaac-Velocity-Flat-K1-v0-Play \
  --num_envs 4096 --headless --max_iterations 3000"'
```

---

## Task B — Kick Task `Isaac-Kick-Ball-K1-v0` (`89b13a9`)

### Design
State-based, no vision. K1 learns to kick a dynamic soccer ball into a
goal using teacher–student distillation. Teacher sees privileged ball
state; student is blind (proprio only) after distillation.

### MDP

| Element | Spec |
|---------|------|
| **Robot** | `BOOSTER_K1_CFG`, 12 leg joints, `JointPositionActionCfg` scale 0.25 |
| **Ball** | `RigidObject`, radius 0.11 m, mass 0.43 kg, init (0,0,0.111) |
| **Goal** | Kinematic posts + crossbar at x=+4 m, width 2 m, height 0.9 m |
| **Policy obs** | 48-dim proprio: base lin/ang vel, gravity, joint pos/vel ×12, last action |
| **Teacher obs** | Policy + ball pos (3) + ball lin vel (3) in robot frame, no noise |
| **Episode length** | 20 s (200 Hz physics, 50 Hz control) |

**Rewards:**

| Term | Weight | Notes |
|------|--------|-------|
| `goal_scored` | +100 | Sparse; ball past goal line within goal volume |
| `ball_to_goal_progress` | +1 | Distance decrease normalised to [0,1] |
| `flat_orientation_l2` | −1 | Gait regularisation |
| `action_rate_l2` | −0.005 | Smoothness |
| `dof_torques_l2` | −1.5e-7 | Energy |
| `joint_pos_limits` | −1 | Ankle safety |
| `joint_deviation_arms` | −0.05 | Arms/head near default |
| `termination_penalty` | −200 | Early fall |

**Terminations:** `time_out` · `root_height < 0.35 m` · `bad_orientation > 0.8 rad` · `goal_scored`

### OmniReset
Three reset families, population-sampled each episode:

| Family | Ratio | Ball placement |
|--------|-------|---------------|
| `at_ball_shoot` | 50 % | Ball 0.3–0.6 m ahead of robot, small y jitter |
| `stand_ready` | 30 % | Ball at origin, robot at default stance |
| `walk_up` | 20 % | Ball 2–3.5 m ahead, ±1 m y cone |

Implemented in `reset_ball_omnireset` (single `EventTerm`, mode=`reset`).
Uses `write_root_state_to_sim()` with env origin offset — direct
`data.root_pos_w` assignment was the original bug (buffer-only, physics
not updated).

### Files

| File | Purpose |
|------|---------|
| `source/k1_velocity/tasks/kick/kick_env_cfg.py` | Full env config: scene, obs, actions, rewards, terminations, events |
| `source/k1_velocity/tasks/kick/mdp.py` | Ball obs (robot frame), goal detection, progress reward, OmniReset sampler |
| `source/k1_velocity/tasks/kick/agents/rsl_rl_ppo_cfg.py` | PPO config (rsl_rl 5.x `actor=`/`critic=` schema) |
| `source/k1_velocity/tasks/kick/agents/rsl_rl_distill_cfg.py` | Distillation config |
| `source/k1_velocity/tasks/kick/__init__.py` | `gym.register` for `Isaac-Kick-Ball-K1-v0` |
| `scripts/train_kick.py` | Teacher training entry point |
| `scripts/train_kick_student.py` | Student distillation; includes `output_std` class-property guard |

### Bugs fixed after agent commit
| Bug | Fix |
|-----|-----|
| `ball_to_goal_progress` returned `(N,1)` not `(N,)` | Removed `keepdim=True` from both `torch.norm` calls |
| OmniReset families defined but never called | Replaced 3 broken individual funcs with `reset_ball_omnireset` wired into `EventCfg` |
| Ball reset wrote to data buffer only | Now uses `default_root_state + env_origins + write_root_state_to_sim()` |

### To train on zz-bw
```bash
# Teacher
ssh zz-bw 'tmux new -d -s k1kick "K1_GPU=1 ~/run_k1_train.sh \
  --task Isaac-Kick-Ball-K1-v0 \
  --num_envs 4096 --headless --max_iterations 5000"'

# Student (after teacher checkpoint exists)
ssh zz-bw 'tmux new -d -s k1kick_student "K1_GPU=1 ~/run_k1_train.sh \
  --task Isaac-Kick-Ball-K1-v0 \
  --num_envs 4096 --headless --max_iterations 3000 --distill"'
```

---

## What's not done yet

- **Contact rewards not tested end-to-end** — needs full teacher run on zz-bw with flattened USD to confirm `feet_air_time` / `feet_slide` fire correctly.
- **Kick task curriculum annealing** — OmniReset ratios are fixed (50/30/20). Handoff doc calls for annealing toward uniform over training; not implemented.
- **Ball quaternion in teacher obs** — handoff spec says `ball_pose = pos(3) + quat(4) = 7`. Current impl sends only `pos(3) + lin_vel(3) = 6`. If teacher net input dim mismatches, adjust `rsl_rl_ppo_cfg.py` actor input size or add the quat term to `mdp.ball_pos_in_robot_frame`.
- **3 cameras defined but not wired** — left/right/disparity exist in scene, no obs term reads them.
- **Goal contact detection** — currently geometric (ball x/y/z threshold). Future: `ContactSensor` on goal posts (needs Agent A's flatten work extended to goal structure).
