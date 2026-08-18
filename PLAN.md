# Booster K1 ROS2 Workspace — PLAN

> **Robot:** Booster K1 (22 DoF humanoid, 95 cm, 19.5 kg)  
> **Goal:** Full sim+real stack — Gazebo → Isaac Sim → RSL-RL training → SDK deployment → 6-robot fleet  
> **Today:** 2026-08-18

---

## 1. Architecture Overview

```
booster_ws/
├── src/
│   ├── k1_description/      # URDF, meshes (from booster_assets)
│   ├── k1_interfaces/       # Custom msgs / srvs / actions
│   ├── k1_control/          # SDK bridge → joint commands (fastDDS ↔ ROS2)
│   ├── k1_locomotion/       # Velocity cmd → policy inference node
│   ├── k1_wbc/              # Decoupled whole-body controller (loco + arm)
│   ├── k1_sim_gazebo/       # Gazebo worlds, plugins, gz_ros2_control config
│   ├── k1_sim_isaac/        # Isaac Sim ROS2 bridge launch + config
│   └── k1_bringup/          # Top-level launch files for all modes
├── isaac_tasks/
│   └── k1_velocity/         # RSL-RL Isaac Lab task (K1 velocity controller)
│       ├── source/k1_velocity/
│       │   ├── envs/        # K1VelocityEnvCfg (modelled on G1 rough_env_cfg)
│       │   └── agents/      # RSL-RL PPO agent config
│       └── scripts/         # train.py, play.py, export.py
├── scripts/
│   ├── train/               # RSL-RL wrapper scripts
│   └── deploy/              # Export TorchScript → deploy via SDK
├── config/
│   └── robots/k1/           # Per-robot YAML overrides
├── CLAUDE.md
├── PLAN.md
└── TASKS.md
```

---

## 2. Namespacing Strategy

Every node takes `robot_ns` as a launch argument (default: `k1_0`).  
All topics, services, and TF frames are prefixed: `/{robot_ns}/...`

Fleet of N robots → launch N bringup stacks with distinct `robot_ns`:
```
k1_0, k1_1, k1_2, k1_3, k1_4, k1_5
```

This lets all 6 robots run on one ROS2 domain or separate `ROS_DOMAIN_ID`s.

---

## 3. ROS Versions

| Context | ROS version | Location |
|---------|------------|----------|
| All ROS2 packages | System ROS2 (Humble) | `/opt/ros/humble/` |
| Isaac Sim 6.0.1 | IsaacSim bundled ROS2 | `~/Projects/IsaacLab/isaac6/` |

**Rule:** Isaac Sim launch scripts source IsaacSim env. All other packages use system ROS2. Never mix in same shell.

---

## 4. Robot: Booster K1 Joint Map

22 DoF total:

| Group | Joints |
|-------|--------|
| Head (2) | Head_yaw, Head_pitch |
| L Arm (4) | Left_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch, Left_Elbow_Yaw |
| R Arm (4) | Right_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch, Right_Elbow_Yaw |
| L Leg (6) | Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll |
| R Leg (6) | Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw, Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll |

Locomotion policy controls 12 leg joints. WBC adds arm regulation.

---

## 5. Phase Plan

### Phase 0 — Scaffold (NOW)
- [x] PLAN.md, TASKS.md
- [ ] CLAUDE.md
- [ ] `src/` directory tree + package.xml stubs
- [ ] `k1_interfaces` — define all custom msgs/srvs
- [ ] Fetch URDF from `booster_assets` → `k1_description`

### Phase 1 — Gazebo Sim (1 robot)
- [ ] `k1_description`: URDF + ros2_control config
- [ ] `k1_sim_gazebo`: world, spawn, gz_ros2_control
- [ ] `k1_control`: joint state publisher + effort/position command bridge
- [ ] `k1_bringup`: `sim_gazebo.launch.py robot_ns:=k1_0`
- **Verify:** Robot spawns, joints controllable via `ros2 topic pub`

### Phase 2 — Isaac Sim (1 robot)
- [ ] `k1_sim_isaac`: Isaac Sim 6.0.1 USD scene + ROS2 bridge config
- [ ] OmniGraph / ActionGraph for joint pub/sub
- [ ] `k1_bringup`: `sim_isaac.launch.py robot_ns:=k1_0`
- **Uses:** `~/Projects/IsaacLab/isaac6/` env for Isaac-side process
- **Verify:** Robot spawns in Isaac, joint states visible in system ROS2

### Phase 3 — RSL-RL Velocity Training Task
- [ ] `isaac_tasks/k1_velocity/`: Isaac Lab task
  - `K1VelocityRoughEnvCfg` — port from G1 rough_env_cfg, replace robot asset
  - Obs: 72-dim proprioceptive (base vel/ang, joint pos/vel, projected gravity, commands, last action) × 10-step history
  - Actions: 12 leg joint position targets (offsets from default)
  - Rewards: track_lin_vel_xy, track_ang_vel_z, feet_air_time, feet_slide, joint_deviation, termination penalty
  - Terrain curriculum: flat → rough → stairs
  - 4096 parallel envs, PPO via RSL-RL
- [ ] `scripts/train/train_k1_velocity.sh`
- [ ] `scripts/deploy/export_policy.py` → TorchScript
- **Requires:** IsaacLab 2.2 + Isaac Sim 5.0 (from booster_train spec)
  - Note: workspace uses Isaac Sim 6.0.1 for sim/ROS, check compat

### Phase 4 — Policy Deployment
- [ ] `k1_locomotion`: ROS2 node loads TorchScript policy
  - Sub: `/{robot_ns}/cmd_vel` (Twist)
  - Pub: `/{robot_ns}/joint_commands` (JointCommand msg)
  - 50 Hz control loop (0.02s)
- [ ] `k1_control`: bridges JointCommand → SDK `b1_loco_example_client` or to Gazebo/Isaac
- **Verify:** Policy running in Gazebo, robot walks to velocity commands

### Phase 5 — Multi-Robot (6x)
- [ ] Parametric launch: loop N robots with distinct `robot_ns` + port offsets
- [ ] Fleet config: `config/fleet/6robot.yaml`
- [ ] Multi-robot Gazebo world with 6 spawn points
- [ ] Multi-robot Isaac Sim scene

---

## 6. Key External Repos

| Repo | Purpose |
|------|---------|
| [booster_assets](https://github.com/BoosterRobotics/booster_assets) | K1 URDF, USD, motion data |
| [booster_train](https://github.com/BoosterRobotics/booster_train) | Isaac Lab training pipeline, K1 support |
| [booster_gym](https://github.com/BoosterRobotics/booster_gym) | Isaac Gym baseline (older) |
| [booster_robotics_sdk](https://github.com/BoosterRobotics/booster_robotics_sdk) | C++ SDK, fastDDS, real robot control |
| [IsaacLab G1 rough_env_cfg](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py) | Template for K1 velocity task |

---

## 7. Training: RSL-RL K1 Velocity Task Detail

Based on G1 `rough_env_cfg.py` + booster_train pipeline:

```
K1VelocityRoughEnvCfg
  robot_cfg: K1ArticulationCfg (from booster_assets URDF/USD)
  num_envs: 4096
  episode_length_s: 20.0
  decimation: 4  (sim_dt=0.005, ctrl_dt=0.02s)
  
  observations:
    policy:
      base_lin_vel, base_ang_vel, projected_gravity
      joint_pos (12 legs), joint_vel (12 legs)
      velocity_commands (3), last_action (12)
      height_scan (terrain)
  
  actions: JointPositionActionCfg  # 12 leg joints, scale=0.25
  
  rewards:
    track_lin_vel_xy_exp:   1.0
    track_ang_vel_z_exp:    2.0
    feet_air_time:          0.25
    feet_slide:            -0.1
    flat_orientation_l2:   -1.0
    action_rate_l2:        -0.005
    dof_acc_l2:            -1.25e-7
    dof_torques_l2:        -1.5e-7
    termination_penalty: -200.0
  
  terrain: TerrainImporterCfg (curriculum=True, rough)
  
  agent: RSL_RL PPO, 5000 iterations
```

---

## 8. WBC Architecture (Decoupled, from GR00T)

```
cmd_vel (Twist)
    │
    ▼
[k1_locomotion_node]          [k1_arm_controller_node]
  Policy inference               Default/telop arm poses
  12 leg joint targets           10 arm joint targets
    │                               │
    └──────────┬─────────────────────┘
               ▼
        [k1_wbc_node]        ← optional QP solver
          Full 22-joint command
               │
               ▼
     [k1_control] → SDK / sim
```

Phase 1-4: locomotion only (12 joints). Phase 5+: full WBC with arm regulation.

---

## 9. SDK Bridge

`k1_control` wraps `booster_robotics_sdk`:
- SDK uses **fastDDS** internally (not ROS2 DDS)
- Bridge: ROS2 subscriber → SDK C++ client → robot/sim
- For sim: bypass SDK, publish directly to `gz_ros2_control` / Isaac OmniGraph

---

## 10. Git / Version Control Strategy

### Repo Layout

| Repo | GitHub URL | Type | Notes |
|------|-----------|------|-------|
| `booster_ws` | `github.com/thakk100/booster_ws` | **owned** | This workspace — monorepo for all `k1_*` packages + `isaac_tasks/` |
| `booster_assets` fork | `github.com/thakk100/booster_assets` | **fork** | K1 URDF/USD tweaks; PR upstream when stable |
| `booster_train` fork | `github.com/thakk100/booster_train` | **fork** | K1 Isaac Lab task lives here first; PR upstream |
| `booster_robotics_sdk` | upstream only | **submodule** (read-only) | Never modify; pin to release tag |

### Submodule Map (inside `booster_ws/`)

```
.gitmodules
  sdk/booster_robotics_sdk      → github.com/BoosterRobotics/booster_robotics_sdk  @ pinned tag
  src/k1_description/assets     → github.com/thakk100/booster_assets                @ main
  isaac_tasks/booster_train_ref → github.com/thakk100/booster_train                 @ main
```

`src/k1_description/` is an owned package; it imports assets from the submodule at `assets/`.  
`isaac_tasks/k1_velocity/` is owned code inside this monorepo, NOT a submodule.

### Branching

```
main          — always passing colcon build + train smoke test
dev/phase-N   — active phase work (N = 0..5)
feat/*        — individual features / tasks
```

Tag each phase milestone: `v0.1-scaffold`, `v0.2-gazebo`, `v0.3-isaac`, `v0.4-training`, `v0.5-deploy`, `v0.6-fleet`

### What Gets Committed

| Artifact | Committed? | Notes |
|----------|-----------|-------|
| ROS2 package source | ✅ yes | All `src/k1_*/` |
| Isaac Lab task source | ✅ yes | `isaac_tasks/k1_velocity/source/` |
| Policy checkpoint (`.pt`) | ✅ small | `models/` via Git LFS if >100 MB |
| Training logs | ❌ no | `.gitignore`: `logs/`, `runs/` |
| Gazebo world SDFs | ✅ yes | `src/k1_sim_gazebo/worlds/` |
| Isaac USD scenes | ✅ yes | `src/k1_sim_isaac/assets/` (small; LFS if large) |
| `build/`, `install/`, `.venv/` | ❌ no | `.gitignore` |
| SDK binaries (`sdk/lib/`) | ❌ no | Tracked via submodule tag only |

### Git LFS

Enable for large binaries:
```bash
git lfs track "*.pt" "*.onnx" "*.usd" "*.usda" "*.stl" "*.dae"
```

### CI Plan (GitHub Actions)

```yaml
# .github/workflows/build.yml
on: [push, pull_request]
jobs:
  colcon-build:
    runs-on: ubuntu-22.04
    steps:
      - rosdep install + colcon build --packages-select k1_interfaces k1_description k1_locomotion
      - colcon test
  train-smoke:
    runs-on: self-hosted  # machine with GPU + IsaacLab
    steps:
      - python train_k1_velocity.py --max_iterations 5 --headless
```

### Initial Setup Commands

```bash
# 1. Init this workspace as git repo
cd ~/Projects/booster_ws
git init && git add . && git commit -m "chore: initial scaffold"

# 2. Create GitHub repo (thakk100/booster_ws) then:
git remote add origin git@github.com:thakk100/booster_ws.git
git push -u origin main

# 3. Fork upstream repos on GitHub UI, then add submodules:
git submodule add git@github.com:BoosterRobotics/booster_robotics_sdk.git sdk/booster_robotics_sdk
git submodule add git@github.com:thakk100/booster_assets.git src/k1_description/assets
git submodule add git@github.com:thakk100/booster_train.git isaac_tasks/booster_train_ref

# 4. Pin SDK to latest release
cd sdk/booster_robotics_sdk && git checkout <latest-tag> && cd ../..
git add .gitmodules sdk/ && git commit -m "chore: add submodules, pin sdk"

# 5. Enable LFS
git lfs install
git lfs track "*.pt" "*.onnx" "*.usd" "*.usda" "*.stl" "*.dae"
git add .gitattributes && git commit -m "chore: git lfs for large assets"
```

---

## References

- [Booster Lab paper (arxiv 2606.27813)](https://arxiv.org/html/2606.27813v1) — AMP-based loco pipeline
- [GR00T Decoupled WBC](https://nvlabs.github.io/GR00T-WholeBodyControl/references/decoupled_wbc.html) — WBC architecture
- [booster_train](https://github.com/BoosterRobotics/booster_train) — Isaac Lab training
- [booster_robotics_sdk](https://github.com/BoosterRobotics/booster_robotics_sdk) — SDK
- [booster_assets](https://github.com/BoosterRobotics/booster_assets) — K1 URDF/USD
- [Booster K1 specs](https://www.booster.tech/booster-k1/) — hardware
- [G1 rough_env_cfg](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py) — training template
