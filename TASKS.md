# TASKS — Booster K1 ROS2 Workspace

Status: `[ ]` todo · `[~]` in-progress · `[x]` done · `[!]` blocked

---

## PHASE 0 — Scaffold, Config & Git

- [x] Write PLAN.md
- [x] Write TASKS.md

### Git Setup (do first, everything else commits on top)
- [ ] **T0.G1** Init repo + first commit
  ```bash
  cd ~/Projects/booster_ws
  git init && git add PLAN.md TASKS.md && git commit -m "chore: initial plan and tasks"
  ```
- [ ] **T0.G2** Create GitHub repo `thakk100/booster_ws`, push
  ```bash
  git remote add origin git@github.com:thakk100/booster_ws.git
  git push -u origin main
  ```
- [ ] **T0.G3** Fork on GitHub UI:
  - Fork `BoosterRobotics/booster_assets` → `thakk100/booster_assets`
  - Fork `BoosterRobotics/booster_train` → `thakk100/booster_train`
- [ ] **T0.G4** Add submodules
  ```bash
  git submodule add git@github.com:BoosterRobotics/booster_robotics_sdk.git sdk/booster_robotics_sdk
  git submodule add git@github.com:thakk100/booster_assets.git src/k1_description/assets
  git submodule add git@github.com:thakk100/booster_train.git isaac_tasks/booster_train_ref
  ```
- [ ] **T0.G5** Pin SDK submodule to latest release tag; commit `.gitmodules`
- [ ] **T0.G6** Enable Git LFS
  ```bash
  git lfs install
  git lfs track "*.pt" "*.onnx" "*.usd" "*.usda" "*.stl" "*.dae" "*.bin"
  git add .gitattributes && git commit -m "chore: git lfs for large assets"
  ```
- [ ] **T0.G7** Write `.gitignore`: `build/`, `install/`, `log/`, `.venv/`, `logs/`, `runs/`, `__pycache__/`, `*.egg-info/`
- [ ] **T0.G8** Create branch `dev/phase-0`, work there; merge to main at phase milestone

### Workspace Scaffold
- [ ] **T0.1** Write CLAUDE.md (workspace coding rules, ROS env split, naming)
- [ ] **T0.2** Create `src/` tree with empty package stubs (package.xml + CMakeLists / setup.py)
  - `k1_description`, `k1_interfaces`, `k1_control`, `k1_locomotion`, `k1_wbc`, `k1_sim_gazebo`, `k1_sim_isaac`, `k1_bringup`
  - Commit stub: `git commit -m "feat: add ROS2 package stubs"`
- [ ] **T0.3** `booster_assets` submodule active — copy K1 URDF from `src/k1_description/assets/robots/K1/` into package; do NOT edit assets/ directly (PR upstream fork instead)
- [ ] **T0.4** `sdk/booster_robotics_sdk` submodule active; build C++ SDK locally
- [ ] **T0.5** `isaac_tasks/booster_train_ref` submodule active — reference only; K1 task goes in `isaac_tasks/k1_velocity/` (owned code)
- [ ] **T0.6** Create `isaac_tasks/k1_velocity/` owned package (NOT a submodule), commit

### CI
- [ ] **T0.C1** `.github/workflows/build.yml` — colcon build + test on push (Ubuntu 22.04, ROS Humble)
- [ ] **T0.C2** `.github/workflows/train_smoke.yml` — 5-iteration training smoke test (self-hosted GPU runner)

---

## PHASE 1 — Gazebo Sim (1 robot)

### T1.1 — k1_description
- [ ] Convert/fix URDF for ROS2 + gz_ros2_control tags
- [ ] Add `ros2_control` hardware interface block (`GazeboSystem`)
- [ ] Verify URDF with `check_urdf` and `urdf_to_graphiz`
- [ ] Add `robot_ns` xacro arg → all link/joint names prefixed `{robot_ns}/`
- [ ] `ros2 launch k1_description view_robot.launch.py` — RViz preview

### T1.2 — k1_sim_gazebo
- [ ] Create flat world SDF (`worlds/flat.world`)
- [ ] Create `spawn_robot.launch.py`: spawn URDF with `robot_ns` arg
- [ ] `gz_ros2_control` YAML: position controllers for 22 joints
- [ ] Joint state broadcaster + effort/position controllers

### T1.3 — k1_interfaces
- [ ] Define `JointCommand.msg`: `string[] joint_names`, `float64[] positions`, `float64[] velocities`, `float64[] efforts`
- [ ] Define `LocomotionCommand.msg`: Twist + gait mode enum
- [ ] Define `RobotStatus.msg`: battery, contact states, fault flags
- [ ] Build and verify msgs

### T1.4 — k1_bringup (Gazebo)
- [ ] `sim_gazebo.launch.py`: arg `robot_ns:=k1_0`, launches Gazebo + spawn + controllers
- [ ] `teleop_twist_keyboard` remapped to `/{robot_ns}/cmd_vel`
- [ ] **Verify:** `ros2 topic list` shows `/{robot_ns}/joint_states`, `/{robot_ns}/cmd_vel`

---

## PHASE 2 — Isaac Sim (1 robot, IsaacSim 6.0.1)

### T2.1 — k1_sim_isaac
- [ ] Create Isaac Sim USD stage with K1 asset (USD from booster_assets or convert URDF→USD)
- [ ] OmniGraph: publish `/{robot_ns}/joint_states` (JointState), subscribe `/{robot_ns}/joint_commands`
- [ ] ROS2 bridge config using `~/Projects/IsaacLab/isaac6/` ROS2
- [ ] `isaac_sim.launch.py`: sets Isaac env, launches standalone Python script

### T2.2 — Isaac Sim env split script
- [ ] `scripts/run_isaac_ros.sh`: sources isaac6 env, runs Isaac Sim Python
- [ ] Document: system ROS2 shell ↔ Isaac ROS2 shell — DO NOT mix

### T2.3 — k1_bringup (Isaac)
- [ ] `sim_isaac.launch.py`: arg `robot_ns:=k1_0`, bridges Isaac ↔ system ROS2
- [ ] **Verify:** `ros2 topic echo /{robot_ns}/joint_states` visible from system shell

---

## PHASE 3 — RSL-RL Velocity Training Task

### T3.1 — Isaac Lab task scaffold
- [ ] Create `isaac_tasks/k1_velocity/` as Isaac Lab extension
  - `setup.cfg`, `pyproject.toml`, `__init__.py`
  - Register task: `Isaac-Velocity-Rough-K1-v0`
- [ ] `source/k1_velocity/envs/__init__.py`
- [ ] `source/k1_velocity/envs/velocity_env_cfg.py`: `K1VelocityRoughEnvCfg`
  - Port G1 rough_env_cfg, replace `G1_MINIMAL_CFG` with K1 asset
  - K1 joint limits: Hip P=-171~126°, R=-22~89°, Y=±59°, Knee=0~127°, Ankle P=-50~20°, R=±20°
  - 12 leg joints in action space
  - 4096 envs, episode 20s, decimation 4
- [ ] `source/k1_velocity/envs/velocity_play_cfg.py`: eval config (50 envs, no noise)

### T3.2 — K1 articulation config
- [ ] `source/k1_velocity/robots/k1.py`: `K1ArticulationCfg`
  - USD path: from booster_assets (or converted URDF)
  - Actuator model: DCMotorCfg with K1 torque/speed specs
  - Joint names matching booster_assets exactly

### T3.3 — RSL-RL agent config
- [ ] `source/k1_velocity/agents/rsl_rl_ppo_cfg.py`: PPO hyperparams
  - num_steps_per_env: 24, num_mini_batches: 4
  - learning_rate: 1e-3, entropy_coef: 0.01
  - 5000 iterations
- [ ] `scripts/train_k1_velocity.py` (or symlink to booster_train train.py)
- [ ] `scripts/play_k1_velocity.py`

### T3.4 — Train + verify
- [ ] Train for 100 iterations, confirm reward increases
- [ ] Play with `play.py`, verify robot walks in IsaacLab viewer
- [ ] Export TorchScript: `policy.pt`

---

## PHASE 4 — Policy Deployment Node

### T4.1 — k1_locomotion
- [ ] `locomotion_node.py`: ROS2 node
  - Load `policy.pt` (TorchScript)
  - Sub: `/{robot_ns}/cmd_vel` (Twist, 50 Hz)
  - Sub: `/{robot_ns}/joint_states` (JointState)
  - Pub: `/{robot_ns}/joint_commands` (JointCommand)
  - Obs builder: match training obs exactly (normalize, history buffer)
  - 50 Hz timer, 10-step history
- [ ] `locomotion.launch.py`: `robot_ns`, `policy_path` args

### T4.2 — k1_control (SDK bridge)
- [x] `sim_bridge_node.py` (sim mode):
  - Sub `/{robot_ns}/joint_commands` → pub `/{robot_ns}/forward_position_controller/commands`
  - Verified in Gazebo fleet + MuJoCo fleet
- [ ] `sdk_bridge_node.cpp` (C++): real robot via booster_robotics_sdk

### T4.3 — End-to-end verify (Gazebo)
- [ ] Sim loop: policy_node → sim_bridge → Gazebo → joint_states → policy_node
- [ ] `ros2 topic pub /{robot_ns}/cmd_vel` → robot walks

---

## PHASE 5 — Multi-Robot (6x)

- [x] **T5.1** Parametric launch: `sim_gazebo_fleet.launch.py n_robots:=6`
  - Loop spawn with `robot_ns:=k1_{i}`, x_offset per robot (verified n=2 in Gazebo, n=3 in MuJoCo)
- [ ] **T5.2** Fleet config: `config/fleet/6robot.yaml` — per-robot IP, ns, domain_id
- [x] **T5.3** Multi-robot Gazebo world (shared flat.sdf, grid offsets)
- [x] **T5.4b** MuJoCo backend: `k1_sim_gazebo/scripts/mujoco_fleet_node.py` — N independent K1 worlds, SDK-style PD (`qfrc_applied`, IMPLICITFAST), same ROS endpoints as Gazebo
- [ ] (stretch) Multi-robot Isaac Sim scene

---

## BACKLOG / STRETCH

- [ ] AMP-based training (booster_train BeyondMimic) for natural gait
- [ ] Decoupled WBC: `k1_wbc` node with upper/lower body QP
- [ ] Teleoperation: Twist → locomotion + gamepad → arm
- [ ] SLAM + Nav2 integration (port from g1_perception_ws pattern)
- [ ] Real robot bring-up: `k1_bringup/real.launch.py`
- [ ] CI: colcon build + test on push

---

## Implementation Order

```
T0.G1→T0.G2→T0.G3→T0.G4→T0.G5→T0.G6→T0.G7→T0.G8   branch: dev/phase-0
T0.1→T0.2→T0.3→T0.4→T0.5→T0.6→T0.C1→T0.C2
  → merge main, tag v0.1-scaffold

T1.3→T1.1→T1.2→T1.4      branch: dev/phase-1
  → merge main, tag v0.2-gazebo

T2.1→T2.2→T2.3            branch: dev/phase-2
  → merge main, tag v0.3-isaac

T3.1→T3.2→T3.3→T3.4      branch: dev/phase-3
  → merge main, tag v0.4-training   (commit policy.pt via LFS)

T4.1→T4.2→T4.3            branch: dev/phase-4
  → merge main, tag v0.5-deploy

T5.1→T5.2→T5.3            branch: dev/phase-5
  → merge main, tag v0.6-fleet
```

### PR Workflow for Upstream Forks

When K1-specific changes to `booster_assets` or `booster_train` stabilize:
1. Push to `thakk100/booster_assets` (or `booster_train`) fork branch
2. Open PR to `BoosterRobotics/booster_assets` upstream
3. After merge, update submodule pin in this repo
