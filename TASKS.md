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

## PHASE 6 — Soccer HRL (single robot, ball → goal) · plan: `PLAN_PHASE6_SOCCER_HRL.md`

> Rewards **LOCKED** (user-approved in chat 2026-09-22). Stack: Isaac Sim 6.1.0 + IsaacLab v3.0.0-EA.
> Deployability invariant: deployable obs groups never read GT ball/goal state (GT only in rewards + teacher groups).
> Status legend as above: `[ ]` todo · `[~]` in-progress · `[x]` done · `[!]` blocked

### T6.0 — Phase-0 closure (git)
- [x] **T6.0.1** Commit leftovers on `dev/phase-0` (head/kick drafts, `k1_soccer_compose.py`, `PLAN_MULTILAYER.md`, `ROBOTS.md`, `isaac_fleet_vis.py`, `docker_isaac_fleet.sh`, `STATE.md`, `PLAN_PHASE6_SOCCER_HRL.md`) — `6ae1d94`
- [x] **T6.0.2** `main ← dev/phase-0` merge, push, tag `v0.7-phase0-complete`, push tags
- [x] **T6.0.3** Create branch `dev/phase-6-soccer-hrl`; all Phase-6 work here
- [x] **T6.0.4** *(added)* **LFS repair**: 62 `.py` files restored from pointer corruption (`002a647` had `*.py filter=lfs`), + truncated `ball_to_goal_progress` RewTerm fixed — `88f73fc`, pushed to main

### T6.1 — Environment rebuild (blocking)
- [x] **T6.1.1** Install Isaac Sim 6.1.0 pip → **`~/Projects/IsaacLab-ea/.venv`** (NOT `isaac6/.venv` — that one is owned by a parallel G1_sim session recreating it in loops); disk guard live: **16 GB free** after install + 3.7 GB cache prune
- [x] **T6.1.2** IsaacLab **v3.0.0-EA** worktree `~/Projects/IsaacLab-ea` (shared checkout stays on `perf-2026-07-06`); `uv sync` = editable workspace members into the venv; our task packages on PYTHONPATH via `scripts/phase6_env.sh` (EA dropped the entry-point group; root-owned egg-info removed)
- [x] **T6.1.3** `rsl_rl` (lockfile 5.4.1), `ultralytics` 8.4.158, `imageio[ffmpeg]` — installed; reproduce with `scripts/install_phase6_env.sh`
- [x] **T6.1.4** Rewrite `scripts/start_training.sh` (stale venv path, wrong wandb entity → `thakk100-dhyan-home`)
- [~] **T6.1.5** **Gate G0**: headless Kit 110.3 app launch ✅ + 9 K1 task registrations ✅; velocity 16-env env smoke deferred to **G1** (needs T6.2 migration) → **G1 PASSED 2026-09-22** (`16a6803`)

### T6.2 — Migrate existing tasks to IsaacLab 3.0 API
- [x] **T6.2.1** Migration audit per PLAN §2 table (quats WXYZ→XYZW, `ProxyArray.torch`, `write_*_to_sim_index/mask`, `isaaclab train` CLI, `VideoRecorderCfg`, `enable_extension`, actuator renames, contact contracts) — done; deferred by design: `VideoRecorderCfg` → T6.3.6, play scripts → T6.5
- [x] **T6.2.2** Velocity task migration + **Gate G1**: contact rewards fire (`feet_air_time` > 0); re-validate `flatten_k1_usd.py` on 6.1 importer (drop flatten if IL 3.0 native contact fix suffices) — **PASSED 2026-09-22**: 16 envs × 2 iters rc=0 @ `16a6803`; wandb run `x1ptwkae`; `model_0/1.pt` saved; `feet_air_time`=3e-05 (>0), `feet_slide`=-2.2e-04 → contact fires directly from `K1_22dof.urdf` on the 6.1 importer (**flatten script not on the training path** — kept for reference)
- [~] **T6.2.3** Kick task migration (OmniReset reset paths use removed `write_root_state_to_sim`) — done: cfg imports+instantiates ✅; mdp restored to authentic `89b13a9` (244-line — earlier LFS restore had landed a stale 210-line object missing `reset_ball_omnireset`) + `write_root_pose/velocity_to_sim_index` + `ProxyArray.torch`; **remaining:** obs-fn ProxyArray/quat audit + kick smoke
- [x] **T6.2.4** Patch `thakk100/booster_train` fork (`BOOSTER_K1_CFG`) for 3.0 actuator/quat API if needed — done: fork `2879b1a` (`_parse_joint_parameter` → `resolve_joint_parameter`); deprecated `effort_limit_sim`/`velocity_limit_sim` aliases still accepted with warning

### T6.3 — Shared MDP infrastructure
- [ ] **T6.3.1** Domain randomization: ball position (OmniReset extended ±1.5 m cone, rolling resets 0–1.5 m/s), **ball color palette** (white/orange/hivis/black-panel/red), ball physics (mass/restitution/friction/radius), lighting, camera noise — PLAN §3.1
- [ ] **T6.3.2** `random_body_push` EventTerm: 20–80 N, 0.05–0.15 s, interval 3–8 s, random body ∈ {Trunk/waist, pelvis, chest, upper legs}, ~60 % envs — PLAN §3.2 (P1/P2 required, P4/P5 enabled)
- [ ] **T6.3.3** Head/scene camera 320×240 (camera subsets) + **YOLO wrapper** (`yolov8n` @10 Hz, `sports ball`, YOLO vector = visible/du/dv), geometric FOV proxy for non-camera envs, `ball_detect_rate` wandb metric
- [ ] **T6.3.4** **Vision estimator module** (PLAN §3.4): ball = YOLO bbox + depth lookup → pos/vel est + flags; goal = rectangular-goal tracker/rememberer (noisy detection snap / odometry dead-reckon / memorized prior); estimator-model backend (noise σ, latency 50–150 ms, dropout 0–400 ms randomized per episode)
- [ ] **T6.3.5** **Static deployability test**: assert every `policy`/student obs group references estimator outputs only — no GT ball/goal terms (grep/unit test in CI)
- [ ] **T6.3.6** Video recorder: every **200 iterations → 30 s clip** → `logs/videos/<run>/iter_XXXX.mp4` + `wandb.Video`; 4 video envs/run; `VideoRecorderCfg` or runner callback (PLAN §3.5)
- [ ] **T6.3.7** `scripts/train_guard.sh` watchdog (GPU > 7 GB / RAM < 1 GiB / disk < 5 GB → kill) + `systemd-run --scope MemoryMax=9G` wrapper (PLAN §3.7)
- [ ] **T6.3.8** wandb project `booster_k1_soccer_hrl` created; run naming `p{1..5}_*` verified in smoke

### T6.4 — Base policies (sequential; each: smoke → train → gate)
- [ ] **T6.4.1** **P1 BASIC teacher** `Isaac-Basic-Teacher-K1-v0`: rough terrain + height scan + foot contacts + `random_body_push`; rewards LOCKED table (9 terms); 256 envs / 2000 it → `p1_basic_teacher`
- [ ] **T6.4.2** **P1 BASIC student** (blind, 42×K10=420 MLP 512-256-128) distill → `p1_basic_student`; **Gate G3**
- [ ] **T6.4.3** **P2 MOVE teacher** `Isaac-Move-Teacher-K1-v0`: rough curriculum + pushes + vel cmds; rewards LOCKED (11 terms); 512 envs / 3000 it → `p2_move_teacher`
- [ ] **T6.4.4** **P2 MOVE student** (48×K10=480) distill → `p2_move_student`; **Gate G3**
- [ ] **T6.4.5** **P3 HEAD TRACK** `Isaac-HeadTrack-K1-v0`: deployable 12-dim YOLO obs, 2-dim head actions; rewards LOCKED (6 terms, GT angle = reward-only); DR colors/lighting; 512 envs (64 YOLO) / 2000 it → `p3_head_track`; **Gate G2** (`ball_detect_rate` > 0.9)
- [ ] **T6.4.6** **P4 CHASE&KICK teacher** `Isaac-Kick-Teacher-K1-v0`: GT ball/goal obs (53), head driven by frozen P3, rewards LOCKED (12 terms); 512 envs / 3000 it → `p4_kick_teacher`
- [ ] **T6.4.7** **P4 CHASE&KICK student** `Isaac-Kick-Student-K1-v0`: estimator obs 54×K10=540, distill with estimator in the loop; 512 envs (64 YOLO) / 3000 it → `p4_kick_student`; **Gates G2.5 + G3**
- [ ] **T6.4.8** **Auto-stability eval** (Gate G4): scripted 60 N waist shove → P1 no-fall ≥ 90 %, P2 recovers; save video

### T6.5 — P5 HRL top-level
- [ ] **T6.5.1** **P5 teacher** `Isaac-HRL-Teacher-K1-v0`: 5 Hz manager, 7-dim out (4 skill logits + vx,vy,wz), 23-dim privileged obs, frozen P1–P4 options, rewards LOCKED (7 terms); 256 envs / 2000 it → `p5_hrl_teacher`
- [ ] **T6.5.2** **P5 student** `Isaac-HRL-Student-K1-v0`: 20×K10=200 estimator obs (YOLO+ball_est+goal_est+flags), distill → `p5_hrl_student`; 256 envs / 1500 it
- [ ] **T6.5.3** **Gate G5**: P5 student eval (GT disconnected) — goal-scoring rate, skill-switch behavior, videos reviewed by user

### T6.6 — Integration, export, handoff
- [ ] **T6.6.1** Export TorchScript: `models/k1_{basic,move,head_track,chase_kick,hrl}_policy[_student].pt` (LFS)
- [ ] **T6.6.2** Extend `k1_soccer_compose.py` → P3 (head/YOLO) + P4 legs + P5 manager with **estimator inputs only**; end-to-end fleet-sim run (**Gate G6**, video)
- [ ] **T6.6.3** YOLO fine-tune on auto-labeled synthetic frames (ball palette + rectangular goal posts) if `ball_detect_rate` < 0.9
- [ ] **T6.6.4** Update `STATE.md` + `HANDOFF.md` with final results; PR `dev/phase-6-soccer-hrl` → `main`, tag `v0.8-soccer-hrl`

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
