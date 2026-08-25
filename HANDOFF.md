# Booster K1 Workspace — Handoff

> **Date:** 2026-08-24 (evening)
> **Branch:** `dev/phase-0`
> **GitHub:** https://github.com/thdhyan/booster_ws

---

## Current State

### Training — DONE ✅
- Run `logs/rsl_rl/k1_velocity_rough/2026-08-24_12-46-29`, 5000 iters, 2h33m
- Final: mean reward **+12.5**, ep length **826/1000**, 71% episodes time out
- Policy exported: **`models/k1_velocity_policy.pt`** (TorchScript, LFS)
- Env is **contact-free** (blind): Isaac Sim 6 URDF importer nests link prims;
  `activate_contact_sensors` only reaches root (IsaacLab#5918) and ContactSensor
  can't address nested bodies (pre-PR#6378). Gait shaping dropped; terminations
  = height(0.35 m) + orientation(0.8 rad).
- WandB: `thakk100-dhyan-home/booster_k1_locomotion` (entity verified via API;
  'thakk100'/'thdhyan' are NOT valid entities)

### Fleet sim — 3 backends verified ✅
| Backend | Entry | Cmd topic type | Notes |
|---|---|---|---|
| Gazebo Harmonic | `ros2 launch k1_sim_gazebo sim_gazebo_fleet.launch.py n_robots:=N` | `k1_interfaces/JointCommand` → sim_bridge → forward_position_controller | gz_ros2_control 1.2.19; per-robot CM under `/{ns}` via `<ros><namespace>` |
| MuJoCo | `python3 src/k1_sim_gazebo/scripts/mujoco_fleet_node.py --n_robots N` | `k1_interfaces/JointCommand` → PD `qfrc_applied` | IMPLICITFAST required (39 g ankle-cross diverges otherwise) |
| Isaac Sim | `python3 src/k1_sim_isaac/scripts/fleet_sim.py --n_robots N --headless` | **`sensor_msgs/JointState`** (bundled stack can't import ws msgs) | `ROS_DOMAIN_ID=77`; targets via `set_joint_position_target_index(int32)` + `write_data_to_sim()` |

All: `/{ns}/joint_states` @50 Hz + `/{ns}/joint_commands`. Isolation verified
(command robot A, robot B frozen) on every backend.

**Isaac env isolation**: fleet_sim re-execs with isaacsim-bundled jazzy rclpy
(`isaacsim/exts/isaacsim.ros2.core/jazzy`) prepended to PYTHONPATH/LD_LIBRARY_PATH.
NEVER source /opt/ros into that process — cross-stack interop is pure DDS.

**Gotchas learned**:
- K1 URDF has links sharing names with joints → generator renames LINKS
  (`*_link`) for SDF uniqueness (Gazebo only; joints stay canonical)
- URDF joint names: `AAHead_yaw`, `ALeft/ARight_Shoulder_Pitch` (A-prefix!)
- pkill patterns self-match if the same string appears elsewhere in your
  shell command — split kill/launch into separate calls
- colcon: force `-DPython3_EXECUTABLE=/usr/bin/python3` (stray py3.10 in PATH)

### Stereo head (ZED 2i replica) — working ✅
`src/k1_sim_isaac/scripts/stereo_cam_test.py` (needs `--enable_cameras`):
- Two cameras under `Robot_0/Geometry/Trunk/Head_1/Head_2/zed_{left,right}`
- ±60 mm baseline, 1280×720, HFOV 101° (ZED 2i 2.1 mm lens, HD720, per
  Stereolabs rectified-FOV table), ROS-optical rot `(0.5,-0.5,0.5,-0.5)`
- Captures left/right RGB + SGBM disparity + depth → `docs/images/k1_stereo_*.png`
- Mount pos (0.10, ±0.06, 0.0) — x=0.10 clears the head mesh

### RoboCup 3v3 (MuJoCo) — capture working ✅
`src/k1_sim_gazebo/scripts/mujoco_robocup_demo.py`:
- MjSpec assembly: pitch markings, goals, ball, 6 K1s (URDF attach, prefixed)
- `--capture`: scripted striker kick + saves 3 shots to docs/images (ROS-free)
- without `--capture`: full ROS endpoints (source ws first)
- Stability: timestep 1e-3, IMPLICITFAST, feet-only contacts
  (feet c=2/a=1, other robot geoms 0/0), arm kd ≤0.02 (tiny inertias),
  qvel>50 → reset guard

### Research — see `docs/robocup_gmr_research.md`
- **robocup_demo** (Booster): robot-side brain; sim contract =
  `/camera/{r}_rgbd_camera/*` in, `LocoApiTopic{r}Req` (vendored
  booster_msgs/RpcReqMsg) out. Replication plan staged in doc.
- **3v3 sims**: RCSSServerMJ_GH (MuJoCo, supports Booster T1!) closest base
- **GMR** (General Motion Retargeting): Booster K1 supported natively.
  User's pipeline lives at `~/Projects/Thesis/workin_ws`
  (gmr_ros, gvhmr_ros, park_interfaces; GMR assets include
  `booster_k1/K1_serial.xml` + `smplx_to_k1.json`).
  Integration: set `robot_type: booster_k1` → bridge `/gmr_results` →
  `/{ns}/joint_commands`; batch pkl→CSV → BeyondMimic tracking tasks.

---

## Environment Setup

### Python / IsaacLab (training + Isaac scripts)
```bash
source /home/thakk100/Projects/IsaacLab/.venv-isaac/bin/activate
# IsaacSim 6.0.1 pip — Python 3.12, IsaacLab 3.0.0b2, rsl_rl_lib 5.0.1
```

### System ROS2 (all k1_* packages, Gazebo, MuJoCo nodes)
```bash
source /opt/ros/jazzy/setup.bash
cd ~/Projects/booster_ws && source install/setup.bash
```
**NEVER source both in the same shell** (RMW/env double-init).

Build (note the Python pin):
```bash
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

---

## Key files

| File | Purpose |
|---|---|
| `isaac_tasks/k1_velocity/.../velocity_env_cfg.py` | env (contact-free, URDF-exact joint names) |
| `isaac_tasks/k1_velocity/.../agents/rsl_rl_ppo_cfg.py` | PPO (rsl-rl≥4 model schema; entity thakk100-dhyan-home) |
| `isaac_tasks/k1_velocity/scripts/train.py` | strips legacy noise kwargs (rsl_rl 5.x) |
| `isaac_tasks/k1_velocity/scripts/play.py` | eval + JIT/ONNX export |
| `scripts/probe_k1_bodies.py` | dumps imported articulation body/joint names |
| `src/k1_description/scripts/generate_k1_urdf.py` | per-ns URDF (mesh file:// + link renames) |
| `src/k1_sim_gazebo/launch/sim_gazebo_fleet.launch.py` | N-robot Gazebo fleet |
| `src/k1_sim_gazebo/scripts/mujoco_fleet_node.py` | MuJoCo fleet |
| `src/k1_sim_gazebo/scripts/mujoco_robocup_demo.py` | RoboCup field + capture |
| `src/k1_sim_isaac/scripts/fleet_sim.py` | Isaac fleet (bundled rclpy) |
| `src/k1_sim_isaac/scripts/stereo_cam_test.py` | ZED 2i stereo capture |
| `models/k1_velocity_policy.pt` | trained policy (LFS) |

---

## Next Steps

| Task | Priority |
|---|---|
| Phase 4: `k1_locomotion` policy node — load `models/k1_velocity_policy.pt`, sub `/{ns}/cmd_vel` (Twist) + `joint_states`, pub `joint_commands`; obs builder must match training exactly (72-dim, 10-step history) | High |
| Improve policy: longer train / re-add contact rewards when upstream fixes land / BeyondMimic+GMR motions | High |
| GMR bridge node: `/gmr_results` (park_interfaces) → `/{ns}/joint_commands`; set workin_ws gmr_config `robot_type: booster_k1` | Medium |
| loco_api_bridge for Booster robocup_demo (vendored booster_msgs → our endpoints) | Medium |
| Download `robocup_3Dsim_field` (models.gazebosim.org) → USD for Isaac field | Low |
| Isaac RoboCup: extend fleet_sim with field+ball+cameras | Low |

---

## WandB / logs

- WandB project: `booster_k1_locomotion`, entity `thakk100-dhyan-home`
- Training log: `logs/train_k1_velocity.log`
- Checkpoints: `logs/rsl_rl/k1_velocity_rough/2026-08-24_12-46-29/model_*.pt`

---

## Git workflow

```bash
# current branch: dev/phase-0 (all work so far)
# suggested: merge to main, tag v0.2-fleet-sim at next milestone
git checkout main && git merge dev/phase-0
git tag v0.2-fleet-sim && git push origin main --tags
```
