# Booster K1 ROS2 Workspace

Multi-robot simulation, locomotion training, and perception for the Booster
Robotics **K1** humanoid (22 DoF) — ROS 2 Jazzy + Isaac Sim 6.0.1 / Isaac Lab
3.0 + Gazebo Harmonic + MuJoCo.

![RoboCup 3v3 side view](docs/images/robocup_field_side.png)

## Status snapshot

| Area | State |
|---|---|
| Velocity locomotion policy | **Trained** (5000 iters, blind/contact-free) — `models/k1_velocity_policy.pt` (TorchScript) |
| Fleet sim, namespaced endpoints | **Verified on 3 backends**: Gazebo Harmonic, MuJoCo, Isaac Sim |
| Stereo head (ZED 2i replica) | Working in Isaac Sim: ±60 mm baseline, HD720, HFOV 101°, SGBM disparity |
| RoboCup 3v3 field demo | MuJoCo: 6 K1s + goals + ball + scripted kick, image capture |
| Multi-robot Isaac fleet | `fleet_sim.py` — bundled-jazzy rclpy, cross-stack DDS verified |

## Repo layout

```
src/
  k1_interfaces     msgs: JointCommand, LocomotionCommand, RobotStatus, PolicyObs
  k1_description    K1 URDF/xacro + gz_ros2_control blocks + booster_assets submodule
  k1_control        sim_bridge (JointCommand -> controllers), sdk_bridge stub
  k1_sim_gazebo     Gazebo fleet launch, MuJoCo fleet + RoboCup demo scripts
  k1_sim_isaac      Isaac fleet + ZED stereo capture scripts
  k1_locomotion     policy deployment node (Phase 4)
isaac_tasks/
  k1_velocity       RSL-RL task suite: velocity/basic/partial/kick families (table below)
  k1_kicking        kicking task cfg (scaffolding, unregistered)
  k1_head_tracking  head-tracking task cfg (scaffolding, unregistered)
  booster_train_ref Booster upstream training reference (submodule)
sdk/                booster_robotics_sdk (read-only submodule)
models/             trained policies (Git LFS)
docs/               research notes + captured imagery
```

## RL task families

Every RL task, its folder, and its trained-policy status. Per-task READMEs have
the full observation / action / reward tables, checkpoints, and play videos.

| Family | Folder | Gym ids (teacher → deployable) | Obs (deployable) | Action | Status |
|---|---|---|---|---|---|
| **P1** basic stand/balance | [`isaac_tasks/k1_velocity/`](isaac_tasks/k1_velocity/README.md#p1-basic-standbalance) | `Isaac-Basic-Teacher-K1-v0` → `Isaac-Basic-Student-K1-v0` | 42 blind | 12 legs | ✅ trained (teacher `model_6498`, student `model_1499`) |
| **P1f** P1 + force/torque shoves | same | `Isaac-Basic-Teacher-K1-F-v0` → `Isaac-Basic-Student-K1-F-v0` | 42 blind (teacher also sees the 6-dim shove wrench) | 12 legs | 🚀 launched 2026-09-24 (spark04) |
| **P2** velocity on rough terrain | [`isaac_tasks/k1_velocity/`](isaac_tasks/k1_velocity/README.md#p2-velocity-rough-terrain) | `Isaac-Velocity-Rough-K1-Teacher-v0` → `Isaac-Velocity-Distill-K1-Play-v0` | 48 blind | 12 legs | ✅ trained (`model_2999` both) |
| **P2f** P2 + force/torque shoves | same | `Isaac-Velocity-Rough-K1-Teacher-F-v0` → `Isaac-Velocity-Distill-K1-F-v0` | 48 blind (teacher 241-dim) | 12 legs | 🚀 launched 2026-09-24 (spark04) |
| **Partial control** legs+head, randomized arms | [`isaac_tasks/k1_velocity/`](isaac_tasks/k1_velocity/README.md#partial-control-leghead-randomized-arms) | `Isaac-Velocity-PartialCtrl-K1-v0` → `Isaac-Velocity-PartialCtrl-K1-Play-v0` | 68 blind | 14 (12 legs + 2 head) | ⏳ Run-10 training on spark02 |
| **Kick ball** locomotion + ball obs | [`isaac_tasks/k1_velocity/`](isaac_tasks/k1_velocity/README.md#kick-ball) | `Isaac-Kick-Ball-K1-Teacher-v0` → `Isaac-Kick-Ball-K1-Distill-v0` | 45 blind (teacher 51 w/ ball pose) | 12 legs | ⛔ gated on obs-53 goal-env upgrade |
| Kicking (ball-relative legacy) | [`isaac_tasks/k1_kicking/`](isaac_tasks/k1_kicking/README.md) | *not registered* | 47–48 | 12 legs | 📝 cfg only, no checkpoints |
| Head tracking | [`isaac_tasks/k1_head_tracking/`](isaac_tasks/k1_head_tracking/README.md) | *not registered* | 11 | 2 head | 📝 cfg only, no checkpoints |
| Upstream reference library | [`isaac_tasks/booster_train_ref/`](isaac_tasks/booster_train_ref/README.md) | beyond_mimic demo tasks | — | — | 📚 library, not one of our tasks |

**Teacher vs deployable:** every family trains a privileged *teacher* (PPO, sees
the 187-point height scan — plus ball state / foot slip / shove wrench where
applicable) and distills a blind *student* that only sees proprioception, so the
exported policy deploys on the real K1 with no terrain sensing.
`tests/test_deployability.py` statically enforces that the deployable group is
GT-free (teacher groups exempt).

## Policy recordings

Headless play videos with a live input/output HUD — velocity commands, every
observation group as value bars, the action vector, step + episode reward —
recorded by `scripts/record_policies_host.sh`
(`isaac_tasks/k1_velocity/scripts/play_record.py`, 750 steps @ 50 fps, 1024×576).
Full per-step traces sit beside each video as `*_trace.npz` (git-ignored);
TorchScript exports go to [`models/`](models/) (Git LFS).

| P1 teacher — stand (rough ground, active shoves) | P1 student — stand (blind 420-dim distill) |
|---|---|
| ![P1 teacher](isaac_tasks/k1_velocity/videos/p1_teacher_stand.mp4) | ![P1 student](isaac_tasks/k1_velocity/videos/p1_student_stand.mp4) |

| P2 teacher — walk (rough, circle command) | P2 student — walk (distill, circle command) |
|---|---|
| ![P2 teacher](isaac_tasks/k1_velocity/videos/p2_teacher_rough.mp4) | ![P2 student](isaac_tasks/k1_velocity/videos/p2_student_walk.mp4) |

Partial-control walk video: [`videos/partial_walk.mp4`](isaac_tasks/k1_velocity/videos/partial_walk.mp4)
(recorded once Run-10 finishes).

## Quick start

### Gazebo Harmonic fleet (CPU)
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch k1_sim_gazebo sim_gazebo_fleet.launch.py n_robots:=2 gui:=false
# endpoints per robot:
#   /k1_{i}/joint_states   /k1_{i}/joint_commands   /k1_{i}/forward_position_controller/commands
```

### MuJoCo fleet / RoboCup demo (CPU)
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
python3 src/k1_sim_gazebo/scripts/mujoco_fleet_node.py --n_robots 3
python3 src/k1_sim_gazebo/scripts/mujoco_robocup_demo.py --capture   # saves docs/images/
```

### Isaac Sim fleet + stereo (GPU)
```bash
/home/thakk100/Projects/IsaacLab/.venv-isaac/bin/python3.12 \
  src/k1_sim_isaac/scripts/fleet_sim.py --n_robots 2 --headless
/home/thakk100/Projects/IsaacLab/.venv-isaac/bin/python3.12 \
  src/k1_sim_isaac/scripts/stereo_cam_test.py --steps 40
```
Isaac fleet uses the **isaacsim-bundled jazzy rclpy** (env re-exec) — never
source `/opt/ros` into that process; system nodes talk to it over DDS
(`ROS_DOMAIN_ID=77`).

## Training

```bash
# train (tmux)
./scripts/start_training.sh
# play + export TorchScript
python isaac_tasks/k1_velocity/scripts/play.py \
  --checkpoint logs/rsl_rl/k1_velocity_rough/<run>/model_4999.pt \
  --num_envs 20 --steps 500 --headless --export models/k1_velocity_policy.pt
```

Result (run `2026-08-24_12-46-29`): mean reward **+12.5**, episode length
**826/1000**, 71% of episodes reach timeout. Blind proprioceptive policy
(contact sensors blocked upstream — see `docs/robocup_gmr_research.md`).

## Stereo head (ZED 2i replica)

K1 ships a Stereolabs **ZED 2i** (120 mm baseline). `stereo_cam_test.py`
mounts two pinhole cameras on `Head_2` (±60 mm, HD720, HFOV 101° = 2.1 mm
lens per Stereolabs' rectified-FOV table) and captures:

| Left / Right | SGBM disparity |
|---|---|
| ![pair](docs/images/k1_stereo_pair.png) | ![disparity](docs/images/k1_stereo_disparity.png) |

## RoboCup 3v3 (MuJoCo)

| Top | Kick close |
|---|---|
| ![top](docs/images/robocup_field_top.png) | ![kick](docs/images/robocup_kick_close.png) |

`mujoco_robocup_demo.py` assembles the pitch (markings, goals, ball) and six
K1s via `MjSpec` URDF attach with per-robot name prefixes. Each robot exposes
the standard fleet endpoints; `--capture` runs a scripted striker kick and
saves snapshots. See `docs/robocup_gmr_research.md` for the Booster
`robocup_demo` contract analysis and 3v3 replication plan.

## Docs

- `HANDOFF.md` — environment setup, current state, next steps
- `docs/robocup_gmr_research.md` — RoboCup sim research + GMR (video→motion)
  pipeline integration plan
- `TASKS.md` — phase checklist
