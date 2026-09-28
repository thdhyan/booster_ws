# Quest 3 VR teleop of the Booster K1 (Isaac Teleop)

Branch `feat/vr-teleop-k1` (worktree `~/Projects/wt-vr-teleop`), forked from
`feat/video-to-motion` at `f1bd55a`. Port of the working G1 stack
(`~/Projects/thesis/G1_sim/teleop/`, `docs/KT_VR_TELEOP.md`; read its traps
section before touching the headset side).

Phase 1 scope (2026-09-28): **no walking**. The K1 hangs in the air on a fixed
base; the thumbsticks move the suspended trunk. Walking (P2 locomotion policy)
replaces the gantry later and keeps the same `/{ns}/cmd_vel` contract.

## 1. What runs

```text
Quest 3 browser (nvidia.github.io/IsaacTeleop/client) ──CloudXR 48322/49100/47998──► host
k1_teleop/xr_teleop_node.py  (Televiz owns OpenXR; TeleopSession borrows it)
  panels, fixed in the room (X re-centres):
    front  ZED 2i pair as ONE stereo quad (left image -> left eye, right -> right eye,
           paired on the sim stamp)
    left   chase camera          right  YOLO on the left eye
    below  controls / robot pose (URDF FK) / your input
  sticks  -> base_twist -> /k1_0/cmd_vel  (L: fwd/strafe, R-X: turn, R-Y: up/down)
  headset -> head_targets -> AAHead_yaw, Head_pitch  ─┐
  --arm-src ik   : grip pos -> clutch -> K1 4-DoF IK ─┼─► /k1_0/teleop/upper_body_cmd (JointState)
  --arm-src soma : HMD-Poser -> /k1_0/teleop/smpl_pose -> soma_retarget_node (arms) ─┘
k1_teleop/yolo_node.py : /k1_0/zed/left/image_raw -> /k1_0/yolo/image_raw (+ /yolo/detections JSON)
src/k1_sim_isaac/scripts/k1_teleop_sim.py (Isaac Lab, headless, bundled Jazzy rclpy)
  K1 on a fixed-base gantry, ZED pair on Head_2, chase cam, people + boxes
  -> /k1_0/joint_states, /k1_0/base_pose, /k1_0/{zed/left,zed/right,chase}/image_raw
  -> debug video logs/k1_teleop_sim/videos/*.mp4 (overview / top / chase / ZED L / ZED R / YOLO)
```

Buttons: A arm follow · B pause · X re-centre panels + "straight ahead" for the
head · Y head follow on/off · left-stick click T-pose calibration.

## 2. Decisions and facts found on the way

* **Joint names are the URDF's, sort prefixes included**: `AAHead_yaw`,
  `ALeft_Shoulder_Pitch`, `ARight_Shoulder_Pitch` (CLAUDE.md lists them without
  the prefix; `k1m/schema.py` already uses the real ones).
* **K1 arm = 4 DoF, no wrist.** `Elbow_Pitch` twists the upper arm;
  `Elbow_Yaw` (at `*_hand_link`) is the elbow bend. IK tracks hand-tip
  position only (tip 0.20 m along the forearm; reach 0.366 m from the roll
  joint). A straight arm (elbow at its 0 limit) has no gradient back toward
  the shoulder, so the solve starts with the elbow bent 0.15 rad.
* **Suspension = gantry.** The URDF gets a world-fixed `gantry_world` link and
  `gantry_x/y/z` (prismatic) + `gantry_yaw` joints above `Trunk`
  (fixed-base articulation, position + velocity drives, 2e4 N/m, 1e3 Nm/rad).
  Two simpler routes failed in Isaac Lab 3.0 EA: PhysX ignores root-pose
  writes on a fixed-base articulation (robot stayed at the origin), and a
  FixedJoint to a kinematic body is not dragged when that body's pose is
  written. Gains 1e6/1e4 on 0.05 kg virtual links oscillated (legs flailing,
  3.7e5 N efforts); the current gains track to <= 1 cm (z sags 1 cm under
  gravity) and exact yaw.
* **Stereo in the headset**: `isaacteleop.viz.QuadLayerConfig.stereo = True`,
  `submit(left, right)` (checked in the 1.4.145 wheel). Window mode shows the
  left eye only.
* ZED lenses at (0.065, ±0.060, 0.085) m in `Head_2` (front plate of
  `Head_2_ZED.STL`), HFOV 101°, 960x540 per eye by default (`--eye-size`,
  must match between sim and XR node).
* Base motion is integrated in **sim time**: at RTF 0.3 the base moves at 0.3x
  wall speed. dl should be faster; measure there.

## 3. Verified (laptop, RTX 4060, 2026-09-28)

| check | result |
|---|---|
| `tests/test_k1_teleop.py` (frames, sticks, head, clutch, IK vs FK x400, clutch path) | 7/7 pass |
| sim `--demo` smoke (scripted base / head / IK arms), debug video frames checked | trunk follows command (x 1.00, yaw 86° = commanded), arms/head move, markers visible |
| sim live + fake operator (system Jazzy) + YOLO node | all 4 image topics ~7.3 Hz, head/arm targets reached, base moved + turned on cmd_vel |
| YOLOv8n on ZED left (GPU, Isaac venv) | 10 ms/frame, both people detected (0.85, 0.92) |
| XR node `--display window` against the live sim (`~/.venvs/k1teleop`) | 201 stereo pairs, 201 chase, 142 YOLO frames shown in 30 s; no size errors |
| RTF laptop | 0.27–0.32 with 5 cameras (2 ZED + chase + 2 debug) |

dl, 2026-09-28, sim container only (`dl_k1_teleop_up.sh sim`, GPU 0, isaac-lab
3.0.0-beta2-post1 image; works unchanged), no operator, no YOLO:

| phase | RTF | rates |
|---|---|---|
| recording debug video (ZED L/R + chase + overview + top) | 0.51 | |
| streaming only (ZED L/R 960x540 + chase 640x360, `--cam-hz 30`) | 0.63–0.65 | ZED / chase ~15.8 Hz, joint_states ~31 Hz |

Topics are received both as root and as the host uid (`--ipc host`), so the
XR / YOLO containers (host uid) see the root sim container. `--cam-hz` is
in sim time: wall image rate = cam_hz x RTF.

Full stack on dl (`dl_k1_teleop_up.sh build` then `all`, GPU 0), no headset:
sim RTF 0.61–0.63 with YOLO + XR attached; YOLOv8n on CPU 15.5 Hz, 28 ms/frame,
3 detections; XR node reaches "Televiz up (xr)", CloudXR on 48322/49100.

* The teleop image must use **Cyclone DDS** (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  as the G1 image): with the default Fast DDS it discovers the sim's topics
  but receives no data (Fast DDS <-> the sim's bundled Fast DDS on one host
  goes through shared memory across containers).
* Without a headset the XR loop blocks in `xrWaitFrame`, so
  `/k1_0/teleop/status` is not published; that is expected.
* The sim container runs Python unbuffered (`PYTHONUNBUFFERED=1`), otherwise
  `docker logs` shows its status lines minutes late.

**Headset session on dl (2026-09-28)**: stereo ZED view, panels, sticks, head
and arm follow worked. The first scene's boxes could not be lifted: they were
static colliders (collision, no rigid body). Now the big boxes are dynamic
(3 / 6 / 2 kg, pushable) and a counter (top 0.93 m) in front of the start pose
holds two small boxes (0.2 / 0.3 kg, friction 1.2, combine max). The K1 has no
hands, so a lift is a squeeze between the forearms: `--demo lift` raises the
0.2 kg box 14 cm (0.99 -> 1.13 m) with a 3 cm squeeze.

**Not verified yet**: `--arm-src soma` end to end (see 5).

## 4. Run

Laptop (no headset):

```bash
cd ~/Projects/wt-vr-teleop
source ~/Projects/booster_ws/scripts/phase6_env.sh            # Isaac Lab EA venv, booster_assets
export ROS_DOMAIN_ID=45
$PHASE6_VENV/bin/python src/k1_sim_isaac/scripts/k1_teleop_sim.py --demo --duration 20   # smoke + video
$PHASE6_VENV/bin/python src/k1_sim_isaac/scripts/k1_teleop_sim.py \
    --people-dir ~/Projects/thesis/G1_sim/assets/people &                               # live
$PHASE6_VENV/bin/python -m k1_teleop.yolo_node &
# XR node needs isaacteleop + cupy: ~/.venvs/k1teleop (uv venv, py3.12)
(source /opt/ros/jazzy/setup.bash; PYTHONPATH=$PWD:$PYTHONPATH \
    ~/.venvs/k1teleop/bin/python -m k1_teleop.xr_teleop_node --display window)
python3 -m pytest tests/test_k1_teleop.py -q
```

dl (headset): push this branch, check it out in `~/Projects/booster_ws` there, then

```bash
scripts/dl_k1_teleop_up.sh build && scripts/dl_k1_teleop_up.sh all
# Quest browser: https://nvidia.github.io/IsaacTeleop/client/ -> dl IP -> Connect
scripts/dl_k1_teleop_up.sh down      # shared machine: always
```

## 5. Next

1. **Headset follow-ups**: RTF (fewer / smaller camera frames, debug video off
   after the check), stereo comfort (world-fixed quad vs lag), `--eye-size` vs
   frame rate, lifting the small boxes live.
2. **`--arm-src soma` on dl**: `k1_teleop/soma_retarget_node.py` runs in G1_sim's
   `g1-soma-retarget` image but needs the `booster_k1` target inside
   soma-retargeter (`scripts/mk_soma_k1_target.py` on `feat/retarget-soma-k1`,
   plus the K1 URDF in `desc/`). Build a derived image with it; HMD-Poser
   weights + SMPL-X-derived joints go in `assets/` (gitignored, licensed).
3. File playback (BVH / SMPL / SOMA -> K1 CSV via the existing GMR / SOMA
   converters) streamed onto `/k1_0/teleop/upper_body_cmd`.
4. Walking: swap the gantry for the P2 policy on `/k1_0/cmd_vel`.
