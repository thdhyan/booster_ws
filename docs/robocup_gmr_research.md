# RoboCup 3v3 Simulation & Motion Pipeline Research

Date: 2026-08-24 · Status: research notes for fleet/RoboCup replication

---

## 1. Booster robocup_demo — architecture analysis

Repo: https://github.com/BoosterRobotics/robocup_demo (cloned to /tmp for reference)

It is the **robot-side autonomy stack** (not a simulator):

| Component | Role |
|---|---|
| `vision` | YOLOv8 (TensorRT/onnx) — ball/robot/field detection from K1 cameras |
| `brain` | Behavior tree decision layer; publishes SDK LocoApi RPCs |
| `game_controller` | Referee UDP packets → ROS2 topics |
| `src/interface/booster_ros2_interface` | **Vendored `booster_msgs`** (RpcReqMsg etc.) — no /opt/booster needed |

### Multi-fleet sim launcher (their contract)

`scripts/sim_start_multi.sh` runs 3v3 (teams 29 vs 30, GK+2 strikers):
- Simulator side (Booster Studio) must provide per robot `{r}`:
  - `/camera/{r}_rgbd_camera/rgb|depth(/camera_info)` (+ detections topic)
- Brain subscribes those + GameController, then drives the robot via
  `booster_msgs/RpcReqMsg` on `LocoApiTopic/{r}Req` (SDK LocoApi: moveHead,
  standUp, kVisualKick, velocity moves...).

### Replication plan in our stack (feasible, staged)

1. Field world: soccer field SDF/MJCF (5.4×8m RoboCup adult-size lite) + ball.
2. Fleet spawn: reuse `sim_gazebo_fleet.launch.py` / `mujoco_fleet_node.py`
   with field poses instead of a line.
3. Per-robot RGBD camera (gz sensor / mujoco offscreen render).
4. `loco_api_bridge`: sub `LocoApiTopic{r}Req` (vendored msgs) → translate
   API ids to our `/{ns}/joint_commands` + `cmd_vel`.
5. game_controller node from robocup_demo with `sim:=true` as-is.

Effort estimate: bridge ~2 days; field+cameras ~1 day; brain tuning open-ended.

---

## 2. Existing 3v3 RoboCup sims (researched)

| Project | Engine | Robots | Notes |
|---|---|---|---|
| **RCSSServerMJ_GH** | MuJoCo | **Booster T1** (23dof), Ant | Full referee, TCP S-expression agent protocol, monitor port. Closest existing base. https://github.com/albertox1320-dev/RCSSServerMJ_GH |
| circus | MuJoCo | mixed humanoid teams | Newer multi-platform soccer sim |
| SimSpark/rcssserver3d | custom | Nao-class | Official RoboCup 3D league; 2025 rules include a MuJoCo kick challenge |
| BahiaRT 2025 base | MuJoCo | Nao-class | Python team base release |
| Bit-Bots framework | Gazebo classic | humanoids | ROS-native, 5v5 practice games |

Recommendation: prototype 3v3 by porting RCSSServerMJ's T1 model slot to K1
(both Booster, similar kinematics), or adapt our MuJoCo fleet into their
agent protocol.

---

## 3. GMR pipeline (video → K1 motion)

GMR = General Motion Retargeting (ICRA 2026, Ze et al.) — real-time CPU
retargeting SMPL-X/BVH/FBX/PICO → humanoid joint space.
**Booster K1 is officially supported** (22 DoF: neck 2 + arms 2×4 + legs 2×6).

### Already present locally: ~/Projects/Thesis/workin_ws
- `gmr_ros` (sub `/gvhmr_results` → pub `/gmr_results`),
  `gvhmr_ros` (monocular video → SMPL params), `twist_ros`
- GMR assets include `booster_k1/K1_serial.xml` (MuJoCo) + `K1_locomotion.urdf`,
  IK config `ik_configs/smplx_to_k1.json`

### Integration into booster_ws (proposed)
1. Set `gmr_config.yaml: robot_type: booster_k1` (already in supported list).
2. Bridge node in k1_control: sub `/gmr_results` (park_interfaces/GMRResult)
   → pub `/{ns}/joint_commands` → any of our three fleet backends replays it.
3. Batch path: GMR pkl → CSV (`batch_gmr_pkl_to_csv.py`) → BeyondMimic-style
   tracking tasks already in `isaac_tasks/booster_train_ref` → AMP/tracking RL
   for expressive K1 skills on top of the velocity policy.

Pipeline: video → GVHMR → GMR(K1) → {live replay via fleet sim} or
{dataset → motion-tracking RL}.
