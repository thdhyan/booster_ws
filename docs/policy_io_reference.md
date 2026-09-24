# Booster K1 — Policy I/O Reference (ASCII Diagrams)

This document shows the exact input/output tensor shapes and data flow for each trained policy.

---

## Policy Output DoF Summary

| Policy File | Output DoF | Controls | Scale | Arms/Head/Waist |
|-------------|-----------|----------|-------|-----------------|
| `k1_velocity_policy.pt` | **12** | Legs only | 0.25 | ❌ Not in action space |
| `k1_velocity_student.pt` | **12** | Legs only | 0.25 | ❌ Not in action space |
| `p2_move_student.pt` | **12** | Legs only | 0.25 | ❌ Not in action space |
| `p1_basic_student.pt` | **12** | Legs only | 0.25 | ❌ Not in action space |
| `k1_partialctrl_base.pt` | **14** | Legs (12) + Head (2) | 0.25 / 0.5 | ❌ Arms not controlled |

**⚠️ NO policy outputs the full 22 DoF body.** The arms (8 DoF) and waist (K1 has no waist) are always handled separately:
- **In sim**: PD controllers hold arms at default (or randomized pose for partial control)
- **On real robot**: `k1_wbc` (whole-body controller) regulates upper body independently

---

## Why P1/P2 Only Control 12 Leg DoFs (Not Full 22 DoF)

This is a deliberate design choice in the RL training pipeline:

### 1. Sim-to-Real Gap on Arms
- Arms have low torque, high gear ratios, no force sensing — not designed for dynamic balance
- Real arms ≠ Sim arms: friction, backlash, cable routing, motor saturation differ significantly
- If policy learns to use arms for balance in sim, it **fails on hardware**

### 2. Credit Assignment & Sample Efficiency
```
22 DoF action space → massive exploration space, 10× more samples to converge
12 DoF (legs) → focused learning on what actually matters for locomotion
```
- P1/P2 already take 5000+ iterations; 22 DoF could need 50k+

### 3. Legs Are the Balance Actuators
- Ankle/hip strategies provide balance; arms are for manipulation
- Arm-assisted balance in sim often exploits sim artifacts (perfect torque, no latency)

### 4. Separation of Concerns — WBC Architecture
```
┌─────────────────────────────────────────────────────────────┐
│  LOCOMOTION POLICY (12 DoF)                                 │
│  "Where do I put my feet to track vx, vy, wz?"             │
└─────────────────────┬───────────────────────────────────────┘
                      │ q_des[12 legs]
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  WHOLE-BODY CONTROLLER (WBC) — 22 DoF                       │
│  - Tracks leg q_des exactly                                │
│  - Regulates arms/head to default (or task pose)           │
│  - Handles contact forces, friction cones, torque limits   │
│  - Runs at 1-2 kHz (vs policy at 50 Hz)                    │
└─────────────────────────────────────────────────────────────┘
```
- WBC is the right layer for full-body coordination — it's a **QP with exact dynamics**, not a neural net

### 5. Partial Control (14 DoF) Exists for a Reason
- Adds head (2 DoF) for gaze stabilization in visual tasks
- Arms still NOT controlled — randomized + PD-held so policy learns robustness
- Stepping stone toward manipulation-while-walking, not final form

### When Would You Want Arms in Policy?
Only for manipulation-locomotion tasks (carry, push, open door) where arms **must** coordinate with legs. Even then: hierarchical (locomotion 12 DoF + manipulation arm DoF + WBC merger), not flat 22-DoF PPO.

**Evidence**: H1/G1/Unitree/Atlas all use 12-leg DoF locomotion policies + separate upper-body control.

---

## Policy Inventory

| Policy File | Task Family | Gym ID (Deployable) | Obs Dim | Act Dim | Input Mode |
|-------------|-------------|---------------------|---------|---------|------------|
| `k1_velocity_policy.pt` | P2 Velocity | `Isaac-Velocity-Distill-K1-Play-v0` | 48 | 12 | `latest` |
| `k1_velocity_student.pt` | P2 Velocity Distilled | `Isaac-Velocity-Distill-K1-v0` | 480 (48×10) | 12 | `stacked` |
| `p2_move_student.pt` | P2 Move Distilled | `Isaac-Velocity-Distill-K1-v0` | 480 (48×10) | 12 | `stacked` |
| `p1_basic_student.pt` | P1 Basic Stand | `Isaac-Basic-Student-K1-v0` | 42 | 12 | `latest` |
| `k1_partialctrl_base.pt` | Partial Control | `Isaac-Velocity-PartialCtrl-K1-Play-v0` | 68 | 14 | `latest` |

---

## 1. P2 Velocity Policy — `k1_velocity_policy.pt`

**Most commonly used for real-robot walking.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        P2 VELOCITY POLICY (k1_velocity_policy.pt)           │
│                              TorchScript • 48→12                            │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT (48-dim, single step, "latest" mode)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Index │  Dim │  Name                  │ Source                    │ Unit  │
├────────┼──────┼────────────────────────┼───────────────────────────┼───────┤
│  0:3   │  3   │ base_lin_vel           │ Odometry (base frame)     │ m/s   │
│  3:6   │  3   │ base_ang_vel           │ IMU (base frame)          │ rad/s │
│  6:9   │  3   │ projected_gravity      │ IMU quaternion → gravity  │  -    │
│  9:12  │  3   │ velocity_command       │ /cmd_vel (vx, vy, wz)     │ m/s   │
│ 12:24  │ 12   │ joint_pos_rel          │ JointState - default_pos  │ rad   │
│ 24:36  │ 12   │ joint_vel              │ JointState velocity       │ rad/s │
│ 36:48  │ 12   │ last_action            │ Previous policy output    │  -    │
└────────┴──────┴────────────────────────┴───────────────────────────┴───────┘

JOINT ORDER (12 leg joints — MUST match training exactly):
  0: Left_Hip_Pitch    3: Left_Knee_Pitch     6: Right_Hip_Pitch    9: Right_Knee_Pitch
  1: Left_Hip_Roll     4: Left_Ankle_Pitch    7: Right_Hip_Roll    10: Right_Ankle_Pitch
  2: Left_Hip_Yaw      5: Left_Ankle_Roll     8: Right_Hip_Yaw     11: Right_Ankle_Roll

DEFAULT_LEG_POS = [0,0,0, 0,0,0, 0,0,0, 0,0,0]  (all zeros at training init)

OUTPUT (12-dim)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Index │  Name              │  Meaning                              │ Scale │
├────────┼────────────────────┼───────────────────────────────────────┼───────┤
│  0:11  │ action[0:11]       │ Joint position offsets (Δq)         │ 0.25  │
└────────┴────────────────────┴───────────────────────────────────────┴───────┘

COMMAND TO ROBOT:
  q_des = DEFAULT_LEG_POS + 0.25 * action   →  k1_interfaces/JointCommand

ROS2 TOPICS:
  Sub: /{ns}/joint_states    (sensor_msgs/JointState)  @ 50 Hz
  Sub: /{ns}/cmd_vel         (geometry_msgs/Twist)     @ 10-50 Hz
  Sub: /{ns}/imu             (sensor_msgs/Imu)         @ 200 Hz (optional)
  Sub: /{ns}/odom            (nav_msgs/Odometry)       @ 50 Hz (optional)
  Pub: /{ns}/joint_commands  (k1_interfaces/JointCommand) @ 50 Hz
  Pub: /{ns}/policy_obs      (k1_interfaces/PolicyObs) @ 50 Hz (debug)

LAUNCH PARAMS:
  policy_path:=models/k1_velocity_policy.pt
  input_mode:=latest
  action_scale:=0.25
  command_type:=k1_interfaces/JointCommand
```

---

## 2. P2 Velocity Distilled (History) — `k1_velocity_student.pt` / `p2_move_student.pt`

**Smoother gait — uses 10-step observation history (480-dim).**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              P2 VELOCITY DISTILLED POLICY (k1_velocity_student.pt)          │
│                         TorchScript • 480→12  (history-stacked)             │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT (480-dim = 48 terms × 10 history steps, "stacked" mode)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layout: Term-major, oldest→newest per term                                 │
│                                                                             │
│  [ base_lin_vel_t-9 ... base_lin_vel_t ]      →  3×10 = 30                  │
│  [ base_ang_vel_t-9 ... base_ang_vel_t ]      →  3×10 = 30                  │
│  [ projected_gravity_t-9 ... projected_gravity_t ] → 3×10 = 30              │
│  [ velocity_command_t-9 ... velocity_command_t ] → 3×10 = 30                │
│  [ joint_pos_rel_t-9 ... joint_pos_rel_t ]    → 12×10 = 120                 │
│  [ joint_vel_t-9 ... joint_vel_t ]            → 12×10 = 120                 │
│  [ last_action_t-9 ... last_action_t ]        → 12×10 = 120                 │
│                                                                             │
│  TOTAL: 30+30+30+30+120+120+120 = 480                                       │
└─────────────────────────────────────────────────────────────────────────────┘

NOTE: The locomotion_node maintains a 10-step rolling buffer internally.
      Each control cycle (50 Hz), it shifts history and appends latest obs.
      Policy sees the FLATTENED 480-dim vector (term-major order).

OUTPUT (12-dim) — SAME as P2 Velocity
  q_des = DEFAULT_LEG_POS + 0.25 * action

ROS2 TOPICS: Same as P2 Velocity, but add:
  Param: input_mode:=stacked
  Param: history_len:=10

USE CASE: Better disturbance rejection, smoother transitions
TRADEOFF: Slightly higher latency (10-step buffer), more compute
```

---

## 3. P1 Basic Stand — `p1_basic_student.pt`

**Standing balance only — no walking commands.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        P1 BASIC STAND POLICY (p1_basic_student.pt)          │
│                              TorchScript • 42→12                            │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT (42-dim, "latest" mode)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Index │  Dim │  Name                  │ Source                    │ Unit  │
├────────┼──────┼────────────────────────┼───────────────────────────┼───────┤
│  0:3   │  3   │ base_lin_vel           │ Odometry (base frame)     │ m/s   │
│  3:6   │  3   │ base_ang_vel           │ IMU (base frame)          │ rad/s │
│  6:9   │  3   │ projected_gravity      │ IMU quaternion → gravity  │  -    │
│  9:12  │  3   │ velocity_command       │ ALWAYS ZERO for P1        │ m/s   │
│ 12:24  │ 12   │ joint_pos_rel          │ JointState - default_pos  │ rad   │
│ 24:36  │ 12   │ joint_vel              │ JointState velocity       │ rad/s │
│ 36:42  │  6   │ last_action (truncated)│ Previous policy output    │  -    │
└────────┴──────┴────────────────────────┴───────────────────────────┴───────┘

NOTE: P1 uses 42-dim obs (no velocity command in training, or 6-dim last_action).
      The last_action is 6-dim in P1 vs 12-dim in P2.

OUTPUT (12-dim)
  q_des = DEFAULT_LEG_POS + 0.25 * action

USE CASE: Stand still, reject disturbances, no walking
LAUNCH:  input_mode:=latest, velocity commands ignored
```

---

## 4. Partial Control (Legs + Head) — `k1_partialctrl_base.pt`

**Walking + head tracking (14 DoF: 12 legs + 2 head).**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PARTIAL CONTROL POLICY (k1_partialctrl_base.pt)           │
│                              TorchScript • 68→14                            │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT (68-dim, "latest" mode)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Index  │ Dim │ Name                      │ Source                   │ Unit │
├─────────┼─────┼───────────────────────────┼──────────────────────────┼──────┤
│  0:3    │  3  │ base_lin_vel              │ Odometry                 │ m/s  │
│  3:6    │  3  │ base_ang_vel              │ IMU                      │ rad/s│
│  6:9    │  3  │ projected_gravity         │ IMU quaternion           │  -   │
│  9:12   │  3  │ velocity_command          │ /cmd_vel                 │ m/s  │
│ 12:24   │ 12  │ leg_joint_pos_rel         │ JointState - default     │ rad  │
│ 24:36   │ 12  │ leg_joint_vel             │ JointState               │ rad/s│
│ 36:38   │  2  │ head_joint_pos_rel        │ Head joints - default    │ rad  │
│ 38:40   │  2  │ head_joint_vel            │ Head joints              │ rad/s│
│ 40:52   │ 12  │ arm_joint_pos_rel         │ Arm joints - default     │ rad  │
│ 52:64   │ 12  │ arm_joint_vel             │ Arm joints               │ rad/s│
│ 64:68   │  4  │ last_action (legs+head)   │ Previous output (14→4?)  │  -   │
└─────────┴─────┴───────────────────────────┴──────────────────────────┴──────┘

JOINT ORDER (14 DoF output):
  LEGS (12): Same as P2 velocity policy
  HEAD (2):  AAHead_yaw, Head_pitch

OUTPUT (14-dim)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Index │ Name              │ Meaning                              │ Scale │
├────────┼───────────────────┼──────────────────────────────────────┼───────┤
│  0:11  │ leg_action[0:11]  │ Leg position offsets               │ 0.25  │
│ 12:13  │ head_action[0:1]  │ Head position offsets (yaw, pitch) │ 0.5   │
└────────┴───────────────────┴──────────────────────────────────────┴───────┘

COMMAND TO ROBOT:
  Legs:  q_des[leg] = DEFAULT_LEG_POS + 0.25 * leg_action
  Head:  q_des[head] = DEFAULT_HEAD_POS + 0.5 * head_action
  Arms:  NOT controlled by this policy — held by WBC or sim PD controllers

REQUIRES: Head tracking policy running in parallel (separate node)
          for ball/object tracking → provides head reference

LAUNCH: command_type:=k1_interfaces/JointCommand (22 joints, but policy only fills 14)
```

---

## 5. Soccer Composer — Multi-Policy Runtime Composition

**Not a single policy — runtime state machine combining 3 policies.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    K1 SOCCER COMPOSER (k1_soccer_compose.py)                │
│           Runtime composition: Velocity + Head Tracking + Kicking           │
└─────────────────────────────────────────────────────────────────────────────┘

STATE MACHINE:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ┌─────────┐     ball_dist > 2.0m      ┌─────────┐                        │
│   │  WALK   │ ───────────────────────►  │  TRACK  │                        │
│   │         │                           │         │                        │
│   │ Velocity│◄──────────────────────────│ Velocity│                        │
│   │ Policy  │   ball_dist ≤ 2.0m        │ + Head  │                        │
│   └─────────┘                           │ Policy  │                        │
│        │                                └────┬────┘                        │
│        │                                     │                              │
│        │ ball_dist ≤ 0.5m & aligned         │ ball_dist > 2.0m             │
│        ▼                                     ▼                              │
│   ┌─────────┐                           ┌─────────┐                        │
│   │  KICK   │                           │ SEARCH  │                        │
│   │         │                           │         │                        │
│   │ Kick    │                           │ CCW     │                        │
│   │ Policy  │                           │ Rotate  │                        │
│   └─────────┘                           └─────────┘                        │
│        │                                     │                              │
│        │ kick_done / ball_lost              │ ball_found                     │
│        └─────────────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘

POLICIES REQUIRED:
  1. Velocity:      models/k1_velocity_policy.pt     (legs, 48→12)
  2. Head Track:    models/k1_head_tracking_policy.pt (head, 12→2)
  3. Kicking:       models/k1_kicking_policy.pt      (legs, 48→12)

INPUTS (per robot, from sensors):
  /{ns}/joint_states      → full 22 DoF state
  /{ns}/imu               → base ang_vel, projected_gravity
  /{ns}/odom              → base lin_vel
  /{ns}/ball_detection    → (visible, du, dv) from vision
  /{ns}/cmd_vel           → velocity command (vx, vy, wz)

OUTPUTS (22 DoF JointCommand):
  0:11  → Leg joints  (from active leg policy: velocity OR kick)
  12:13 → Head joints (from head tracking policy)
  14:21 → Arm joints  (held at default, regulated by WBC)

BALL DETECTION INPUT (for head tracking):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  detection = [visible, du, dv]                                          │
  │    visible: 1.0 = ball in FOV, 0.0 = lost                               │
  │    du, dv:  normalized pixel offset (-1..1) from image center           │
  │                                                                          │
  │  CCW Search: if visible==0 for > LOST_FRAMES (25 @ 50Hz = 0.5s),        │
  │              issue in-place rotation command (wz=+0.6) to velocity policy │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow Diagram (Real Robot)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REAL ROBOT DATA FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  JOYSTICK/   │     │  LOCOMOTION      │     │  SDK BRIDGE             │
│  NAV STACK   │────►│  NODE            │────►│  (sdk_bridge_node.cpp)  │
│  (cmd_vel)   │     │  (Python)        │     │  ROS2 → FastDDS         │
└──────────────┘     │                  │     │                         │
                     │  ┌───────────┐   │     │  ┌──────────────────┐   │
                     │  │ POLICY    │   │     │  │ LowCmd Publisher │   │
                     │  │ (TorchScript)   │     │  │ rt/joint_ctrl    │   │
                     │  │ 48→12     │   │     │  └────────┬─────────┘   │
                     │  └─────┬─────┘   │     └───────────┼─────────────┘
                     │        │         │                 │
                     │  ┌─────┴─────┐   │                 ▼
                     │  │ History   │   │         ┌───────────────┐
                     │  │ Buffer    │   │         │ BOOSTER K1    │
                     │  │ (10 steps)│   │         │ ROBOT         │
                     │  └───────────┘   │         │ (22 DoF)      │
                     └──────────────────┘         └───────┬───────┘
                                                          │
                     ┌──────────────────┐                 │
                     │  SENSORS         │◄────────────────┘
                     │  (LowState)      │    FastDDS
                     │                  │    rt/low_state
                     │  • Motor pos/vel │
                     │  • IMU           │
                     │  • Odometer      │
                     └────────┬─────────┘
                              │
                     ┌────────┴─────────┐
                     │  SDK BRIDGE      │
                     │  LowState Sub    │
                     │  → /joint_states │
                     │  → /imu          │
                     │  → /odom         │
                     └──────────────────┘

FREQUENCIES:
  Policy inference:     50 Hz  (control loop)
  SDK LowCmd publish:   50 Hz  (PARALLEL mode)
  SDK LowState recv:    500 Hz (robot internal)
  ROS2 joint_states:    50 Hz  (downsampled)
  cmd_vel input:        10-50 Hz
```

---

## Joint Name Mapping Reference

```
URDF / ROS2 (policy order)          SDK JointIndexK1        SDK Enum Value
──────────────────────────────────────────────────────────────────────────
Left_Hip_Pitch                      kLeftHipPitch           10
Left_Hip_Roll                       kLeftHipRoll            11
Left_Hip_Yaw                        kLeftHipYaw             12
Left_Knee_Pitch                     kLeftKneePitch          13
Left_Ankle_Pitch                    kCrankUpLeft            14   ← 4-bar
Left_Ankle_Roll                     kCrankDownLeft          15   ← 4-bar
Right_Hip_Pitch                     kRightHipPitch          16
Right_Hip_Roll                      kRightHipRoll           17
Right_Hip_Yaw                       kRightHipYaw            18
Right_Knee_Pitch                    kRightKneePitch         19
Right_Ankle_Pitch                   kCrankUpRight           20   ← 4-bar
Right_Ankle_Roll                    kCrankDownRight         21   ← 4-bar

Head (not in velocity policy):
AAHead_yaw                          kHeadYaw                0
Head_pitch                          kHeadPitch              1

Arms (not in velocity policy):
ALeft_Shoulder_Pitch                kLeftShoulderPitch      2
Left_Shoulder_Roll                  kLeftShoulderRoll       3
Left_Elbow_Pitch                    kLeftElbowPitch         4
Left_Elbow_Yaw                      kLeftElbowYaw           5
ARight_Shoulder_Pitch               kRightShoulderPitch     6
Right_Shoulder_Roll                 kRightShoulderRoll      7
Right_Elbow_Pitch                   kRightElbowPitch        8
Right_Elbow_Yaw                     kRightElbowYaw          9
```

---

## Quick Validation Commands

```bash
# Check policy input/output dimensions
python3 -c "
import torch
p = torch.jit.load('models/k1_velocity_policy.pt')
print('Input:', p(torch.zeros(1,48)).shape)   # should be [1, 12]
print('Output dim:', p(torch.zeros(1,48)).numel())  # should be 12
"

# For history-stacked policies
python3 -c "
import torch
p = torch.jit.load('models/k1_velocity_student.pt')
print('Input:', p(torch.zeros(1,480)).shape)  # should be [1, 12]
"

# Verify joint order matches
python3 -c "
LEG_JOINTS = [
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw',
    'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw',
    'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll',
]
print('Policy joint order:', LEG_JOINTS)
print('Count:', len(LEG_JOINTS))
"
```