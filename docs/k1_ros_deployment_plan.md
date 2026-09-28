# Booster K1 — RL Policy → Real Robot ROS 2 Deployment Plan

Date: 2026-09-28 · Worktree: `/home/thakk100/Projects/booster_ws` (branch `feat/video-to-motion`)
Type: research + code inspection. **Nothing was run on hardware.** Everything below is from
repo inspection + official Booster sources. Items I could not verify are marked **UNVERIFIED**.

---

## 0. Executive summary

1. **Our 22-DoF joint order is CORRECT** — confirmed against three independent sources
   (SDK enum, official docs table, official `booster_deploy` K1 config, and our URDF).
2. **The official command path exists and is documented**: DDS topic `rt/joint_ctrl`
   (`booster_interface/msg/LowCmd`, position PD per joint, **radians**), **effective only in
   Custom mode**, plus an RPC service for mode switching. On-robot ROS 2 exposes the same
   contract as `/joint_ctrl`, `/low_state`, `booster_rpc_service`.
3. **The real, vendor-sanctioned RL deployment path is `BoosterRobotics/booster_deploy`**,
   which runs **on the robot** (SSH → `source /opt/booster/BoosterRos2Interface/install/setup.bash`
   → `python3 scripts/deploy.py --task <TASK>`), TorchScript or ONNX-CPU inference at
   **50 Hz** (`policy_dt = 0.02`), with a defined safety layer (posture precondition,
   hold-prime, 1 s ramp, fall detection, explicit exit mode). Our guide never mentions it.
4. **Our `sdk_bridge_node.cpp` would not work as-is** — biggest holes: no mode switch to
   Custom (commands silently ignored), no IMU/odom feed to the policy (obs frozen → fall),
   PD gains ~3–10× below both training and official values (and 0 for arms/head),
   PARALLEL/SERIAL convention differs from official code.
5. **The ankle 4-bar gap is narrower than feared but still UNVERIFIED**: sim and official
   deploy both use *ankle pitch/roll space* on motor indices 14/15/20/21 (CrankUp/CrankDown);
   no crank kinematics exist in any model we hold. Firmware-side mapping is implied, not
   documented → must be proven with a bench test before load-bearing use.
6. **`scripts/motion_feasibility_gate.py` is a good reference-motion gate but NOT a
   pre-flight gate for hardware** — it never sees the policy's output. Missing checks listed
   in §7.

---

## 1. Verified ROS 2 / DDS contract for commanding K1 joints

### 1.1 Command (laptop-side C++ SDK path — what `sdk_bridge_node.cpp` uses)

| Field | Value | Source |
|---|---|---|
| Topic | `rt/joint_ctrl` | `sdk/booster_robotics_sdk/include/booster/robot/b1/b1_api_const.hpp:10`; official docs [Low-Level Topics](https://docs.booster.tech/docs/developer-guide/cpp/low-level-topics/) |
| Type | `booster_interface::msg::LowCmd` (`{int8 cmd_type, MotorCmd[] motor_cmd}`) | `booster/idl/b1/LowCmd.h`; upstream [LowCmd.msg](https://raw.githubusercontent.com/BoosterRobotics/booster_robotics_sdk_ros2/main/booster_ros2_interface/msg/LowCmd.msg) |
| `cmd_type` | `CMD_TYPE_PARALLEL=0`, `CMD_TYPE_SERIAL=1` | upstream `LowCmd.msg` |
| `MotorCmd` | `{int8 mode; float32 q, dq, tau, kp, kd, weight}` | upstream [MotorCmd.msg](https://raw.githubusercontent.com/BoosterRobotics/booster_robotics_sdk_ros2/main/booster_ros2_interface/msg/MotorCmd.msg) |
| Units | `q` **rad**, `dq` **rad/s**, `tau` **Nm** (low-level); high-level head API uses deg | `k1_interfaces/msg/JointCommand.msg` (`positions # rad`); `booster_deploy/robots/k1.py` values are URDF-rad; SDK example `b1_low_sdk_example.cpp:72-77` uses rad |
| Control law | `tau = kp*(q_des - q) + kd*(dq_des - dq) + tau_ff` per motor, position style | SDK example `b1_low_sdk_example.cpp:148-154` (sets q, dq, kp, kd, tau each cycle) |
| Indexing | `motor_cmd[i]` where `i = JointIndexK1` value 0..21 | `b1_api_const.hpp:64-96`; official docs joint-index table |
| Rate | example runs `control_dt = 0.02` (**50 Hz**) | `b1_low_sdk_example.cpp:59`; `booster_deploy` `ControllerCfg.policy_dt = 0.02` |
| **Precondition** | robot must be in **Custom mode**: "commands take effect only while the robot is in Custom mode. Call `ChangeMode(RobotMode::kCustom)` first" | official docs Low-Level Topics warning box; `b1_low_sdk_example.cpp:12-14` comment |
| Mode switch API | `B1LocoClient::ChangeMode(RobotMode)` (RPC id 2000), `GetMode` (2017), `GetStatus` (2018) | `b1_loco_client.hpp:55,71`; `b1_loco_api.hpp:25-47` |
| Robot modes | `kDamping=0, kPrepare=1, kWalking=2, kCustom=3, kSoccer=4` | `booster/robot/common/robot_shared.hpp:7-16` |

### 1.2 Command (on-robot ROS 2 path — what official `booster_deploy` uses)

| Field | Value | Source |
|---|---|---|
| Publish | `joint_ctrl` (→ `/joint_ctrl`), type `booster_interface/msg/LowCmd`, **QoS RELIABLE, depth 1, KEEP_LAST** | [booster_robot_controller.py](https://raw.githubusercontent.com/BoosterRobotics/booster_deploy/main/booster_deploy/controllers/booster_robot_controller.py) `create_low_cmd_publisher()` |
| Subscribe | `/low_state`, type `booster_interface/msg/LowState`, **QoS BEST_EFFORT, depth 1, KEEP_LAST** | same file `_start_low_state_subscription()` |
| State fields used | `low_state.imu_state.rpy` (Euler!), `imu_state.gyro`, `motor_state_serial[i].{q,dq,tau_est}` | same file `_low_state_handler()` |
| `cmd_type` | **`LowCmd.CMD_TYPE_SERIAL` (=1)**; `weight` left at 0; `mode` never set | same file |
| RPC | service `booster_rpc_service`, type `booster_interface/srv/RpcService`, `api_id=2000` (ChangeMode) / `2018` (GetStatus), body = JSON | same file `_call_booster_rpc()`, `_change_robot_mode()` |
| Runtime | on the robot: `source /opt/booster/BoosterRos2Interface/install/setup.bash` then `python3 scripts/deploy.py --task <TASK>` | [booster_deploy README](https://github.com/BoosterRobotics/booster_deploy); official [booster_deploy User Guide](https://docs.booster.tech/docs/developer-guide/open-source/booster-deploy/) |
| Robot-side ROS | ROS 2 **Humble** + `booster_interface` "already installed on the robot" | official booster_deploy User Guide, Prerequisites |

⚠ **Convention conflict (must be resolved on hardware):** our bridge uses `cmd_type=PARALLEL(0)`
and reads `motor_state_parallel`; official deploy uses `SERIAL(1)` and reads
`motor_state_serial`. `LowState.msg` carries **both** arrays
([LowState.msg](https://raw.githubusercontent.com/BoosterRobotics/booster_robotics_sdk_ros2/main/booster_ros2_interface/msg/LowState.msg)).
Semantics of PARALLEL vs SERIAL are **UNVERIFIED** (no doc page explains them).
Test: publish one in each mode, observe which one moves the robot and which state array tracks.

### 1.3 State topics (robot → us)

| Topic | Type | Min firmware | Note |
|---|---|---|---|
| `rt/low_state` | `booster_interface/msg/LowState` | ≥ v1.0.0.0 | raw motor + IMU(euler) state; what our bridge already subscribes |
| `rt/joint_states` | `sensor_msgs/msg/JointState` | ≥ v1.7.1.0 | "requires the ROS bridge" — enabled how, on a stock robot, **UNVERIFIED** |
| `rt/imu/data` | `sensor_msgs/msg/Imu` (quaternion!) | ≥ v1.7.1.0 | same caveat |
| `rt/odom` | `nav_msgs/msg/Odometry` (has twist) | ≥ v1.7.1.0 | same caveat |
| `rt/fall_down` | `booster_interface/msg/FallDownState` | ≥ v1.2.0.2 | fall detection, currently unused by us |
| `rt/robot_states` | `RobotStatesMsg` (RobotMode/BodyControl/Action) | ≥ v1.3.1.1 | mode monitoring |
| `rt/trained_traj_status` | `TrainedTrajStatus` | ≥ v1.8.0.9 | only for the custom-trained-traj path |

Source for the whole table: official [Low-Level Topics](https://docs.booster.tech/docs/developer-guide/cpp/low-level-topics/)
(firmware column) + `b1_api_const.hpp:10-24`. SDK already ships IDL headers for
`sensor_msgs/JointState`, `sensor_msgs/Imu`, `nav_msgs/Odometry`
(`sdk/.../include/booster/idl/{sensor_msgs,nav_msgs}/`), so a laptop-side subscriber for the
`rt/*` ROS-like topics is implementable with `ChannelSubscriber` — this is exactly what
`docs/sdk_ros2_audit.md:71-110` recommends and what our bridge does **not** do.

### 1.4 Custom-model RPC (alternative official path, mostly UNVERIFIED)

`b1_loco_api.hpp:941-1050` + `b1_loco_client.hpp:590-630` define:
`LoadCustomTrainedTraj(CustomTrainedTraj{traj_file_path, model: CustomModel{file_path,
params[{action_scale[], kp[], kd[]}], joint_order ∈ {kMuJoCo=0, kIsaacLab=1}}}) → tid`,
then `ActivateCustomTrainedTraj(tid)` / `UnloadCustomTrainedTraj(tid)` (api ids 2032/2033/2034),
status on `rt/trained_traj_status` (fw ≥ 1.8.0.9).

This looks like a **vendor-side "run my model on the robot" API with an explicit joint-order
flag** — potentially the cleanest answer to the joint-order/unit trust problem. But: model file
format, obs contract, control rate, and K1 support are **UNVERIFIED** (no doc page, no example
in the SDK). Do not plan around it; spike it in parallel.

---

## 2. The 22-DoF joint order — CONFIRMED

Claimed order (user) vs sources:

| # | URDF name (`K1_22dof.urdf`, in file order) | `JointIndexK1` (`b1_api_const.hpp:64-96`) | `booster_deploy` `K1_CFG.joint_names` |
|---|---|---|---|
| 0 | `AAHead_yaw` | `kHeadYaw = 0` | `aahead_yaw_joint` |
| 1 | `Head_pitch` | `kHeadPitch = 1` | `aahead_pitch_joint` |
| 2 | `ALeft_Shoulder_Pitch` | `kLeftShoulderPitch = 2` | `aaleft_shoulder_pitch_joint` |
| 3 | `Left_Shoulder_Roll` | `kLeftShoulderRoll = 3` | `left_shoulder_roll_joint` |
| 4 | `Left_Elbow_Pitch` | `kLeftElbowPitch = 4` | `left_elbow_pitch_joint` |
| 5 | `Left_Elbow_Yaw` | `kLeftElbowYaw = 5` | `left_elbow_yaw_joint` |
| 6 | `ARight_Shoulder_Pitch` | `kRightShoulderPitch = 6` | `aaright_shoulder_pitch_joint` |
| 7 | `Right_Shoulder_Roll` | `kRightShoulderRoll = 7` | `right_shoulder_roll_joint` |
| 8 | `Right_Elbow_Pitch` | `kRightElbowPitch = 8` | `right_elbow_pitch_joint` |
| 9 | `Right_Elbow_Yaw` | `kRightElbowYaw = 9` | `right_elbow_yaw_joint` |
| 10–13 | `Left_Hip_Pitch/Roll/Yaw`, `Left_Knee_Pitch` | `kLeftHipPitch..kLeftKneePitch = 10..13` | same (lowercase `_joint`) |
| 14 | `Left_Ankle_Pitch` | **`kCrankUpLeft = 14`** | `left_ankle_pitch_joint` |
| 15 | `Left_Ankle_Roll` | **`kCrankDownLeft = 15`** | `left_ankle_roll_joint` |
| 16–19 | `Right_Hip_Pitch/Roll/Yaw`, `Right_Knee_Pitch` | `kRightHipPitch..kRightKneePitch = 16..19` | same |
| 20 | `Right_Ankle_Pitch` | **`kCrankUpRight = 20`** | `right_ankle_pitch_joint` |
| 21 | `Right_Ankle_Roll` | **`kCrankDownRight = 21`** | `right_ankle_roll_joint` |

- Official docs repeat this table verbatim: https://docs.booster.tech/docs/developer-guide/cpp/low-level-topics/#joint-indices
- `kJointCntK1 = 22` (`b1_api_const.hpp:143`).
- **Verdict: order CONFIRMED.** The user's GMR `K1_serial.xml` cross-check is consistent.
- ⚠ Note: index **10 on K1 is left hip pitch** — the *23-joint default layout* has
  `kWaist = 10` and legs shifted by one (`b1_api_const.hpp:27-62`). Using `JointIndex`
  instead of `JointIndexK1` would offset every leg command by +1. Our bridge correctly uses
  `JointIndexK1` (`sdk_bridge_node.cpp:28,104-117`).

### 2.1 The ankle 4-bar (`CrankUp`/`CrankDown`) — status: **mapped by name, kinematics unverified**

Facts:
- No model we hold contains crank links: `K1_22dof.urdf` and `K1_22dof.xml` define plain
  `Left/Right_Ankle_Pitch` + `_Ankle_Roll` revolute joints (URDF ranges `[-0.87, 0.345]` and
  `[-0.345, 0.345]` rad). `grep -r crank` over the workspace hits only SDK enums, our bridge,
  and docs.
- `booster_train` models the ankle as a **parallel-actuation dynamics wrapper**, not a
  kinematic transmission: `BoosterK1AnkleParaWrapperCfg` (effort_ratio 1/1, velocity_ratio 1/1,
  **armature_ratio 2/2**) wrapping base joint `E4310` with `serial_index = 0 (pitch) / 1 (roll)`
  — `isaac_tasks/booster_train_ref/.../actuator.py:305-367`.
- **Official `booster_deploy` commands and reads indices 14/15/20/21 in ankle-pitch/roll
  space**: `K1_CFG.joint_names[14] = "left_ankle_pitch_joint"`, `motor_cmd[i].q` and
  `motor_state_serial[i].q` are used directly as that joint's position, and the MuJoCo model
  (`K1_22dof.xml`) uses the same values.
- Our bridge already maps `Left_Ankle_Pitch → kCrankUpLeft`, `Left_Ankle_Roll → kCrankDownLeft`
  (`sdk_bridge_node.cpp:109-110`).

**Inference:** firmware translates between ankle (task) space and crank (motor) space on both
command and feedback for indices 14/15/20/21 — otherwise Booster's own deploy code could not
work. **But this is inference from code, not a documented guarantee → UNVERIFIED.**

Bench test required (Step 3 of runbook): with robot suspended in Custom mode, command
`left_ankle_pitch_joint = +0.10 rad` alone (small kp), confirm (a) foot pitch changes in the
expected direction, (b) `motor_state_serial[14].q` reads back ≈ +0.10, (c) sign/scale matches
`K1_22dof.xml` FK. Repeat for roll. Abort plan if readback ≠ command within tolerance.

---

## 3. What's already done vs what's missing

### 3.1 `src/k1_control/k1_control/sdk_bridge_node.cpp` (289 lines, compiles — `install/k1_control/lib/libsdk_bridge_node.so` exists)

| # | Area | Status | Evidence / gap |
|---|---|---|---|
| D1 | DDS publish to `rt/joint_ctrl` with `LowCmd`, PARALLEL | ✅ done | `sdk_bridge_node.cpp:60-62,71` |
| D2 | DDS subscribe `rt/low_state` → `/k1_0/joint_states` (22 names, URDF order) | ✅ done | `:65-68,183-217,193-201` |
| D3 | Name→index mapping via `JointIndexK1` (12 leg joints) | ✅ correct order | `:104-117` |
| D4 | Rate-limited position ramp (2 rad/s) | ✅ done (weak) | `:232-242` |
| D5 | 50 Hz control timer | ✅ matches training decimation & official `policy_dt` | `:45,83-85` |
| D6 | Built as rclcpp component, wired into `control.launch.py` real mode | ✅ | `CMakeLists.txt`, `k1_control/launch/control.launch.py` |
| **M1** | **Never switches robot to Custom mode** (no `B1LocoClient::ChangeMode`, no RPC at all) | ❌ **BLOCKER** | no include of `b1_loco_client.hpp` anywhere; official docs: commands only active in Custom mode |
| **M2** | **Publishes no `/k1_0/imu` and no `/k1_0/odom`** — yet `real.launch.py:95` sets `imu_topic: 'imu'` and locomotion expects them | ❌ **BLOCKER** | bridge has only `joint_state_pub_` + `joint_cmd_sub_` (`:49-53`); policy obs terms 0–8 (`locomotion_node.py:262-272`) freeze: gravity stays default `[0,0,-1]`, ang vel stays 0 → policy flies blind |
| **M3** | **PD gains wrong by 3–10×** and absent for arms/head | ❌ **BLOCKER** | bridge kp `{30.2,21.4,17.8,60.4,35.7}` (`:121-149`) = booster_train's *derived* gains, which our own docs prove cannot stand (`velocity_env_cfg.py:82-87`: "trunk sinks monotonically 0.589→0.076 m"); training uses **100/100/60/100, kd 5/5/3/5** (`velocity_env_cfg.py:94-105`); official K1 configs use **80…/30** (beyond_mimic) or **100…/65** (walk). Arms/head get `kp_map_[i]` default-inserted = **0** → arms limp in Custom mode (official uses 4.0/1.0) |
| **M4** | `cmd_type=PARALLEL`, reads `motor_state_parallel` vs official `SERIAL`/`motor_state_serial` | ⚠️ unverified | `:71,209` vs `booster_robot_controller.py` |
| M5 | Sets `mode(0x01)` and `weight(1.0)`; official code sets neither (`mode=0`, `weight=0`) | ⚠️ unverified | `:250-251`; `b1_low_sdk_example.cpp:107-110` never sets mode; docs example uses `weight(0)` |
| M6 | Only 12 leg joints mapped; arms/head not commandable (known audit P2) | ⚠️ partial | `:104-117`; `docs/sdk_ros2_audit.md:143-146` |
| M7 | `last_pos_` init to 0, not to measured pose → first command slews from fake zero at 2 rad/s | ⚠️ | `:155,221-242` |
| M8 | No joint-limit clamp, no effort clamp, no fall detection (`rt/fall_down` unused), no damping exit on shutdown, no command-age watchdog in bridge (holds last target forever if policy dies) | ❌ missing safety | no reference to `kTopicFallDown`/`RobotMode` in file |
| M9 | `sdk_ip` param declared but **never used**; `ChannelFactory::Init(0, network_interface_)` only | ⚠️ | `:44,57`. Docs call arg `"<network_ip>"`, SDK example calls it `networkInterface`, README runs examples with `127.0.0.1` — semantics **UNVERIFIED** |
| M10 | Does not consume robot-native `rt/joint_states` / `rt/imu/data` / `rt/odom` (audit's recommended fix) | ❌ | `docs/sdk_ros2_audit.md:71-110` |

### 3.2 `guide_real.md` (335 lines)

| # | Item | Status | Evidence / gap |
|---|---|---|---|
| D1 | Network/DDS bring-up, topic names, monitor commands, policy table | ✅ useful | whole file |
| D2 | Safety checklist + troubleshooting | ✅ skeleton | `:213-223` |
| D3 | Honest about stub/implementation status (mostly) | ✅ | `:109,263-269` |
| **G1** | Never mentions **Custom mode / mode machine / remote X+A trigger** | ❌ | absent; official docs make it a hard precondition |
| **G2** | Firmware requirement "≥ v1.2.0" (`:32`) — official needs **≥ v1.4** (docs) / **≥ v1.7.2** (booster_deploy README) for deploy, ≥ v1.7.1.0 for `rt/imu/data` etc. | ❌ wrong | compare sources §1.3 |
| **G3** | Claims bridge "Maps SDK LowState → joint_states / imu / odom" (`:314-317`) — **bridge does not publish imu/odom at all** | ❌ false | see M2 |
| **G4** | `action_scale:=0.25` "must match training" (`:133`) — current P2 velocity cfg trains with **`scale=1.0`, clip ±1.0 rad** (`velocity_env_cfg.py:270-277`, comment: 0.25 "not enough to recover a tilt") | ❌ likely wrong for `k1_velocity_policy.pt` | per-checkpoint verification needed |
| **G5** | E-stop = `pkill` + publishing empty `JointCommand` (`:226-233`) — empty message does nothing; no `ChangeMode(damping)`, no physical e-stop test procedure, no fall reaction | ❌ inadequate | official exit: explicit RPC to `walking`/`damping` (`booster_robot_controller.py` `run()`) |
| **G6** | Discovery story: `sdk_ip` + nmap (`:39-51`) — `sdk_ip` unused by the bridge; real transport is DDS multicast | ⚠️ misleading | M9 |
| **G7** | Never mentions official path `booster_deploy` / running inference on the robot | ❌ | our own `docs/video_to_motion_plan.md:62,241` already flags it |
| **G8** | `imu_topic`/`odom_topic` params set to `imu`/`odom` with no publisher (`:95,135-136`) | ❌ | dead config |
| G9 | Default robot IP `192.168.1.100` (wired-Ethernet K1s reported at `192.168.10.102` by a third party) | ⚠️ **UNVERIFIED** | roboticscenter.ai (third-party, treat cautiously) |

**Bottom line:** the *shape* of our path (policy node → `JointCommand` → bridge → `LowCmd`) is
correct and matches the vendor contract at the message level. The *prerequisites* (mode,
state feeds, gains) and the *safety envelope* are missing, and the guide documents behavior
the code does not have.

---

## 4. What the real deployment path looks like

### Path A (RECOMMENDED baseline): official `booster_deploy`, on the robot

Evidence: [booster_deploy](https://github.com/BoosterRobotics/booster_deploy),
[official User Guide](https://docs.booster.tech/docs/developer-guide/open-source/booster-deploy/),
[booster_train](https://github.com/BoosterRobotics/booster_train) (exports TorchScript/ONNX to
`logs/rsl_rl/<EXP>/<RUN>/exported/`).

1. Train/export in `booster_train` (BeyondMimic for K1) → `.pt`/TorchScript or `.onnx`.
2. Sim2sim: `python scripts/deploy.py --task <TASK> --mujoco` (MuJoCo model
   `K1_22dof.xml`, same gains).
3. Copy repo to the **robot** over SSH; venv; `source /opt/booster/BoosterRos2Interface/install/setup.bash`.
4. `python3 scripts/deploy.py --task <TASK>` → operator prompts: press **X** (remote/key) to
   enter Custom, **A/r** to start RL; velocity masked to zero until triggered.
5. Runtime facts we should copy verbatim:
   - `/low_state` BEST_EFFORT depth1; `joint_ctrl` RELIABLE depth1; policy 50 Hz.
   - **Posture precondition:** in walking prep, refuse Custom if
     `projected_gravity[2] > -0.5` → switch to damping (`booster_robot_controller.py`).
   - **Hold-prime:** publish current-position hold with `prepare_state` gains *before* the
     Custom transition; low-level retains it (`_prime_custom_command`).
   - **Ramp:** `np.linspace(current, prepare_state.joint_pos, 500)` at 2 ms → **1 s ramp**.
   - **Prepare state for K1** (gains + pose): stiffness
     `40,40,40,50,20,20 / 350,350,180,350,250,250 ×2`, damping `1.5,1.5,0.5,1.5,0.2,0.2 /
     7.5,7.5,3,5.5,5,5 ×2`, joint_pos knee `+0.105`, ankle pitch `−0.10`
     (`booster_deploy/robots/k1.py`).
   - **Policy gains (K1 locomotion):** stiffness legs `100,100,100,100,65,65`, arms `20`,
     head `4`; damping legs `2,2,2,2,1,1` (`tasks/locomotion/robots/k1/__init__.py`).
   - **Fall safety:** locomotion policy stops if `projected_gravity[2] > -0.5`; BeyondMimic
     stops if `dot(gravity_real, gravity_ref) < 0.5` (`enable_safety_fallback=True` default).
   - **Exit:** always `ChangeMode("walking")` (default) or `"damping"`; safety abort forces
     damping; inference-process death is caught and triggers exit.
   - **Inference backends:** TorchScript (`.pt/.jit/.torchscript`, optional `.actor` submodule)
     or ONNX Runtime **CPU** — no TensorRT needed at 50 Hz
     (`booster_deploy/utils/policy_runner.py`).
   - **Kd note for parallel joints:** `Kd = 2·ζ·J_eq·(2π·f_n)`, do **not** reuse simulator Kd
     (booster_deploy README) — matters for the ankle (armature_ratio 2).

### Path B (our current design): policy on laptop, SDK bridge over WiFi

Viable — it is the same DDS contract the vendor uses, just crossed over WiFi.
But it must be fixed per §3.1 and it adds risk the official path avoids:

- WiFi latency/jitter on the 50 Hz loop; **UNVERIFIED** whether firmware runs its own
  `joint_ctrl` watchdog if our publisher dies (booster_deploy always exits modes explicitly,
  which suggests you should not rely on one).
- Obs problem is unchanged: our P2 policy consumes `mdp.base_lin_vel`
  (`velocity_env_cfg.py:204`) — **true world-frame base linear velocity is not measurable on
  hardware**; official deploy just feeds zeros (their obs stack omits linear terms —
  `beyond_mimic.py` comments out `root_lin_vel_b`; locomotion policy uses only ang vel +
  gravity + cmd + joints). Our `locomotion_node._odom_cb` fills obs[0:3] from `odom.twist`
  (child-frame convention, frame mismatch vs training) and today nothing publishes odom anyway.
  **Fix options:** (i) retrain/deploy with `base_lin_vel` removed or zeroed + noise in training,
  (ii) feed zeros consistently at deploy *and* retrain, (iii) implement a small state estimator.
  Feeding zeros to a policy trained *with* the term is a sim-to-real lie — pick (i)/(ii).

### Path C (spike only): on-robot custom-trained-trajectory RPC (§1.4)

Official, joint-order aware (`JointOrder::kIsaacLab`!), gains passed as JSON. Details
UNVERIFIED. 1-day spike: call `LoadCustomTrainedTraj` with a trivial model, see what
`tid`/`rt/trained_traj_status` report.

---

## 5. Safety requirements before any `rt/joint_ctrl` command

Hard prerequisites (all map to official code where possible):

1. **Mode machine with confirmation.** `ChangeMode(kPrepare/kCustom/kDamping)` + `GetMode`
   polled until confirmed (booster_deploy retries 20×, 0.5 s apart). Never publish-and-pray.
2. **Physical E-stop + spotter.** Software stop = `ChangeMode(kDamping)` (all motors damping —
   robot falls, so only usable when suspended or prepared for). Hardware e-stop existence on K1
   **UNVERIFIED** (third-party sources say it exists; verify on the unit).
3. **Suspension/fixture for first Custom entry.** Third-party guidance (roboticscenter.ai) says
   CUSTOM mode requires a lifting fixture because locomotion control is off; Booster's own
   `prepare_mode="walking"` instead starts a *stock zero-command walk policy* inside Custom to
   hold the robot upright on the ground. **Either** fixture **or** the sanctioned walking-prep
   policy — never raw Custom on the ground with our untested policy.
4. **Posture precondition** before enabling: `projected_gravity[2] ≤ -0.5`, else abort to damping.
5. **Hold-prime + 1 s ramp** from measured pose (not zeros) with `prepare_state` gains.
6. **Per-joint gain table sourced from training config**, not guessed; arms/head never at
   kp=0 in Custom (use official 4.0/1.0 or prepare-state values).
7. **Joint clamps:** URDF limits (`K1_22dof.urdf`, listed in §2) + rate limit + torque proxy
   with *real* kp/kd; clamp actions before scaling (`clip ±1.0 rad` mirrors training clip).
8. **Watchdogs:** (a) `/low_state` age → stop publishing + exit mode; (b) policy cmd age →
   bridge holds pose then damps; (c) policy-process alive (booster_deploy treats death as abort);
   (d) loop-rate metrics (`low_state_handler`, `policy_step` frequency printed on exit).
9. **Fall detection:** IMU gravity check each cycle (official thresholds above) **and/or**
   subscribe `rt/fall_down` (fw ≥ 1.2.0.2) → `ChangeMode(kDamping)`.
10. **Explicit exit path** in `finally`: damping on abort, walking/damping otherwise.
11. **Known-bad policy veto:** our repo already produced a motion asking `AAHead_yaw` for
    **9.58× its 6 Nm effort limit** and putting a sole **91 mm** through the floor
    (`k1m/pipeline.py:11`). Gate output must be a *hard* precondition (see §7), and head/arm
    joints should be clamped to their 6/14 Nm limits regardless.

---

## 6. Pre-flight gate assessment: `scripts/motion_feasibility_gate.py`

What it does (10 checks, `motion_feasibility_gate.py:531-542`): joint limits (URDF),
joint velocity, joint accel (assumed 80 rad/s²), torque proxy (`tau ≈ I·ddq + kd·m·dq`,
**fixed kp=60, kd=4**, `:366`), foot penetration (20 mm tol), foot float, foot slip,
root height, double-support fraction, static CoM margin.

**Verdict: adequate as a *reference-motion* gate; NOT sufficient as a hardware pre-flight gate.**

Missing checks (concrete):

| # | Missing check | Why it matters |
|---|---|---|
| 1 | **Policy-in-the-loop gate** — roll out the exported policy in sim2sim (MuJoCo, deployment gains) and gate the *commanded targets*, not the CSV | the gate never sees what the robot will actually be told; the 9.58× example came from retarget, but a policy can exceed limits on its own |
| 2 | **Effort estimate with real per-joint kp/kd** (τ ≈ kp·Δq + kd·Δq̇ + gravity load) instead of fixed 60/4 | bridge/official gains differ 4–10× from the assumed 60 → current numbers are off by construction |
| 3 | **Action-scale authority check**: `max|action_scale·clip|` vs URDF range and vs `DEFAULT_LEG_POS` offset (0.25 vs 1.0 ambiguity, G4) | 1.0 scale + ±1.0 clip can request e.g. ankle roll ±1.0 rad vs limit ±0.345 |
| 4 | **Joint-order/hash assertion**: exported policy joint list == `JointIndexK1` table (§2) | silently wrong order = wrong-sign joints = instant fall |
| 5 | **Default/prepare-pose consistency**: `DEFAULT_LEG_POS = 0` (`locomotion_node.py:87`) vs official K1 prepare (knee +0.105, ankle −0.10) and vs walk defaults (hip −0.15, knee +0.3) | policy obs `joint_pos − default` shifts by a constant → biased policy from step 0 |
| 6 | **Ankle crank-space validation** (§2.1) | only unchecked hardware-specific kinematics |
| 7 | **Self-collision / inter-joint proximity** (e.g., shin–thigh, arm–torso) | retargeted dance clips can fold the robot into itself |
| 8 | **Dynamic CoM/ZMP margin** (static margin exists; no support-polygon *rate of change*, no Angular Momentum / flywheel term) | fast upper-body swings unmodeled |
| 9 | **Obs-availability check**: every training obs term must have a hardware source (fails today: `base_lin_vel`; IMU path dead) | gate assumes motion files, not obs feasibility |
| 10 | **Command-rate / latency budget test** (WiFi RTT jitter vs 20 ms period), plus battery sag and thermal derating | ops-layer failures the geometric gate can't see |
| 11 | **Resample-to-50 Hz check**: CSV is 30/50 fps; deployed loop is 50 Hz with `action_filter=0.8` low-pass (`K1LocomotionPolicyCfg`) | filter lag changes tracking error vs the gate's fps assumption |

---

## 7. Ordered deployment runbook (with abort criteria)

Pre-robot (offline, still mandatory):
- **S0.** Export policy → TorchScript/ONNX. Record: obs terms + order, action scale, default
  pose, gains, joint list, control freq, into a `deployment.yaml` next to the checkpoint.
  Gate the reference motion (`motion_feasibility_gate.py`) — **abort if any `fail`**.
  Sim2sim in MuJoCo via `booster_deploy --mujoco` with the *same* gains — **abort if it falls
  in sim**.

Bring-up (robot powered, no Custom mode yet):
1. **Transport + version.** SSH to robot; confirm firmware ≥ v1.4 (target ≥ v1.7.2), confirm
   `/opt/booster/BoosterRos2Interface` exists; from laptop confirm SDK channel reaches
   `rt/low_state`. **ABORT if:** firmware below required, or `rt/low_state` rate < 50 Hz,
   or zero messages for 5 s.
2. **State-only calibration (no commands).** Verify 22 `motor_state` entries, order matches
   §2; IMU rpy ≈ (0,0,0)±0.05 rad upright, gyro ≈ 0 when still; compare `rt/low_state` vs
   `rt/joint_states`/`rt/imu/data` if available. Hand-move one joint (robot limp in damping)
   and confirm index/sign. **ABORT if:** joint count ≠ 22, any order mismatch, IMU NaN/frozen,
   or rpy nonzero by >0.1 rad while upright.
3. **Ankle crank bench test (suspended).** Small kp, ±0.10 rad single-joint commands to
   `left_ankle_pitch` (idx 14) and `left_ankle_roll` (idx 15); check readback and direction vs
   `K1_22dof.xml` FK. **ABORT deployment via crank-space mapping if readback ≠ commanded
   within 0.02 rad or sign is inverted.**
4. **Mode machine dry-run.** `GetMode` → `ChangeMode(kPrepare)` → verify → `ChangeMode(kDamping)`
   → verify. **ABORT if mode never confirms within 10 s (20 retries).**

Static posture test (robot suspended or on stand, spotter + e-stop):
5. **Hold-prime** current pose with `prepare_state` gains, enter Custom, **1 s linear ramp**
   to prepare pose, hold 10 s. **ABORT if:** any joint tracking error > 0.05 rad sustained
   1 s, any effort > 80 % of URDF limit, IMU tilt > 5°, or loop rate < 40 Hz.
6. **Known-pose step test:** ±0.05 rad square wave on one hip at a time. **ABORT if** response
   direction/scale wrong, oscillates, or other joints couple unexpectedly (indicates wrong
   index mapping).

Low-gain supervised ground test:
7. Ground, flat area ≥ 3×3 m, spotter on e-stop. Enter Custom via **sanctioned walking prep**
   (stock zero-command walk policy holds the robot) — **ABORT if** stock prep itself cannot
   stand (indicates gain/transport problems, not ours to debug mid-test).
8. Hand over to our policy at **50 % gains, output clamp ±0.1 rad, zero cmd_vel, 12-joint legs
   only, arms/head at official 4.0/1.0 hold**. Run 10 s. **ABORT if:** gravity-z > −0.6,
   any foot slip > 0.15 m/s, tracking RMSE > 0.1 rad, head/arm joints exceed limits, or
   operator e-stop pressed.

Full tracking:
9. Full gains from `deployment.yaml`, gate-passed motion (or `cmd_vel` ramp: 0.1 → 0.5 m/s).
   Live monitors: policy/low_state rates, joint error, effort, gravity-z, `rt/fall_down`.
   **ABORT on:** `projected_gravity[2] > −0.5` (official threshold), effort > 100 % limit for
   > 2 cycles, `/low_state` age > 100 ms, policy step jitter > 1.5× period, or fall_down
   topic trigger.
10. **Exit:** stop policy → hold-prime → `ChangeMode(kWalking)` (or `kDamping` after any abort)
    → verify mode → power down. Log everything (rate metrics like booster_deploy's
    `METRICS ...` print).

---

## 8. Top blockers, most severe first

1. **No Custom-mode switching anywhere in our stack** — commands would be silently ignored
   while everything "looks running"; worse, partial transitions are undefined. Add
   `B1LocoClient::ChangeMode/GetMode` (or `booster_rpc_service`) with confirmation + retries.
2. **No IMU/odom feed to the policy** — obs gravity/angular-velocity terms freeze at defaults;
   the audit's `ImuState` quaternion bug (`docs/sdk_ros2_audit.md:40-59`) is *still live* in
   effect because the bridge never publishes `imu` at all. Subscribe SDK `rt/imu/data`
   (has quaternion, fw ≥ 1.7.1.0) or translate `imu_state.rpy` (official approach).
3. **Gain mismatch (3–10× too soft; arms/head at 0)** — our own training notes prove those
   kp values cannot hold the robot up (`velocity_env_cfg.py:82-87`). Ship gains from training
   config + official prepare-state table; arms/head never zero.
4. **Obs contract violation: `base_lin_vel` unmeasurable** — must retrain/zero it consistently
   or add an estimator; feeding odom twist (frame-mismatched) or zeros to an unmodified policy
   is unacceptable.
5. **`action_scale` ambiguity 0.25 vs 1.0** (`guide_real.md:133` vs `velocity_env_cfg.py:270-277`)
   — 4× authority error either way; resolve per checkpoint before power is applied.
6. **Ankle crank-space semantics UNVERIFIED** (§2.1) — bench test gate in runbook step 3.
7. **PARALLEL vs SERIAL `cmd_type` / `motor_state_*` choice unverified** vs official code
   — bench test in runbook step 2/5.
8. **No safety layer**: no fall detection, no damping exit, no cmd-age watchdog, no limit
   clamp in bridge; `guide_real.md`'s E-stop procedure is a no-op (`:226-233`).
9. **Policy validation gap**: gate validates reference motion only; known-bad output
   (9.58× head effort, 91 mm sole penetration) already observed (`k1m/pipeline.py:11`).
10. **Transport**: `sdk_ip` unused, DDS-matrix over WiFi, firmware `joint_ctrl` watchdog
    behavior UNVERIFIED — official path runs on the robot; consider moving inference onboard
    (Path A) and keeping the laptop as monitor/E-stop only.
11. **Version drift**: guide requires fw ≥ 1.2.0; reality ≥ 1.4 (docs) / ≥ 1.7.2 (deploy
    README) / ≥ 1.7.1.0 for ROS-like topics; `sdk/` submodule pinned at commit `d5d8f7a`.

---

## 9. Explicitly UNVERIFIED list

- Semantics of `CMD_TYPE_PARALLEL` vs `CMD_TYPE_SERIAL` (and which the K1 firmware expects).
- Whether firmware maps ankle task-space ↔ crank motor-space (inferred from `booster_deploy`).
- Whether stock firmware publishes `rt/joint_states` / `rt/imu/data` / `rt/odom` without a
  robot-side ROS bridge, and at what rate (docs say "requires the ROS bridge").
- Firmware watchdog on `rt/joint_ctrl` loss (no doc found).
- `ChannelFactory::Init(domain, arg)` — interface name vs robot IP (SDK example, README and
  official docs use the arg differently).
- `MotorCmd.mode` values (our `0x01` vs official unset) and `weight` semantics.
- Whether `LoadCustomTrainedTraj` works for K1 and what model format it wants.
- K1 physical E-stop button existence/location; true default robot IP (192.168.1.100 vs
  192.168.10.102).
- Whether `github.com/BoosterRobotics/booster_ros2` exists — **404 as of 2026-09-28**; the
  roboticscenter.ai page that recommends it is unreliable on this point (rest of its K1
  specifics matched our findings but is third-party).
- URDF-vs-hardware effort limits: `K1WalkControllerCfg` overrides hip/knee/ankle effort to
  `30,20,15,35,24,15` while URDF/`K1_CFG` say `30,35,20,40,20,20` — which reflects hardware
  is **UNVERIFIED** (URDF matches `K1_CFG` base; the walk override does not).

---

## 10. Citations

**Official Booster docs**
- Low-Level Topics (command contract, Custom-mode warning, joint-index tables, firmware minimums):
  https://docs.booster.tech/docs/developer-guide/cpp/low-level-topics/
- booster_deploy User Guide (on-robot deploy, fw ≥ v1.4, ROS 2 Humble, `/opt/booster/BoosterRos2Interface`):
  https://docs.booster.tech/docs/developer-guide/open-source/booster-deploy/
- Developer Guide index (booster_gym / booster_train / booster_deploy / booster_assets):
  https://docs.booster.tech/developer-guide/

**Official Booster code**
- `BoosterRobotics/booster_deploy` — README (fw ≥ v1.7.2, prepare/exit modes, remote X/A, Kd formula):
  https://github.com/BoosterRobotics/booster_deploy
- `booster_robot_controller.py` (topics, QoS, RPC, posture check, hold-prime, 1 s ramp, exit mode):
  https://raw.githubusercontent.com/BoosterRobotics/booster_deploy/main/booster_deploy/controllers/booster_robot_controller.py
- `robots/k1.py` (joint order, stiffness/damping, effort limits, prepare_state):
  https://raw.githubusercontent.com/BoosterRobotics/booster_deploy/main/booster_deploy/robots/k1.py
- `tasks/locomotion/locomotion.py` (obs layout, fall check `gravity_z > -0.5`, action filter 0.8):
  https://raw.githubusercontent.com/BoosterRobotics/booster_deploy/main/tasks/locomotion/locomotion.py
- `tasks/locomotion/robots/k1/__init__.py` (K1 walk gains, 20 policy joints):
  https://raw.githubusercontent.com/BoosterRobotics/booster_deploy/main/tasks/locomotion/robots/k1/__init__.py
- `tasks/beyond_mimic/beyond_mimic.py` (safety fallback `dot < 0.5`, action scale = 0.25·effort/stiffness):
  https://raw.githubusercontent.com/BoosterRobotics/booster_deploy/main/tasks/beyond_mimic/beyond_mimic.py
- `booster_deploy/utils/policy_runner.py` (TorchScript / ONNX-CPU backends):
  https://raw.githubusercontent.com/BoosterRobotics/booster_deploy/main/booster_deploy/utils/policy_runner.py
- `BoosterRobotics/booster_train` (Isaac Lab BeyondMimic for K1, TorchScript/ONNX export):
  https://github.com/BoosterRobotics/booster_train
- `BoosterRobotics/booster_robotics_sdk_ros2` (message definitions):
  https://github.com/BoosterRobotics/booster_robotics_sdk_ros2 ·
  [LowCmd.msg](https://raw.githubusercontent.com/BoosterRobotics/booster_robotics_sdk_ros2/main/booster_ros2_interface/msg/LowCmd.msg) ·
  [MotorCmd.msg](https://raw.githubusercontent.com/BoosterRobotics/booster_robotics_sdk_ros2/main/booster_ros2_interface/msg/MotorCmd.msg) ·
  [LowState.msg](https://raw.githubusercontent.com/BoosterRobotics/booster_robotics_sdk_ros2/main/booster_ros2_interface/msg/LowState.msg) ·
  [ImuState.msg](https://raw.githubusercontent.com/BoosterRobotics/booster_robotics_sdk_ros2/main/booster_ros2_interface/msg/ImuState.msg)
- PyPI `booster-robotics-sdk-python` 1.6.3 (2026-09-16): https://pypi.org/project/booster-robotics-sdk-python/

**This workspace (file:line)**
- `sdk/booster_robotics_sdk/include/booster/robot/b1/b1_api_const.hpp:10-24,64-96,141-143`
- `sdk/booster_robotics_sdk/include/booster/robot/b1/b1_loco_api.hpp:25-60,941-1050` (RPC ids, JointOrder, CustomModel)
- `sdk/booster_robotics_sdk/include/booster/robot/b1/b1_loco_client.hpp:55,71,590-630`
- `sdk/booster_robotics_sdk/include/booster/robot/common/robot_shared.hpp:7-16`
- `sdk/booster_robotics_sdk/example/low_level/b1_low_sdk_example.cpp:12-14,59,68,72-97,148-154`
- `src/k1_control/k1_control/sdk_bridge_node.cpp:28-32,44,57,60-75,104-149,155,183-217,219-257`
- `src/k1_bringup/launch/real.launch.py:95` · `src/k1_control/launch/control.launch.py`
- `src/k1_locomotion/k1_locomotion/locomotion_node.py:79-89,139,157-168,262-325`
- `src/k1_description/assets/robots/K1/K1_22dof.urdf` (order + limits), `K1_22dof.xml:56-174`
- `isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_env_cfg.py:56-105,199-252,270-277`
- `isaac_tasks/booster_train_ref/source/booster_train/booster_train/assets/robots/{actuator.py:305-367, booster.py:60-166}`
- `scripts/motion_feasibility_gate.py:181-260,313-545` · `k1m/pipeline.py:11` (9.58× / 91 mm)
- `docs/sdk_ros2_audit.md` · `docs/policy_io_reference.md` · `guide_real.md:32,109,133,213-233,311-318`
- `docs/video_to_motion_plan.md:61-62,241` (already flagged `booster_deploy` as the sim2real path)

**Third-party (weaker, flagged)**
- roboticscenter.ai K1 setup/software/safety pages (lifting fixture for CUSTOM, IP
  `192.168.10.102`, joint_states @500 Hz) — https://www.roboticscenter.ai/hardware/booster-k1/software
  — consistent with our findings but not vendor-sourced; `booster_ros2` repo it recommends is 404.
