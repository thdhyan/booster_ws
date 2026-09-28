# ROS 2 SDK Audit — `BoosterRobotics/booster_robotics_sdk_ros2`

Date: 2026-09-27 · Branch: `feat/video-to-motion`
Verdict: **keep our C++ bridge. Adopt the official message types.**

---

## TL;DR

| Question | Answer |
|---|---|
| Does it replace `src/k1_control/k1_control/sdk_bridge_node.cpp`? | **No.** It is message definitions + two trivial examples. No bridge, no publisher, no transport config. |
| Should we use its messages? | **Yes** — they are field-identical to the SDK IDL and expose a foot/IK state our bridge currently drops. |
| Does it help Gazebo? | **Indirectly** — see "Sim parity" below. |
| Did it find a bug in our code? | **Yes, one real one.** See below. |

---

## What is actually in the repo

```
booster_ros2_interface/          <- message + service definitions ONLY
  msg/   LowCmd MotorCmd LowState MotorState ImuState Odometer
         FallDownState RobotStatesMsg BoosterApiReqMsg/RespMsg
         HandCommand HandDdsMsg HandParam ButtonEventMsg RemoteControllerState
         ProneBodyControlStatus RobotReplayTrajID RawBytesMsg Subtitle …
  srv/   RpcService AgentService
booster_ros2_example/
  low_level/src/low_level_subscriber.cpp   <- 30 lines, subscribes /low_state, logs acc
  rpc_client/  rtc_client/  python_concurrency_example/
```

There is **no `Node` that publishes `LowCmd`**, no `rmw`/zenoh/FastDDS bridge
config, and no `booster_robotics_sdk` CMake package. The examples assume a
`LowState` topic already exists on the bus — which means the robot's own
firmware must be publishing it, not this repo.

## The finding that matters

`ImuState.msg` is **not** a `sensor_msgs/Imu`:

```
float32[3] rpy     # roll, pitch, yaw   <-- EULER, no quaternion
float32[3] gyro
float32[3] acc
```

Our `locomotion_node.py` computes projected gravity from a **quaternion**:

```python
def _imu_cb(self, msg):
    q = msg.orientation                       # <-- all zeros on real hardware
    self._projected_gravity = projected_gravity_from_quat(q.w, q.x, q.y, q.z)
```

`sensor_msgs/Imu.orientation` would be the identity/zero quaternion on any
`ImuState`-sourced feed, so `projected_gravity` silently degenerates to a
constant and the policy loses its gravity reference. **This would not throw —
it would just make the robot fall over**, which is the worst failure mode.

### Second, smaller one: odometry has no twist

```
Odometer.msg:  float32 x, float32 y, float32 theta
```

`locomotion_node._odom_cb` reads `msg.twist.twist.linear` for base linear
velocity. `Odometer` has no twist and no z, so `base_lin_vel` is unobtainable
from it. Positions would also be unbounded floats accumulating drift.

## The fix: use the robot's own ROS-native topics

`b1_api_const.hpp` (our `sdk/booster_robotics_sdk`) already declares these:

```cpp
static const std::string kTopicRosJointStates = "rt/joint_states";
static const std::string kTopicRosImu         = "rt/imu/data";
static const std::string kTopicRosOdometer    = "rt/odom";
```

**The robot firmware publishes proper ROS-native topics.** So for *state* we do
not need to translate anything — we just need to consume the right ones:

| Need | Use | Why |
|---|---|---|
| joint positions/velocities | `rt/joint_states` | already `sensor_msgs/JointState` |
| base orientation + ang vel | `rt/imu/data` | `sensor_msgs/Imu` → **has a quaternion** |
| base linear velocity | `rt/odom` | `nav_msgs/Odometry` → **has twist** |

This removes the entire IMU/odometry translation problem *and* removes most of
the reason the C++ bridge exists.

### Revised bridge responsibilities

```
BEFORE (our bridge does everything)
  ROS2 JointCommand ──► LowCmd ──► rt/joint_ctrl          (command)
  rt/low_state ──► translate ──► /joint_states             (state)
                 └─► translate ──► /imu   (rpy→quat, lossy, BUG)
                 └─► translate ──► /odom  (no twist, unusable)

AFTER (bridge only owns the command path)
  ROS2 JointCommand ──► LowCmd ──► rt/joint_ctrl          (command, our C++)
  rt/joint_states ──────────────────────► /k1_0/joint_states   (remap)
  rt/imu/data     ──────────────────────► /k1_0/imu           (remap)
  rt/odom         ──────────────────────► /k1_0/odom          (remap)
```

`robot_ns` still needs a namespace-aware bridge/relay for the state topics, but
that is a ~40-line node, not the SDK translation we have now.

## Sim parity (the Gazebo angle)

Using `booster_interface/msg/LowCmd` + `MotorCmd` as the command type in
`sim_bridge_node.py` too would make the sim and hardware command paths
**byte-identical**:

```
locomotion_node ──► JointCommand ─┬─► sim_bridge  ──► gazebo controller   (sim)
                                   └─► sdk_bridge  ──► rt/joint_ctrl      (real)
```

Today we have two command types (`k1_interfaces/JointCommand` for Gazebo/real,
`sensor_msgs/JointState` for Isaac) which is a known wart. Unifying on the
official `MotorCmd` (mode, q, dq, tau, kp, kd, weight) would let us pass
per-joint gains and a weight ramp-down for safe shutdown — things
`k1_interfaces/JointCommand` cannot express.

**Caveat:** this is a refactor, not a quick win. It touches every backend. Do it
*after* the video-to-motion pipeline works, not before.

## Concrete actions

1. **[P0]** Add an `imu_source` param to `locomotion_node.py`: `quat` (default,
   `sensor_msgs/Imu`) vs `rpy` (`booster_interface/ImuState`). Fail loudly if
   `rpy` is selected but the quaternion is all-zero. *This is a latent
   hardware-only fall-over bug today.*
2. **[P0]** Point `real.launch.py` state topics at `rt/joint_states`,
   `rt/imu/data`, `rt/odom` and verify the real K1 actually publishes them
   before assuming it does.
3. **[P1]** Vendor `booster_ros2_interface` as a submodule for the message types
   (Apache/default ROS2 licence, no conflict with our BSD-3 assets).
4. **[P2]** Extend `sdk_bridge_node.cpp` to 22 DoF — the dance is full-body, and
   today it only maps 12 leg joints. Add `kHeadYaw`, `kHeadPitch` and the 8 arm
   indices from `JointIndexK1`.
5. **[P3]** Consider the sim/hardware command-type unification above.

## Licences

- `booster_ros2_interface`: default ROS2 licence.
- `booster_robotics_sdk` (our existing submodule): see `sdk/booster_robotics_sdk/LICENSE`.
- `booster_assets` (K1 URDF/XML, meshes): BSD-3-Clause.
