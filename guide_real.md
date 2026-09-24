# Booster K1 — Real Robot Policy Deployment Guide

This guide covers running trained locomotion policies on physical Booster K1 robots over WiFi from your laptop.

---

## Prerequisites

### Hardware
- **Booster K1 robot(s)** powered on and in a safe, open area (≥ 3m × 3m clear space)
- **Laptop** with ROS 2 Jazzy + this workspace built (`colcon build --symlink-install`)
- **WiFi network** connecting laptop ↔ robot(s) (recommended: dedicated 5 GHz AP, robot in AP mode or station mode)
- **E-stop / safety tether** within reach

### Software (on laptop)
```bash
# ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Workspace
cd ~/Projects/booster_ws
source install/setup.bash

# PyTorch CPU (for policy inference)
pip install --user torch --index-url https://download.pytorch.org/whl/cpu
# OR use the Isaac Lab venv which has torch + isaaclab:
# export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH
# /home/thakk100/Projects/IsaacLab/.venv-isaac/bin/python3.12 -m k1_locomotion.locomotion_node ...
```

### Software (on robot)
- Booster Robotics SDK running (firmware ≥ v1.2.0)
- Robot discoverable via SDK IP (default `192.168.1.100` for first robot)

---

## Network Setup

### 1. Find robot IP
```bash
# On laptop: scan for K1 on WiFi
nmap -sn 192.168.1.0/24 | grep -i booster
# Or use the discovery script:
python3 scripts/discover_k1.py  # (create this if needed)
```

### 2. Verify connectivity
```bash
ping 192.168.1.100  # replace with your robot IP
# Should get < 5 ms latency on good WiFi
```

### 3. Configure DDS (CycloneDDS recommended for WiFi)
Create `~/cyclonedds.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
  <Domain id="0">
    <General>
      <NetworkInterfaceAddress>wlan0</NetworkInterfaceAddress>  <!-- your WiFi interface -->
      <AllowMulticast>true</AllowMulticast>
      <MaxMessageSize>65536</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
```

```bash
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
export ROS_DOMAIN_ID=0
```

---

## Available Policies

| Policy File | Description | Obs Dim | Action Scale | Input Mode | Use Case |
|-------------|-------------|---------|--------------|------------|----------|
| `models/k1_velocity_policy.pt` | P2 velocity student (blind, 48-dim, latest step) | 48 | 0.25 | `latest` | **Default walking** — rough terrain, up to 1.5 m/s |
| `models/k1_velocity_student.pt` | P2 velocity distilled student (history-stacked, 480-dim) | 48×10 | 0.25 | `stacked` | **Smoother gait** — uses 10-step history buffer |
| `models/p2_move_student.pt` | P2 move student (same as above, different checkpoint) | 48×10 | 0.25 | `stacked` | Alternative distilled checkpoint |
| `models/p1_basic_student.pt` | P1 basic stand/balance (42-dim) | 42 | 0.25 | `latest` | Standing only, no walking |
| `models/k1_partialctrl_base.pt` | Partial control (legs + head, 14 DoF) | 68 | 0.25 | `latest` | Head-tracking + walking (needs head policy) |

**Recommendation for first real-robot runs**: Use `k1_velocity_policy.pt` with `input_mode:=latest` — simplest, no history buffer dependency.

---

## Single-Robot Bringup

### 1. Start the SDK bridge (on laptop, talks to robot over WiFi)
```bash
# Terminal 1
ros2 launch k1_bringup real.launch.py \
    robot_ns:=k1_0 \
    sdk_ip:=192.168.1.100
```

This launches:
- `k1_control/sdk_bridge_node` — bridges ROS2 `JointCommand` → booster_robotics_sdk → real robot
- Publishes `/k1_0/joint_states` (from robot SDK)
- Subscribes `/k1_0/joint_commands` (from locomotion node)

> **Note**: `sdk_bridge_node.py` is currently a **stub** — you must implement the booster_robotics_sdk integration. See `src/k1_control/k1_control/sdk_bridge_node.py` and the SDK examples in `sdk/booster_robotics_sdk/example/low_level/`.

### 2. Start the locomotion policy node
```bash
# Terminal 2
ros2 run k1_locomotion locomotion_node \
    --ros-args \
    -p robot_ns:=k1_0 \
    -p policy_path:=models/k1_velocity_policy.pt \
    -p control_freq:=50.0 \
    -p action_scale:=0.25 \
    -p command_type:=k1_interfaces/JointCommand \
    -p imu_topic:=imu \
    -p odom_topic:=odom \
    -p input_mode:=latest \
    -p publish_obs_debug:=true
```

**Parameters explained**:
| Param | Value | Notes |
|-------|-------|-------|
| `robot_ns` | `k1_0` | Must match SDK bridge namespace |
| `policy_path` | `models/k1_velocity_policy.pt` | Resolved relative to workspace root |
| `control_freq` | `50.0` | Matches training decimation (4 × 200 Hz physics) |
| `action_scale` | `0.25` | Must match `JointPositionActionCfg.scale` from training |
| `command_type` | `k1_interfaces/JointCommand` | Real robot + Gazebo use this; Isaac Sim uses `sensor_msgs/JointState` |
| `imu_topic` | `imu` | Enables projected gravity + base ang vel from robot IMU |
| `odom_topic` | `odom` | Enables base lin vel from robot odometry |
| `input_mode` | `latest` | Use single-step obs (48-dim); use `stacked` for history policies |
| `publish_obs_debug` | `true` | Publishes `/k1_0/policy_obs` for debugging |

### 3. Send velocity commands
```bash
# Terminal 3: walk forward at 0.5 m/s
ros2 topic pub /k1_0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10

# Turn in place
ros2 topic pub /k1_0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}" -r 10

# Stop
ros2 topic pub /k1_0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --once
```

### 4. Monitor
```bash
# Joint states from robot
ros2 topic echo /k1_0/joint_states

# Policy observations (debug)
ros2 topic echo /k1_0/policy_obs

# Joint commands sent to robot
ros2 topic echo /k1_0/joint_commands
```

---

## Multi-Robot Fleet (WiFi)

### Network topology
```
Laptop (192.168.1.50) ←WiFi→ AP (192.168.1.1) ←WiFi→ Robot k1_0 (192.168.1.100)
                                                          Robot k1_1 (192.168.1.101)
                                                          Robot k1_2 (192.168.1.102)
```

### Launch fleet
```bash
# Terminal 1: SDK bridges for all robots
for i in 0 1 2; do
    ros2 launch k1_bringup real.launch.py \
        robot_ns:=k1_$i \
        sdk_ip:=192.168.1.$((100+i)) &
done
wait
```

```bash
# Terminal 2: Locomotion nodes for all robots
for i in 0 1 2; do
    ros2 run k1_locomotion locomotion_node \
        --ros-args \
        -p robot_ns:=k1_$i \
        -p policy_path:=models/k1_velocity_policy.pt \
        -p control_freq:=50.0 \
        -p action_scale:=0.25 \
        -p command_type:=k1_interfaces/JointCommand \
        -p imu_topic:=imu \
        -p odom_topic:=odom \
        -p input_mode:=latest &
done
wait
```

### Fleet commands
```bash
# All robots walk forward
for i in 0 1 2; do
    ros2 topic pub /k1_$i/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10 &
done
```

---

## Safety Checklist (MANDATORY before every run)

- [ ] **E-stop accessible** — physical button or keyboard shortcut tested
- [ ] **Clear area** — no obstacles, people, cables in 3m radius
- [ ] **Robot posture** — starts from default stand (all leg joints ~0)
- [ ] **Battery > 30%** — check `/k1_0/battery_state` if available
- [ ] **IMU calibrated** — robot upright, stationary for 5s after power-on
- [ ] **Network latency < 10ms** — `ping` test passes
- [ ] **Policy verified in sim** — same policy ran successfully in Gazebo/Isaac/MuJoCo
- [ ] **Joint limits verified** — command targets stay within URDF limits
- [ ] **First command is ZERO velocity** — send `cmd_vel=0` before enabling policy

### Emergency stop
```bash
# Kill locomotion node (stops publishing joint_commands)
pkill -f "locomotion_node.*k1_0"

# Or send zero velocity + zero position command
ros2 topic pub /k1_0/cmd_vel geometry_msgs/msg/Twist "{}" --once
ros2 topic pub /k1_0/joint_commands k1_interfaces/msg/JointCommand "{joint_names: [], positions: []}" --once
```

---

## Troubleshooting

### "policy not found" error
```bash
# Policy path resolves from workspace root
ls -la ~/Projects/booster_ws/models/k1_velocity_policy.pt
# Use absolute path if needed:
-p policy_path:=/home/thakk100/Projects/booster_ws/models/k1_velocity_policy.pt
```

### "joint_states timeout" warning
- Check SDK bridge is running and publishing `/k1_0/joint_states`
- Verify robot IP and WiFi connectivity
- Check `js_timeout` param (default 0.5s) — increase if WiFi is flaky

### Robot falls / unstable gait
- **Reduce velocity command**: start with `vx=0.1`, `vy=0`, `wz=0`
- **Check IMU**: `ros2 topic echo /k1_0/imu` — projected gravity should be ~[0, 0, -1] when standing
- **Check joint offsets**: policy expects `joint_pos - DEFAULT_LEG_POS` where default is all zeros. Verify robot default pose matches.
- **Try history-stacked policy**: `models/k1_velocity_student.pt` with `-p input_mode:=stacked`

### DDS discovery issues
- Ensure `ROS_DOMAIN_ID` matches on laptop and all robots (if robots run ROS nodes)
- Use CycloneDDS with explicit `NetworkInterfaceAddress` (wlan0)
- Disable firewall: `sudo ufw disable` temporarily

### SDK bridge not implemented
The `sdk_bridge_node.py` is a stub. You need to:
1. Build booster_robotics_sdk Python bindings (or use C++ node)
2. Implement `_cmd_cb` to send `MotorCmd` to SDK
3. Implement state publisher reading `MotorState` from SDK → `/joint_states`

Reference SDK example: `sdk/booster_robotics_sdk/example/low_level/b1_low_sdk_example.cpp`

---

## Policy-to-Task Mapping

| Task Family | Gym ID (Deployable) | Policy File | Notes |
|-------------|---------------------|-------------|-------|
| P1 Basic Stand | `Isaac-Basic-Student-K1-v0` | `p1_basic_student.pt` | 42-dim obs, stand only |
| P2 Velocity | `Isaac-Velocity-Distill-K1-Play-v0` | `k1_velocity_policy.pt` | 48-dim, **latest** mode |
| P2 Velocity Distilled | `Isaac-Velocity-Distill-K1-v0` | `k1_velocity_student.pt` / `p2_move_student.pt` | 480-dim, **stacked** mode |
| Partial Ctrl | `Isaac-Velocity-PartialCtrl-K1-Play-v0` | `k1_partialctrl_base.pt` | 68-dim, 14 DoF (legs+head) |

---

## Next Steps

1. **Implement SDK bridge** — complete `src/k1_control/k1_control/sdk_bridge_node.py`
2. **Add whole-body controller** — integrate `k1_wbc` for upper-body regulation
3. **Add perception stack** — ZED stereo → ball detection → soccer composer (`k1_soccer_compose.py`)
4. **Fleet coordination** — role assignment, pass/kick strategies

---

## Policy I/O Reference (ASCII Diagrams)

See **[docs/policy_io_reference.md](docs/policy_io_reference.md)** for complete ASCII diagrams of every policy's input/output tensors, joint mappings, and data flows.

### Quick Summary

| Policy | Input | Output | Mode | Use Case |
|--------|-------|--------|------|----------|
| `k1_velocity_policy.pt` | 48 (latest) | 12 leg Δq | `latest` | **Default walking** |
| `k1_velocity_student.pt` | 480 (48×10 history) | 12 leg Δq | `stacked` | Smoother gait |
| `p2_move_student.pt` | 480 (48×10 history) | 12 leg Δq | `stacked` | Alt. distilled |
| `p1_basic_student.pt` | 42 | 12 leg Δq | `latest` | Stand only |
| `k1_partialctrl_base.pt` | 68 | 14 (12 leg + 2 head) | `latest` | Walk + head |

---

## SDK Bridge Implementation

The C++ SDK bridge is now implemented at:
- `src/k1_control/k1_control/sdk_bridge_node.cpp` — Full FastDDS LowCmd/LowState bridge
- Maps ROS2 `JointCommand` → SDK `LowCmd` (22 motors, PARALLEL mode)
- Maps SDK `LowState` → ROS2 `joint_states` / `imu` / `odom`
- Per-joint PD gains from booster actuator specs
- Rate-limited position commands for safety

Build with:
```bash
colcon build --packages-select k1_control --cmake-args -DCMAKE_BUILD_TYPE=Release
```

---

## References

- `src/k1_locomotion/k1_locomotion/locomotion_node.py` — policy inference node (read the docstring!)
- `src/k1_bringup/launch/real.launch.py` — real robot launch (now complete)
- `src/k1_control/k1_control/sdk_bridge_node.cpp` — **C++ SDK bridge (implemented)**
- `src/k1_control/k1_control/sdk_bridge_node.py` — Python stub (deprecated)
- `isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_env_cfg.py` — training observation/action definition
- `docs/policy_io_reference.md` — **ASCII diagrams for all policies**
- `models/` — trained TorchScript policies (Git LFS)