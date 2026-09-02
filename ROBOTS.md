# Booster K1 Fleet Configuration

> Generated: 2026-09-02  
> Workspace: `/home/thakk100/Projects/booster_ws`  
> WiFi subnet: `10.86.36.0/24`  
> SSH key: `~/.ssh/booster_k1` (ed25519, passwordless ✅)

---

## 1. Network — Confirmed Device Inventory

| # | Device | SSH Alias | WiFi IP | WiFi Interface | Hostname | Status |
|---|--------|-----------|---------|----------------|----------|--------|
| 1 | **Laptop** (dhyan-LOQ) | — | `10.86.36.89` | `wlo1` | dhyan-LOQ | ✅ |
| 2 | **Router/Gateway** | — | `10.86.36.242` | — | — | ✅ |
| 3 | **K1 A1** | `boosterk1.a1` | `10.86.36.123` | `wlP1p1s0` | robot | ✅ SSH ✅ |
| 4 | **K1 A2** | `boosterk1.a2` | `10.86.36.3` | `wlP1p1s0` | robot | ✅ SSH ✅ |
| 5 | **K1 A3** | `boosterk1.a3` | `10.86.36.18` | `wlP1p1s0` | robot | ✅ SSH ✅ |
| 6 | **K1 B1** | `boosterk1.b1` | `10.86.36.108` | `wlP1p1s0` | robot | ✅ SSH ✅ |
| 7 | **K1 B2** | `boosterk1.b2` | `10.86.36.109` | `wlP1p1s0` | robot | ✅ SSH ✅ |
| 8 | **K1 B3** | `boosterk1.b3` | `10.86.36.248` | `wlP1p1s0` | robot | ✅ SSH ✅ |

### Robot Internal Interfaces (same on all K1)

| Interface | IP | Purpose |
|-----------|-----|---------|
| `usb_eth0` | `192.168.127.101/24` | Internal robot bus (body controller ↔ compute) |
| `enP8p1s0` | `192.168.13.101/24` | Ethernet port |
| `wlP1p1s0` | WiFi IP | WiFi (your network) |

---

## 2. Robot Hardware & Firmware

| Robot | SSH Alias | Firmware | Disk Free | RAM (avail) | ROS2 | Python |
|-------|-----------|----------|-----------|-------------|------|--------|
| **A1** | `boosterk1.a1` | `v1.6.2.2` (release-02145) | 402 GB | 4.2 GB / 7.4 GB | Humble | 3.10 |
| **A2** | `boosterk1.a2` | `v1.6.2.2` (release-02145) | 400 GB | 4.2 GB / 7.4 GB | Humble | 3.10 |
| **A3** | `boosterk1.a3` | `v1.6.2.2` (release-02145) | 402 GB | 4.3 GB / 7.4 GB | Humble | 3.10 |
| **B1** | `boosterk1.b1` | `v1.6.2.2` (release-02145) | 403 GB | 4.4 GB / 7.4 GB | Humble | 3.10 |
| **B2** | `boosterk1.b2` | **`1.7.0.7`** (release-00040) | 401 GB | 4.4 GB / 7.4 GB | Humble | 3.10 |
| **B3** | `boosterk1.b3` | `v1.6.2.2` (release-02145) | 403 GB | 4.4 GB / 7.4 GB | Humble | 3.10 |

> ⚠️ **B2 is on firmware 1.7, the rest are on 1.6.2.2.**  
> The `boosteros` Python SDK requires firmware ≥ 1.7.  
> Options: upgrade all to 1.7, or use the C++ SDK (available on all).

### Software on Robots

| Component | Status | Notes |
|-----------|--------|-------|
| `booster-cli` | ✅ Installed | Commands: `launch`, `upgrade`, `log`, `upload_log` |
| `boosteros` (Python SDK) | ❌ Not installed | Run `pip3 install boosteros` on each robot |
| C++ SDK (`/opt/booster/lib/libbooster_robotics_sdk.a`) | ✅ Available | Pre-built, linked via `/opt/booster/BoosterRos2/` |
| ROS2 packages (on-robot) | ✅ `booster_msgs`, `booster_rpc_bridge`, `booster_video_stream`, `booster_cam_receiver`, `cv_bridge` |
| `robocup_demo` | ✅ On B2 | `~/Workspace/robocup_demo/` |

---

## 3. SSH Access

### Quick Connect

```bash
ssh boosterk1.a1    # → robot@10.86.36.123
ssh boosterk1.a2    # → robot@10.86.36.3
ssh boosterk1.a3    # → robot@10.86.36.18
ssh boosterk1.b1    # → robot@10.86.36.108
ssh boosterk1.b2    # → robot@10.86.36.109
ssh boosterk1.b3    # → robot@10.86.36.248
```

### Remote Command (no interactive shell)

```bash
ssh boosterk1.a1 'free -h'
ssh boosterk1.b2 'source /opt/ros/humble/setup.bash && ros2 topic list'
```

### SCP File Transfer

```bash
scp myfile.txt boosterk1.a1:~/Workspace/
scp boosterk1.b2:~/Workspace/robocup_demo/config/config.yaml ./local_config.yaml
```

### SSH Config Location

`~/.ssh/config` — entries use `~/.ssh/booster_k1` key, user `booster`, `StrictHostKeyChecking accept-new`.

---

## 4. SDK — Direct Python Control

### Install on Each Robot

```bash
ssh boosterk1.a1
pip3 install boosteros          # base SDK
pip3 install "boosteros[brain]" # with vision + speech
```

> ⚠️ `boosteros` requires firmware ≥ 1.7. B2 is ready; A1/A2/A3/B1/B3 need upgrade first.

### Connection Model

The SDK uses **FastDDS** internally. For **multi-robot**, use `domain_id` to isolate each robot's DDS traffic:

| Robot | ROS_DOMAIN_ID | Namespace |
|-------|---------------|-----------|
| A1 | `0` | `k1_0` |
| A2 | `1` | `k1_1` |
| A3 | `2` | `k1_2` |
| B1 | `3` | `k1_3` |
| B2 | `4` | `k1_4` |
| B3 | `5` | `k1_5` |

### Single Robot Example (run ON the robot)

```python
from boosteros.robots.booster import BoosterRobot

robot = BoosterRobot()
info = robot.robot_info
print(f"{info.manufacturer}, {info.model}, {info.serial_number}")

joints = robot.get_joint_states()
imu = robot.get_imu()
img = robot.get_image(img_type="rgb")
img.save("robot_view.jpg")
```

### Multi-Robot Fleet (run from laptop, one domain per robot)

```python
from boosteros.robots.booster import BoosterRobot

fleet = {}
for i, name in enumerate(["A1", "A2", "A3", "B1", "B2", "B3"]):
    robot = BoosterRobot(domain_id=i)
    fleet[name] = robot
    print(f"{name}: {robot.robot_info.serial_number}")

# Read IMU from all
for name, robot in fleet.items():
    imu = robot.get_imu()
    print(f"{name} rpy={imu.rpy}")
```

### Key SDK APIs

| Method | Description |
|--------|-------------|
| `robot.robot_info` | Manufacturer, model, serial number |
| `robot.get_mode()` | DAMP / PREP / WALK / CUSTOM / PROTECT |
| `robot.set_mode(mode)` | Switch operating mode |
| `robot.get_joint_states()` | 22-DoF joint positions |
| `robot.get_imu()` | Roll/pitch/yaw, angular velocity |
| `robot.get_image(img_type)` | `rgb` or `depth` camera frame |
| `robot.get_odometry()` | Position/velocity estimate |
| `robot.get_battery()` | Charge level, voltage |
| `robot.set_velocity(vx, vy, vyaw)` | Walking velocity (WALK mode) |
| `robot.execute_action(name)` | Predefined动作 |
| `robot.play_audio(file)` | Play audio file |

---

## 5. ROS2 Wrapper — Namespaced Multi-Robot Control

### Architecture

```
┌─────────────────────── Laptop ──────────────────────┐
│                                                      │
│  ┌─────────┐  ┌─────────┐       ┌─────────┐        │
│  │ k1_0    │  │ k1_1    │  ...  │ k1_5    │        │
│  │(ROS2 ns)│  │(ROS2 ns)│       │(ROS2 ns)│        │
│  └────┬────┘  └────┬────┘       └────┬────┘        │
│       │  domain=0  │  domain=1       │  domain=5    │
│       └────────────┼─────────────────┘              │
│              FastDDS / ROS2 DDS                      │
└──────────────────┬──────────────────────────────────┘
                   │ WiFi 10.86.36.0/24
      ┌────────────┼────────────────┐
      │            │                │
 ┌────┴────┐  ┌────┴────┐     ┌────┴────┐
 │ K1 A1   │  │ K1 A2   │ ... │ K1 B3   │
 │10.86.36.│  │10.86.36.│     │10.86.36.│
 │  123    │  │   3     │     │  248    │
 └─────────┘  └─────────┘     └─────────┘
```

### ROS2 Packages Already on Robots

```
/opt/booster/BoosterRos2/install/
├── booster_msgs/          # Message definitions
├── booster_rpc_bridge/    # RPC ↔ ROS2 bridge
├── booster_video_stream/  # Camera streaming
├── booster_cam_receiver/  # Camera receiver
└── cv_bridge/             # OpenCV ↔ ROS2 image bridge
```

### Launching ROS2 Nodes Per Robot

```bash
# Terminal 1: Robot A1 (domain 0, namespace k1_0)
ssh boosterk1.a1
source /opt/ros/humble/setup.bash
source /opt/booster/BoosterRos2/install/setup.bash
export ROS_DOMAIN_ID=0
ros2 launch booster_rpc_bridge ...  # or your custom launch

# Terminal 2: Robot A2 (domain 1, namespace k1_1)
ssh boosterk1.a2
source /opt/ros/humble/setup.bash
source /opt/booster/BoosterRos2/install/setup.bash
export ROS_DOMAIN_ID=1
ros2 launch booster_rpc_bridge ...
```

### Topics Per Robot (after launch)

| Topic | Type | Description |
|-------|------|-------------|
| `/{ns}/joint_states` | `sensor_msgs/JointState` | 22-DoF joint positions |
| `/{ns}/joint_commands` | `k1_interfaces/JointCommand` | Joint position targets |
| `/{ns}/imu` | `sensor_msgs/Imu` | IMU data |
| `/{ns}/image_rgb` | `sensor_msgs/Image` | RGB camera |
| `/{ns}/image_depth` | `sensor_msgs/Image` | Depth camera |
| `/{ns}/odom` | `nav_msgs/Odometry` | Odometry |
| `/{ns}/vel_cmd` | `geometry_msgs/Twist` | Velocity commands |
| `/{ns}/battery` | `sensor_msgs/BatteryState` | Battery status |

### Isolating DDS (preventing cross-talk)

Use `ROS_DOMAIN_ID` per robot (0–5). Topics on different domains are invisible to each other:

```bash
# Only see A1's topics:
ROS_DOMAIN_ID=0 ros2 topic list   # → /k1_0/*

# Only see B2's topics:
ROS_DOMAIN_ID=4 ros2 topic list   # → /k1_4/*
```

---

## 6. Workspace SDK Bridge (Your k1_control Package)

Your workspace at `/home/thakk100/Projects/booster_ws/src/k1_control/k1_control/sdk_bridge_node.py` is a stub that bridges ROS2 ↔ SDK. To make it real:

```python
from boosteros.robots.booster import BoosterRobot

class SdkBridgeNode(Node):
    def __init__(self):
        super().__init__('sdk_bridge_node')
        ns = self.declare_parameter('robot_ns', 'k1_0').value
        domain_id = self.declare_parameter('domain_id', 0).value

        self._robot = BoosterRobot(domain_id=domain_id)
        self.get_logger().info(f'Connected to {self._robot.robot_info.serial_number}')

        self._joint_state_pub = self.create_publisher(
            JointState, f'/{ns}/joint_states', 10)
        self.create_timer(0.02, self._poll_sensors)  # 50 Hz
```

```bash
# Launch per robot from laptop:
ROS_DOMAIN_ID=0 ros2 run k1_control sdk_bridge_node \
  --ros-args -p robot_ns:=k1_0 -p domain_id:=0
```

---

## 7. K1 Operating Modes

| Mode | Description | When |
|------|-------------|------|
| **DAMP** | Joints resist but don't hold posture | Safe mode, transport, shutdown |
| **PREP** | Robot stands and holds posture | Before WALK — robot MUST be on flat ground |
| **WALK** | Walk, turn, step, stop, head movement | Primary locomotion mode |
| **CUSTOM** | Joint control handed to user | Secondary development |
| **PROTECT** | Auto-entered on anomaly | Investigate, then restart |

**Normal startup**: DAMP → PREP → WALK

---

## 8. Quick Reference Commands

```bash
# SSH into a robot
ssh boosterk1.a1

# Check firmware version
cat /opt/booster/version.txt

# Check booster-cli options
booster-cli launch --help
booster-cli upgrade --help

# Launch the default agent
booster-cli launch

# Check ROS2 topics (on-robot)
source /opt/ros/humble/setup.bash
source /opt/booster/BoosterRos2/install/setup.bash
ros2 topic list
ros2 topic hz /camera/image_raw

# Check disk/RAM
df -h / && free -h

# Reboot a robot
sudo reboot

# Pull logs
booster-cli log --help
```

---

## 9. References

| Resource | URL |
|----------|-----|
| SDK Docs | https://docs.booster.tech/docs/developer-guide/booster-os-python-sdk/sdk-overview/ |
| SDK Quick Start | https://docs.booster.tech/docs/developer-guide/booster-os-python-sdk/quick-start/ |
| ROS2 Wrapper | https://github.com/BoosterRobotics/booster_robotics_sdk_ros2 |
| RoboCup Demo | https://docs.booster.tech/docs/developer-guide/open-source/robocup-demo/ |
| RoboCup Repo | https://github.com/BoosterRobotics/robocup_demo |
| C++ SDK (in workspace) | `sdk/booster_robotics_sdk/` |

---

## 10. Isaac ROS Compatibility

| | Isaac ROS 3.2 (recommended now) | Isaac ROS 4.6 (future upgrade) |
|---|---|---|
| ROS distro | Humble | Jazzy |
| JetPack | **6.2** ✅ (no change) | 7.2 ⚠️ (major upgrade) |
| Works on Orin NX? | **✅ Yes** | ❌ Needs JetPack flash |
| Laptop | Ubuntu 22.04, CUDA 12+ | Ubuntu 24.04, CUDA 13.2+ |

**Use Isaac ROS 3.2** — directly compatible with current JetPack 6.2 / Humble.

Key packages: `isaac_ros_image_pipeline`, `isaac_ros_nvenc` (hardware video encoding), `isaac_ros_vpi`.

---

## 11. Fleet RViz Status (2026-09-02)

### What works
- ✅ `k1_fleet_rviz` package built and launchable
- ✅ 6 relay parent processes run in domain 0
- ✅ 6 relay child processes spawn in robot domains (spawn mode fix applied)
- ✅ 18 camera topics visible in domain 0 (`/k1_N/boostercamera/head/...`)
- ✅ 6× robot_state_publisher with K1_22dof.urdf (pelvis root link)
- ✅ RViz2 launches with 18-image + TF + RobotModel config

### What's broken
- ❌ **No image data flowing** — topics visible but messages not arriving at RViz
- **Likely cause**: QoS mismatch (robot BEST_EFFORT vs relay) or image transport (compressed vs raw)
- **Debug log**: `logs/fleet_rviz_20260902_154156.log`

### Fix needed when robots are charged
1. Test `ROS_DOMAIN_ID=0 ros2 topic echo /boostercamera/head/rgb --once` from laptop
2. Check if robot publishes compressed or raw images
3. Match relay child QoS to robot's actual QoS
4. Possibly subscribe to compressed topics and use `image_transport` on laptop

### Launch command (when ready)
```bash
cd ~/Projects/booster_ws
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch k1_fleet_rviz fleet_rviz.launch.py
```

### Stop command
```bash
pkill -f "fleet_rviz\|domain_relay\|rviz2\|robot_state_publisher"
```

---

*Last updated: 2026-09-02 | All 6 robots SSH verified. Fleet RViz: topics visible, image flow pending debug.*
