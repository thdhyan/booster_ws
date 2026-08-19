# k1_bringup

Top-level launch entry points for the Booster K1 humanoid robot.

## Overview

This package provides launch wrappers for all K1 simulation and deployment modes:
- **Gazebo Harmonic**: physics-based simulation with ROS2 bridge
- **Isaac Sim 6.0.1**: visual simulation with real-time rendering
- **Real Robot**: hardware bring-up via booster_robotics_sdk
- **Fleet**: multi-robot coordination (Phase 5)

All nodes respect the `robot_ns` parameter (default `k1_0`) for multi-robot operation.

## Launching

### Gazebo Simulation
```bash
ros2 launch k1_bringup sim_gazebo.launch.py robot_ns:=k1_0
```

Brings up:
1. k1_description (robot_state_publisher)
2. Gazebo Harmonic + K1 URDF
3. ros_gz_bridge (joint_states, odom, cmd_vel)
4. k1_control (sim_bridge_node)
5. k1_locomotion (policy inference)

### Isaac Sim Simulation
```bash
ros2 launch k1_bringup sim_isaac.launch.py robot_ns:=k1_0 usd_path:=/path/to/k1.usd
```

Brings up:
1. k1_description (robot_state_publisher)
2. Isaac Sim 6.0.1 + K1 USD asset
3. OmniGraph ROS2 bridge (ActionGraph)
4. k1_control (sim_bridge_node)
5. k1_locomotion (policy inference)

**Note**: Isaac Sim ROS2 is separate from system ROS2 and lives at `~/Projects/IsaacLab/isaac6`.
Do NOT source both in the same shell.

### Real Robot
```bash
ros2 launch k1_bringup real.launch.py robot_ns:=k1_0 sdk_ip:=192.168.1.100
```

**WARNING**: Ensure robot is powered and in a safe area.

### Multi-Robot Fleet (Phase 5)
```bash
ros2 launch k1_bringup fleet.launch.py n_robots:=6 sim:=gazebo
```

Spawns 6 robots (k1_0..k1_5) with 2m spacing.

## Launch Arguments

Common across all modes:
- `robot_ns` (default: `k1_0`) — robot namespace, used in topic names: `/{robot_ns}/topic_name`
- `policy_path` (default: `models/k1_velocity_policy.pt`) — path to TorchScript locomotion policy

Gazebo-specific:
- `world` (default: `flat`) — world name: `flat` or `rough`
- `headless` (default: `false`) — disable visualizer

Isaac-specific:
- `usd_path` (default: `''`) — path to K1 USD (from booster_assets)
- `headless` (default: `false`) — disable visualizer

Real robot-specific:
- `sdk_ip` (default: `192.168.1.100`) — robot SDK IP address

Fleet-specific:
- `n_robots` (default: `1`) — number of robots (1-6)
- `sim` (default: `gazebo`) — backend: `gazebo` or `isaac`

## Dependencies

- `k1_description` — URDF/mesh assets
- `k1_control` — joint control bridge nodes
- `k1_locomotion` — neural policy inference
- `k1_sim_gazebo` — Gazebo integration
- `k1_sim_isaac` — Isaac Sim integration

## Phase Roadmap

- **Phase 1 (v0.1)**: Gazebo sim bringup
- **Phase 2**: Isaac Sim integration
- **Phase 3**: Real robot hardware interface
- **Phase 4**: Whole-body control (k1_wbc)
- **Phase 5**: Multi-robot coordination

## Notes

- All topics are namespaced: e.g., `/k1_0/joint_states`, `/k1_1/odom`
- Namespace default is `k1_0`; use `robot_ns:=k1_1` for second robot
- Joint states come from Gazebo/Isaac; commands go back via ros_gz_bridge
- Odometry estimated from joint kinematics (Gazebo) or physics engine (Isaac)
