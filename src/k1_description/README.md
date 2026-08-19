# k1_description

Booster K1 humanoid robot URDF, meshes, and robot description package.

## Package Contents

- **urdf/** - Robot URDF xacro files
  - `k1.urdf.xacro`: Main robot description (placeholder until booster_assets submodule is populated)

- **meshes/** - Collision and visual meshes for K1 (STL/DAE files)

- **config/** - Configuration files
  - `ros2_control.yaml`: Hardware interface and controller gains for 22 DoF arm + legs
  - `view_robot.rviz`: RViz2 visualization configuration

- **launch/** - ROS2 launch files
  - `view_robot.launch.py`: Bring up robot_state_publisher, joint_state_publisher_gui, and RViz2

## K1 Kinematics

The Booster K1 is a 22 DoF humanoid with:
- **Head** (2 DoF): yaw, pitch
- **Left Arm** (4 DoF): shoulder pitch/roll, elbow pitch/yaw
- **Right Arm** (4 DoF): shoulder pitch/roll, elbow pitch/yaw
- **Left Leg** (6 DoF): hip pitch/roll/yaw, knee pitch, ankle pitch/roll
- **Right Leg** (6 DoF): hip pitch/roll/yaw, knee pitch, ankle pitch/roll

## Setup

1. Ensure booster_assets submodule is initialized:
   ```bash
   cd /home/thakk100/Projects/booster_ws
   git submodule update --init --recursive
   ```

2. Update the URDF xacro to include the actual K1 model from booster_assets.

3. Build the package:
   ```bash
   cd /home/thakk100/Projects/booster_ws
   colcon build --packages-select k1_description
   ```

## Visualization

Launch the robot viewer:
```bash
ros2 launch k1_description view_robot.launch.py robot_ns:=k1_0
```

This will:
- Parse the URDF and publish TF tree via robot_state_publisher
- Launch joint_state_publisher_gui to manually manipulate joints
- Display the robot in RViz2
