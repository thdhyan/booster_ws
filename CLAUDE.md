# Booster K1 ROS2 Workspace Rules

## 1. ROS Environment Isolation

**CRITICAL: Never mix system ROS2 with Isaac Sim ROS2.**

- **System ROS2**: `/opt/ros/jazzy` (all k1_* packages, tests, CI/CD)
- **Isaac Sim ROS2**: `~/../IsaacLab/isaac6/` (simulation nodes only)

Isaac Sim nodes run in a separate shell that sources the Isaac Sim environment. Always verify which ROS environment is active:
```bash
echo $ROS_DISTRO  # Should be "jazzy" for k1_* work
which ros2       # Should be /opt/ros/jazzy/bin/ros2
```

## 2. Package & Node Naming Convention

**All packages must be prefixed `k1_`** (e.g., `k1_interfaces`, `k1_description`, `k1_locomotion`).

Every ROS2 node must accept a `robot_ns` parameter (default: `k1_0`). Use it consistently:

- **Topics**: `/{robot_ns}/topic_name` (e.g., `/k1_0/joint_commands`)
- **Services**: `/{robot_ns}/service_name`
- **TF frames**: `{robot_ns}/frame_name` (e.g., `k1_0/base_link`, `k1_0/left_foot`)
- **Nodes**: Use `node_name_ns` or namespace remapping in launch files

Example node initialization:
```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.declare_parameter('robot_ns', 'k1_0')
        robot_ns = self.get_parameter('robot_ns').value
        self.topic_prefix = f'/{robot_ns}'
```

## 3. Joint Names (Exact—Match booster_assets)

All joint control and state must use these exact names from the URDF/booster_assets:

**Head (2 DoF):**
- `Head_yaw`
- `Head_pitch`

**Left Arm (5 DoF):**
- `Left_Shoulder_Pitch`
- `Left_Shoulder_Roll`
- `Left_Elbow_Pitch`
- `Left_Elbow_Yaw`
- (wrist joint not included in 22 DoF count)

**Right Arm (5 DoF):**
- `Right_Shoulder_Pitch`
- `Right_Shoulder_Roll`
- `Right_Elbow_Pitch`
- `Right_Elbow_Yaw`

**Left Leg (5 DoF):**
- `Left_Hip_Pitch`
- `Left_Hip_Roll`
- `Left_Hip_Yaw`
- `Left_Knee_Pitch`
- `Left_Ankle_Pitch`
- `Left_Ankle_Roll`

**Right Leg (5 DoF):**
- `Right_Hip_Pitch`
- `Right_Hip_Roll`
- `Right_Hip_Yaw`
- `Right_Knee_Pitch`
- `Right_Ankle_Pitch`
- `Right_Ankle_Roll`

**Total: 22 DoF** (2 head + 5 each arm + 5 each leg)

## 4. Locomotion Policy & Control Separation

- **Locomotion policy**: Controls 12 leg joints only (Hip/Knee/Ankle, all 4 legs)
- **Head & Arms**: Regulated separately (upper-body controllers, gesture nodes, manual commands)
- **Frequency**: 50 Hz control loop (0.02 s period)
- **Observation history**: 10 steps (0.2 s buffer)

Locomotion outputs: `/{robot_ns}/leg_commands` (12-dim action). Head/arm outputs: `/{robot_ns}/arm_commands`, `/{robot_ns}/head_commands` (separate msg types).

## 5. Build & Source

```bash
cd /home/thakk100/Projects/booster_ws
colcon build --packages-select k1_interfaces k1_description k1_locomotion ...
source install/setup.bash
```

Always source from workspace root after builds.

## 6. Simulation Modes

Launch files must accept `sim` argument:
```bash
ros2 launch k1_launch example.launch.py sim:=gazebo robot_ns:=k1_0
ros2 launch k1_launch example.launch.py sim:=isaac robot_ns:=k1_1
```

Default: `sim:=gazebo`. URDF/SDF handled by `k1_description` package (conditional includes).

For Isaac Sim: launch nodes in a **separate terminal** with the Isaac Sim ROS environment sourced.

## 7. Git Workflow

- **Branch strategy**: Feature work on `dev/phase-N` (e.g., `dev/phase-1-walk`), merge to `main` at milestones
- **Tags**: Semantic versioning: `v0.N-name` (e.g., `v0.1-base-control`, `v0.2-rough-terrain`)
- **Submodules**: 
  - **Read-only**: `sdk/booster_robotics_sdk/` — never edit directly
  - **Editable**: `src/k1_description/assets/` — edit on `thakk100/booster_assets` fork, then PR upstream

Example workflow:
```bash
git checkout -b dev/phase-1-walk
# ... make changes ...
git add .
git commit -m "feat: add base locomotion control"
git push origin dev/phase-1-walk
# PR → review → merge to main
git checkout main
git merge dev/phase-1-walk
git tag v0.1-base-control
git push origin main --tags
```

## 8. Policy Models

Policy files (PyTorch, ONNX) are stored in `models/` and tracked via **Git LFS**:
```bash
*.pt
*.onnx
```

Enable LFS in CI: `git lfs install` (done in GitHub Actions setup).

Do NOT gitignore `models/*.pt` — they are committed via LFS.

## 9. Code Style

- **Python (ROS2)**: Use `rclpy`, `ament_python` build type. Match existing node patterns (see `k1_locomotion`).
- **C++**: Use `ament_cmake`, follow ROS2 coding conventions (see `booster_robotics_sdk`).
- **Formatting**: Black for Python (if linter in place), clang-format for C++ (match workspace defaults).
- **Linting**: `pylint` or `flake8` per package standards.

## 10. Testing & CI/CD

- Unit tests in `<package>/test/`.
- Integration tests verify multi-node communication (e.g., locomotion policy → joint commands → sim feedback).
- GitHub Actions runs on `push` to `main` and PRs to `dev/**`.
- Build step: install ROS Jazzy base + rosdep, colcon build core packages, run tests.

---

**Last updated**: 2026-08-18  
**Workspace**: `/home/thakk100/Projects/booster_ws`  
**Distro**: ROS2 Jazzy, Python 3.12
