# real_robot — recording and sync for the Booster K1 on hardware

Everything needed to capture real-robot data while the robot is moving, and to
move the resulting bags between the robot and a workstation. Kept separate from
`isaac_tasks/` (simulation) because this directory talks to a physical machine
and has different safety and failure modes.

## Why this exists

Two separate jobs, often confused:

1. **Record** what the robot actually did — TF, IMU, camera, odometry, joints.
   This is the ground truth we currently have none of. Every sim2real claim so
   far rests on actuator models we *inferred*, not measured.
2. **Sync** the bags. The robot is a network client with finite storage; the
   workstation is where the data gets processed. Bags must move both ways.

## Topic contract

The K1 SDK speaks FastDDS (`rt/low_state`, `rt/odometer_state`). ROS 2 also runs
over DDS, but **there is no bridge between them yet** — see `sdk_bridge_node`
in `src/k1_control/k1_control/` (the Python version is a stub; the C++ one is
the intended path). Until that bridge exists, the topics below are what a
bridged rig must publish for these scripts to be useful.

| Topic | Type | Notes |
|---|---|---|
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | Base → link tree. Built from joint states. **Without TF there is no map.** |
| `/imu/data` | `sensor_msgs/Imu` | Orientation as a **quaternion**. See the warning below. |
| `/odom` | `nav_msgs/Odometry` | Base linear/angular velocity. Gives *achieved* speed, not commanded. |
| `/<cam>/color/image_raw` | `sensor_msgs/Image` | RGB |
| `/<cam>/color/camera_info` | `sensor_msgs/CameraInfo` | **Required.** cuVSLAM rejects uncalibrated frames. |
| `/<cam>/depth/image_raw` | `sensor_msgs/Image` | For NVBloX reconstruction |
| `/joint_states` | `sensor_msgs/JointState` | 22 joints |

### IMU orientation warning

`booster_ros2_interface/msg/ImuState` is **not** a `sensor_msgs/Imu`:

```
float32[3] rpy     # roll, pitch, yaw   <-- EULER, no quaternion
float32[3] gyro
float32[3] acc
```

Any bridge that copies rpy straight into an `Imu.orientation` field produces a
silently wrong rotation and a garbage map. Convert properly. This is a real bug
in our own `locomotion_node.py` per `docs/sdk_ros2_audit.md`.

## Usage

```bash
# 1. discover / check the robot is publishing
source config/env.sh                      # or: source /opt/ros/humble/setup.bash
./check_topics.sh                         # what is alive, rates, TF tree

# 2. record a session (stops cleanly on Ctrl-C, then pulls to ./bags)
./record_session.sh walk_fast --duration 120

# 3. or drive it manually
./rosbag_record.sh --out bags/walk_fast --duration 120 --topics-file config/topics_lidar_walk.yaml
```

## Files

| File | Purpose |
|---|---|
| `check_topics.sh` | Health check: topic rates, TF tree, calibration presence |
| `rosbag_record.sh` | The recorder. Chunked, compressed, size-capped, clean shutdown |
| `record_session.sh` | Orchestrates: pre-flight → record → verify → sftp pull → report |
| `sftp_sync.sh` | Batch pull/push against the robot with resume + checksums |
| `verify_bag.py` | Post-hoc check: duration, topic coverage, message counts, NaN scan |
| `config/` | Topic lists, env template |
| `bags/` | Local bag storage (gitignored) |

## Storage reality

Bags are large — a 120 s RGB-D + IMU + TF session is easily 2–10 GB. That is why:

- `bags/` is gitignored. **Never commit bag data.**
- Recording is chunked (default 512 MB) so a crash loses at most one chunk.
- Sync is resumable and checksummed; a partial transfer is detected, not
  silently accepted.

## Safety

- The recorder is read-only. It never publishes commands.
- `record_session.sh` has a pre-flight that refuses to start if TF or IMU is
  missing, because a bag without them is not worth the disk.
- Recording on a robot that is falling over is still fine; do not add compute
  load to a machine that is mid-fall.
