# K1 Walk Capture — Data, Findings, and Plan

Robot **A2** (`10.37.11.3`, hostname `robot`, Ubuntu 22.04 / ROS 2 Humble, aarch64).
Captured 2026-09-29 02:30–02:43 local (12.1 min of walking).

## 1. What was captured, and how

Recording runs **on the robot**, not the laptop. The campus WiFi blocks DDS
multicast between hosts: a laptop-side recorder saw 0 messages in 20 s while SSH
worked fine. Recording locally removes the network from the failure mode.

| Topic | Type | Rate | Payload |
|---|---|---|---|
| `/low_state` | `booster_interface/msg/LowState` | 498 Hz | joint `dq/ddq/tau` + IMU (rpy, gyro) |
| `/joint_states` | `sensor_msgs/msg/JointState` | 498 Hz | clean joint pos/vel |
| `/odometer_state` | `booster_interface/msg/Odometer` | 498 Hz | `x, y, theta` |
| `/tf` | `tf2_msgs/msg/TFMessage` | 498 Hz | **camera frames only — see §3** |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | once | — |
| `/boostercamera/head/raw/rgb/camera_info` | `sensor_msgs/msg/CameraInfo` | 30 Hz | calibration |
| `/boostercamera/head/raw/right/rgb/camera_info` | `sensor_msgs/msg/CameraInfo` | 30 Hz | right-eye calibration |
| `/remote_controller_state` | `booster_interface/msg/RemoteControllerState` | 16–21 Hz | **human joystick input** |
| `/fall_down` | `booster_interface/msg/FallDownState` | 1 Hz | fall labelling |
| `/booster_video_stream` | `sensor_msgs/msg/CompressedImage` | 10 Hz | **frozen — see §4** |

Total **2.5 GB**, 726 s, 1.5 M messages, split into 5-minute segments.

Camera calibration captured separately to `head_cam_intrinsics_*.yaml` because
the `camera_info` topics are unreliable to catch in-bag:
`fx=fy=207.358, cx=237.196, cy=228.022`, 544×448, no distortion, R shows ~2.4°
stereo rectification. **The head camera is a stereo pair**, so depth is derived.

Copies: laptop `real_robot/bags/`, `dl:~/k1_walk_data/`, `zz-bw:/export/scratch/k1_walk_data/`.

## 2. The walk itself — good data

Decoded from `Odometer` (float32 x, y, theta) over 361,975 samples:

- **254.3 m path length**, 12.2 min
- x span 24.1 m, y span 34.8 m, heading swept the full 6.28 rad (360°)
- net displacement 1.74 m — a loop that came back near the start
- 0 falls implied (no fall event; `/fall_down` recorded throughout)

This is a real, substantial walking dataset.

## 3. Behaviour cloning: the controller mapping is there, the TF is not

**What we have.** `RemoteControllerState` decoded 12,307 samples:
`lx, ly, rx, ry` all sweep the full −1…+1 range with 572–1331 distinct values —
continuous analog motion, not a binary remote. Hat inputs were used
(`hat_u` 81, `hat_l` 59, `hat_d` 56, `hat_r` 47 presses). So the human's velocity
intent **is** captured, time-aligned to 498 Hz joint/IMU/odom state.

That gives genuine (observation → action) pairs:
`(joint state, IMU, odometry, joystick) → (lx, ly, rx, ry)`, which is the same
interface a VLM emits for navigation.

**What is missing.** TF is nearly empty. Across 40 sampled windows per segment
the entire tree is:

```
head_point               parent=head_pitch_link
head_color_optical_frame parent=head_pitch_link
```

`head_pitch_link` is referenced but never itself defined, and there is **no
`base_link`, no IMU frame, no odometry frame**. `robot_state_publisher` runs on
the robot but its TF for the body tree is not in the bag (it only emits for joints
whose state changed, and the camera driver is the loud `/tf` publisher).

Consequence: you cannot currently relate the camera to the base, and you have no
body-frame features. Joint-space BC is unaffected (it uses `joint_states`
directly), but any camera- or world-referenced feature is blocked.

**The hard limit, restated.** `/joint_ctrl` has **zero publishers** — verified.
The factory walker's joint-space actions are firmware-internal and can never be
logged. So there is no low-level (joint-target) BC. The velocity/joystick level
above is the ceiling, and the recorded trajectory must serve as a **reference
motion** for motion tracking (BeyondMimic style) rather than something to clone
joint-for-joint.

## 4. Vision: the images are unusable

`/booster_video_stream` logged 2994 messages in segment 1 at a healthy-looking
9.9 Hz. All 2994 were **byte-identical** (one md5), and every message carried the
**same frozen header stamp** `1790619558.306908794` — the moment the camera
driver started, 11 minutes before the walk began. Same in every segment.

The one real frame (viewed) shows a lab interior with two K1s sitting still — a
genuine startup frame. The camera pipeline works; it just stopped updating.

### Can we map the floor with it? No.

Floor mapping needs an image sequence. There is exactly one unique image across
12 minutes, so there is no motion, no parallax, no trajectory, nothing to
triangulate or integrate. A single frame gives you a single view — you could read
a room layout from it, but you cannot build a map or localise.

### Can we run cuVSLAM on it? No — five independent blockers.

1. **No image sequence.** Fatal on its own. The single largest blocker.
2. **No IMU on any topic.** `/booster/ros2_k2_imu` has 0 publishers. IMU exists
   only *inside* `low_state`, as Euler `rpy` + `gyro`. cuVSLAM needs
   `sensor_msgs/Imu` with an orientation quaternion at IMU rate.
3. **No TF chain.** cuVSLAM needs camera↔IMU extrinsics and a world frame. We
   have camera→`head_pitch_link` and nothing else.
4. **Resolution mismatch.** Calibration is 544×**448**; the video stream is
   544×**306**. Applying the intrinsics to the stream would be wrong.
5. **Wrong image type.** cuVSLAM wants a rectified *mono* stream. We captured a
   JPEG colour stream for the RTC video path. The raw left/right topics from
   `mipi_cam` are the correct source but were **not streaming**.

### Root cause of the freeze

`booster-video-stream` serves the RTC/remote-video path, not the ROS camera
driver. It latched a startup frame and republished it. The real camera driver is
`mipi_cam`, whose `/boostercamera/head/raw/*` image topics were dormant while its
`camera_info` topic streamed fine at 30 Hz — an odd split that suggests the image
path needs an explicit start (there is an `/X5CameraControlReq` RPC and a
`/mipi_cam` service surface).

## 5. Plan

### Immediate — make the camera work (blocks all vision work)

1. On the robot, find out why `mipi_cam` publishes `camera_info` but not images.
   Check its log and params, and whether `/X5CameraControlReq` starts the stream.
2. Once raw images flow, record `/boostercamera/head/raw/rgb` + its `camera_info`
   and confirm **consecutive frames differ** before recording a walk again.
3. Validate with `check_bag_freshness.py` — it now fails a bag where any image
   topic has a single unique stamp or payload. It correctly rejects this walk.

### Immediate — fix TF (blocks camera-relative features and all SLAM)

4. Record the body TF. Either capture `robot_state_publisher`'s `/tf` for the
   whole tree (it needs joint states to move, so the robot must actually walk) or
   dump the URDF at `/opt/booster/Gait/configs/K1/robot.urdf` and compute the
   tree offline with `robot_state_publisher --ros-args -p robot_description:=$(cat …)`.
5. That same URDF gives the camera→trunk and IMU→trunk extrinsics, which is
   exactly what cuVSLAM needs. Fetch it to the repo.

### Next — IMU as a real topic (blocks cuVSLAM)

6. Bridge `low_state.imu_state` (Euler rpy + gyro) into a `sensor_msgs/Imu`,
   converting Euler → quaternion, and publish it at 498 Hz. The README already
   warns that the IMU is Euler.
7. `check_imu` must confirm plausible gyro and no NaN before trusting a bag.

### Then — cuVSLAM, in dependency order

8. Re-record a walk with: live raw left images, `camera_info` at matching
   resolution, `sensor_msgs/Imu`, full TF, odom.
9. Convert on `dl` (Ubuntu 22.04 / Humble / CUDA 11.7; the `k1-slam:cuvslam12`
   container is already built and verified). nvblox is not built yet.
10. Feed `/tf` + calibration as a static YAML, and point cuVSLAM at mono left.

### In parallel — the training tracks

11. **P2 velocity curriculum** is still blocked on ±0.5 m/s command range. The
    real robot data now answers whether 3 m/s is even on the hardware, but the
    factory walker is what ran here, so it does not bound our own policy.
12. **Motion tracking for human-like gait**: the 254 m of real walking is the
    reference-motion source we were missing. No GMR install needed if we track
    the robot's own trajectory.
13. **Head-track / kick** unchanged; the soccer topics exist on the robot
    (`/rt/kick_ball`, `rt/robocup_behavior_status`) when we get back to it.

## 6. Honest caveats

- The 12-minute walk is 498 Hz proprioception with no usable vision. It is good
  gait data and a good joystick-demonstration set; it is not a perception dataset.
- The joystick label is the *human's intent*, not the factory walker's actions.
- Head-camera calibration (544×448) does not match the video stream (544×306).
  Do not mix them.
- `robot_state_publisher` is running with the real K1 URDF, which confirms the
  6-DoF legs (3 hip, 1 knee, 2 ankle) — consistent with the earlier retraction.
