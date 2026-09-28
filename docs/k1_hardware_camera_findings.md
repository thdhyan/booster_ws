# K1 A2 Hardware & Camera Findings — 2026-09-29

Investigating why the Booster K1 (unit **A2**) publishes a full camera ROS graph but
delivers no images, and getting cuVSLAM onto the robot.

**Bottom line: A2 has no camera sensor fitted. The ROS camera graph is intact but
has no hardware behind it.** The one exception is a short window right after boot
where the pair ran at ~7.8 Hz and then stopped — see §2, which is the most
diagnostic finding here and the reason this is not simply "no camera".

---

## 1. Robot identity and network

| | |
|---|---|
| Unit | A2 (`boosterk1.a2`) |
| WiFi | `10.37.11.3` / 24 (`wlP1p1s0`) |
| Wired | `192.168.10.102` / 24 (`enP9p1s0`) — the address in Booster's own docs |
| Other NICs | `192.168.13.101` (`enP8p1s0`), `192.168.127.101` (`usb_eth0`, internal) |
| Hardware | Booster K1 Orin NX V1.0, aarch64 |
| Firmware | `v1.6.1.1-release-01967-2026-04-27` |
| Resources | 4 cores, 7.4 GiB RAM (~4.5 GiB free), 396 GB disk free, CUDA 12.6, JetPack 6, ROS 2 Humble |
| ROS domain | `ROS_DOMAIN_ID=0` (from `/opt/booster/BoosterAgent/start.sh`) |

Laptop side: `enp2s0` needed `192.168.10.10/24` added manually plus
`ip route add 224.0.0.0/4 dev enp2s0` — the host had **no multicast routes at
all** by default, which is why DDS discovery found nothing.

### Network gotcha worth remembering

DDS behaviour is counterintuitive here and cost real time:

- Over **WiFi** (`10.37.11.3`): the Booster Python SDK connected **once**, then
  failed on identical parameters. Flaky.
- Over **wired** `192.168.10.102`: SDK discovery **never** connected
  (`req_matched=0`).
- The AP appears to filter multicast; the wired link is direct but the SDK still
  did not discover, so the wired path is *not* automatically the better one.

The vendor's documented setup (host `192.168.10.10`, robot `192.168.10.102`,
wired) is a real requirement, not boilerplate.

---

## 2. The camera published frames after boot, then stopped ⚠️

This is the key finding. It rules out "there has never been a camera here".

**Immediately after a reboot** (robot up ~2 min), with a `rclpy` subscriber using
`RELIABLE` + `VOLATILE` QoS:

| topic | msgs | Hz |
|---|---|---|
| `/boostercamera/head/rgb` (rectified LEFT) | 93 | **7.8** |
| `/boostercamera/head/right/rgb` (rectified RIGHT) | 93 | **7.8** |
| `/boostercamera/head/depth` | 93 | **7.8** |
| `/boostercamera/head/raw/rgb` | 355 | **29.6** |
| `/booster_video_stream` | 90 | 7.5 |

`booster-video-stream` was alive at that moment (PID 5607, 8.7% CPU).

**Roughly five minutes later**, the same probe:

| topic | msgs | Hz |
|---|---|---|
| `/boostercamera/head/rgb` | 0 | **0.0** |
| `/boostercamera/head/right/rgb` | 0 | **0.0** |
| `/boostercamera/head/depth` | 0 | **0.0** |
| `/boostercamera/head/raw/rgb` | 1 | 0.375 |

`booster-video-stream` had **died**, and with it the rectified pair. No OOM
occurred (4.5 GiB free throughout), and nothing supervises that binary — it is
not a systemd unit, so once it exits nothing restarts it.

So the observable history is: **graph present → brief live frames at ~7.8 Hz →
collapse to zero, sensor never re-enumerating.** A graph that produces frames
without an enumerated sensor is unusual and suggests either a warm-up window that
fails to complete, or the publisher was replaying something transient.

### QoS is required to see anything at all

The camera publishers offer a non-standard durability that plain default
subscribers are rejected by:

```
New publisher discovered on topic '/boostercamera/head/rgb',
offering incompatible QoS. No messages will be received from it.
Last incompatible policy: DURABILITY
```

Matching **`RELIABILITY_RELIABLE` + `DURABILITY_VOLATILE` + `KEEP_LAST(10)`** is
what made the topics visible. Without this, a correct-looking `ros2 topic info`
(publisher count 1) sits next to a subscriber that silently receives nothing.
The recorder therefore ships explicit `--qos-profile-overrides-path`.

---

## 3. Hardware enumeration — no sensor present

### `lsusb` (9 devices, all accounted for)

| ID | Device |
|---|---|
| `05e3:0626`, `05e3:0610` ×3 | Genesys USB3.1 / USB2 hubs |
| `0bda:8153` | Realtek RTL8153 USB **Ethernet** |
| `1d6b:0002`, `1d6b:0003` | Linux root hubs |
| `8087:0032` | Intel AX210 — **Bluetooth radio only** |
| `0d8c:0012` | C-Media USB **audio** |
| `1d6b:a4a7` | UAC speaker |

**No Intel RealSense** (`8086:0b3a` / `0b3b` / `0b5c`) and no USB camera of any
class. The AX210 is a common false positive — it is Bluetooth, not a camera.

### `lspci` (10 devices)

`10de:229e` Orin SoC · `8086:2725` AX210 Wi-Fi · `1d97:1202` Longsys NVMe ·
`10ec:8168` Realtek GbE ×2. Filtering for camera/image/media/video: **nothing** —
no PCI class 0x04 multimedia controller.

### `v4l2` — the decisive check

```
v4l2-ctl                 NOT installed
/dev/video*              no nodes
/sys/class/video4linux/  EXISTS but is EMPTY (total 0)
```

The class directory existing proves the V4L2 subsystem is compiled into this
kernel; it being **empty** proves no video device ever registered. A working MIPI
camera would show `video0`/`video1` there.

### MIPI / CSI

- **No `/dev/nvargus*`** — the Jetson camera stack is absent
- device-tree has `camera-ivc-channels` and `tegra-capture-vi` (host-side VI
  nodes, present on any Orin) but **no sensor node**
- dmesg contains **no** `tegra-capture` / `nvcamera` / sensor-driver lines — only
  a systemd notice about `nvargus-daemon.service`

**Conclusion: the Orin's capture hardware and framework exist, but no sensor is
bound to them.**

---

## 4. What the ROS graph contains (and why it misleads)

135 topics, 13 nodes. The camera stack:

```
/StereoNetNode, /mipi_cam, /realsense_subscriber, /rviz, /robot_state_publisher
```

Every camera topic reports **`Publisher count: 1`**, which reads as healthy. The
real pipeline is:

```
MIPI sensor  →  mipi_cam  →  StereoNetNode  →  /boostercamera/head/{rgb, right/rgb, depth}
```

- `mipi_cam` has **no binary anywhere on disk** and is not in the process table
- `StereoNetNode` is not in the process table either
- `booster-daemon-perception.service` **is** `active` — so the perception daemon
  runs while the sensor half does not

`CameraInfo` is populated and static: 544×448, `distortion_model: plumb_bob`,
`d = [0,0,0,0,0]`, `r = identity`, `k = [207.358, 0, 237.196; 0, 207.358, 228.022]`.
That is real calibration but it is **config, not evidence of a live sensor** — I
initially misread it that way.

`booster_rpc_bridge`, `booster-video-stream`, `booster-cam-receiver` packages all
exist in `/opt/booster/BoosterRos2/install/`, but **`booster-video-stream` has no
launch files** (only `package.xml`, `cmake`, `environment`) — so the
`ros2 launch booster_video_stream video_stream.launch.py` command in the
`robot-discover` skill cannot work and should be corrected there.

`booster-video-stream` self-describes as `start realsense receiver` and logs
`Publishing: '0'` indefinitely. It is a **RealSense** subscriber, not the producer
of `/boostercamera/head/*`.

---

## 5. Installed on the robot

### cuVSLAM — working

```
/home/booster/cuvslam/bin/libcuvslam.so
  ELF 64-bit LSB shared object, ARM aarch64
  unresolved libs: 0
  libcudart.so.12, libcublas.so.12, libcusparse.so.12, libcusolver.so.11
```

From NVIDIA's official Orin build,
`cuvslam-cpp-17.0.0-orin-cuda12.6.3-ubuntu22.04.tar.gz` (114 MB). Correct binary
for this unit: **aarch64**, **sm_87** (Orin), **CUDA 12.6**. A container is not
required on the robot — unlike `dl`, whose host only has CUDA 11.7 and therefore
needs the mounted `k1-slam:cuvslam12` container.

`dl` still has the working x86 runtime (`~/k1_slam/cuvslam/bin/`, all CUDA 12 deps
resolve, 4× RTX 6000 Ada available).

### cuVSLAM input requirements (from `cuvslam2.h`)

```cpp
enum class OdometryMode : uint8_t {
  Multicamera,  ///< multiple synchronized STEREO cameras, frustum overlap
  Inertial,     ///< STEREO camera and IMU
  RGBD,         ///< RGB-D; RGB & Depth must be ALIGNED
  Mono,         ///< single camera, accurate UP TO SCALE
};
struct Camera { IntArray<2> size; Array<2> principal, focal;
                Pose rig_from_camera; Distortion distortion; ... };
```

cuVSLAM is **stereo-first**. A single RGB topic cannot drive it, and `Mono` gives
scale only up to an unknown factor. The robot's rectified pair plus its
`CameraInfo` is exactly the right input — **if frames ever arrive**.

### `boosteros` 1.2.0

Installed at `/tmp/venv_bos` on the robot, created with
`--system-site-packages` so rclpy / tf2_ros / sensor_msgs resolve from the ROS
install. Exposes `list_sensors`, `subscribe_image`, `subscribe_imu`,
`subscribe_odom`, `get_state`.

> `/tmp` does not survive reboot. Move it to `~/` if it is to persist.

`boosteros` could not reach the robot's services reliably (`LocoClientInitError`,
`req_matched=0`) — it connected exactly once, on the **WiFi** address.

### Watchdog — installed and running

`/home/booster/watchdog_k1_stereo.py`, service **`k1-stereo-watchdog.service`**
(enabled, `Restart=always`, `MemoryMax=300M`, `Nice=10` so it cannot starve the
balance controller). Checks the pair every 30 s.

Design point: if **no camera device is present** it reports `UNRECOVERABLE` with a
full diagnostic snapshot instead of restarting forever. A watchdog that silently
flaps is worse than one that says the hardware is gone.

```
[07:49:19] watchdog up (interval=30s, min 0.5 Hz)
[07:49:28] DOWN #1: left=0.0 Hz right=0.0 Hz (threshold 0.5)
[07:49:29] UNRECOVERABLE: no camera device on this robot (/dev/v4l/by-path empty)
```

Install: `bash install_watchdog.sh` · report: `--status`.

### Bug I introduced and fixed

The watchdog's first version used `pgrep -a <name>` without `-f`, so it matched
only `comm` (truncated to 15 chars) and falsely reported
`booster-daemon-perception: NOT RUNNING` when the service was `active`. Fixed to
`pgrep -af`. The false negative was mine, not the robot's.

---

## 6. Corrections I made during this investigation

Recording these because they cost time and shaped wrong conclusions:

1. **"There is no camera on A2"** — wrong, and I said it twice. There is a full
   135-topic camera graph, and it briefly carried real frames at 7.8 Hz.
2. **"The stereo pair is live"** — wrong. I trusted `Publisher count: 1`, which
   only proves a node *advertised* a topic. Frames were 0 Hz, and the
   `CameraInfo` I quoted as evidence was static config.
3. **"cuVSLAM is not built"** — wrong, and this one was entirely my own error. I
   ran the link check without the `-v ~/k1_slam/cuvslam:/opt/cuvslam:ro` volume
   mount, so `/opt/cuvslam` was empty inside the container. With the mount it
   passes cleanly.
4. **"A restart that returned HTTP 200 is a restart"** — wrong. PID 100587 kept
   owning :7860 with stale buggy code; `pkill -f` had never matched it. Fixed by
   `scripts/restart_k1m_app.sh`, which verifies the new PID owns the port.
5. **Method error worth repeating:** I spent several rounds on QoS and process
   state before running `lsusb` / `lspci` / `/sys/class/video4linux`, which
   answered the sensor question immediately. Hardware enumeration first next time.

---

## 7. Next steps

1. **Physical check on A2** — is a head camera physically fitted? Reseat the MIPI
   ribbon. The watchdog will pick the pair up automatically once a sensor
   enumerates; watch with `journalctl -fu k1-stereo-watchdog`.
2. **Check the other units** — A1/A3/B1–B3 were off-network; one may have a
   camera fitted. Discovery:
   `nmap -sn -n <subnet> | grep report | awk '{print $5}'`
3. **If A2 should carry a RealSense D455** — `booster-video-stream` is waiting for
   exactly that, and a D455 feeds cuVSLAM's `Inertial` mode with factory
   calibration.
4. **Once frames exist** — record the pair (recorder logic is written, needs a
   live run to validate), then feed cuVSLAM on-board.

### Rosbag recorder

`/tmp/rec_k1_stereo.sh` on the robot, with a preflight that refuses to record an
empty bag, and explicit QoS overrides. It has **not yet completed a successful
recording** — the pair is down, so the bag would be empty by design.

---

## 8. Standing caveat for the walk

The robot was rebooted for low battery, and the perception stack is currently
degraded. Check charge and confirm `booster-daemon-perception` is healthy before
taking A2 out. Unrelated but relevant: `docs/k1_ros_deployment_plan.md` lists
unresolved blockers for hardware operation (no Custom-mode switch, no safety
layer, gain mismatches), and the feasibility gate reports generated motions
demanding up to **9.58×** a joint's effort limit — that gate result is about
*generated motion*, not about driving the trained locomotion policy.
