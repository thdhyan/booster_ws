#!/usr/bin/env python3
"""Grab K1 head-camera intrinsics once and save as YAML.

Why a file and not a bag topic: /boostercamera/head/*/camera_info is a ONE-SHOT
publisher. It fires a couple of messages when something subscribes and then goes
dormant, so whether rosbag2 catches it is luck -- it managed 2 messages in one
test run and 0 in the next. Intrinsics are static, so one capture is all that is
needed; capturing them explicitly is the only reliable way to keep them.

cuVSLAM refuses to initialise without calibration, and nvblox silently produces
a warped reconstruction without it, so this file is not optional.

Writes: <out>/head_cam_intrinsics.yaml
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo


def grab(node, topic, want, timeout_s=12.0):
    """Collect up to `want` CameraInfo messages on `topic`."""
    got = []
    # Durability MUST match the publisher. These publishers offer VOLATILE, and
    # a TRANSIENT_LOCAL subscriber is rejected as incompatible, so asking for
    # "more" durability gets you nothing at all. The one-shot publisher fires as
    # soon as a compatible subscriber appears, which is what we rely on here.
    qos = QoSProfile(
        depth=10,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    def cb(msg):
        if len(got) < want:
            got.append(msg)

    sub = node.create_subscription(CameraInfo, topic, cb, qos)
    deadline = time.time() + timeout_s
    while len(got) < want and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.25)
    node.destroy_subscription(sub)
    return got


def to_yaml(msg, name):
    d = msg.width, msg.height
    return f"""\
# K1 head camera intrinsics, captured {time.strftime('%Y-%m-%d %H:%M:%S')}
# topic: {name}
# One-shot publisher: captured once, stored here so SLAM has calibration even
# though the bag may not contain it.
image_width: {msg.width}
image_height: {msg.height}
distortion_model: {msg.distortion_model}
d: [{', '.join(f'{v:.9g}' for v in msg.d)}]
k: [{', '.join(f'{v:.9g}' for v in msg.k)}]
r: [{', '.join(f'{v:.9g}' for v in msg.r)}]
p: [{', '.join(f'{v:.9g}' for v in msg.p)}]
binning_x: {msg.binning_x}
binning_y: {msg.binning_y}
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    rclpy.init()
    node = Node("k1_intrinsics_grab")
    # The /head/rgb and /head/depth camera_info topics are one-shot and go
    # dormant. The /head/raw/... ones are published by mipi_cam at a steady
    # 30 Hz, so they actually deliver. Verified on A2: raw/camera_info streams,
    # the others do not.
    sources = [
        ("/boostercamera/head/raw/rgb/camera_info", "head_stereo_left"),
        ("/boostercamera/head/raw/right/rgb/camera_info", "head_stereo_right"),
    ]
    written = 0
    for topic, name in sources:
        msgs = grab(node, topic, want=1)
        if not msgs:
            print(f"[WARN] no CameraInfo on {topic}")
            continue
        path = f"{out}/head_cam_intrinsics_{name}.yaml"
        with open(path, "w") as fh:
            fh.write(to_yaml(msgs[0], topic))
        m = msgs[0]
        print(f"[ok] {topic} -> {path}  ({m.width}x{m.height}, {m.distortion_model})")
        written += 1
    node.destroy_node()
    rclpy.shutdown()
    if written == 0:
        print("[FAIL] no intrinsics captured -- SLAM will not initialise")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
