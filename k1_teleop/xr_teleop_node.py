#!/usr/bin/env python3
"""Quest 3 -> Booster K1 teleop through Isaac Teleop (CloudXR), one process.
Ported from G1_sim ``teleop/xr_teleop_node.py`` (9c3c429).

* Televiz (``isaacteleop.viz``) owns the OpenXR session and shows, fixed in the
  room (placed on X): the K1's **ZED 2i pair as one stereo quad** (left image
  -> left eye, right -> right eye, same sim stamp), the chase camera on the
  left, the YOLO view of the left eye on the right, and controls / robot pose /
  your input below;
* a TeleopSession on the same session reads head + controllers:
  thumbsticks -> ``base_twist`` -> ``/{ns}/cmd_vel`` (fixed-base phase: the sim
  moves the suspended root; linear.z = up/down), headset -> ``AAHead_yaw`` /
  ``Head_pitch``;
* arms, ``--arm-src``:
    ``ik``   controller grip positions -> clutch -> K1 4-DoF position IK;
    ``soma`` HMD-Poser SMPL body -> ``k1_teleop/soma_retarget_node.py`` publishes
             the arm joints itself (this node then sends head joints only).
  Both land on ``/{ns}/teleop/upper_body_cmd`` (JointState, hold-last per joint).

Buttons: A (right) arm follow on/off, B (right) pause (zero velocity, hold),
X (left) re-centre panels + "straight ahead" for the head, Y (left) head
follow on/off, left-stick click = T-pose calibration (arm scale, HMD-Poser).

    python3 -m k1_teleop.xr_teleop_node --accept-eula                 # headset + sim
    python3 -m k1_teleop.xr_teleop_node --accept-eula --arm-src soma  # + soma_retarget_node
    python3 -m k1_teleop.xr_teleop_node --display window --video sbs.mp4   # desktop, no headset
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, Twist
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, String

from k1_teleop.teleop_mapping import (
    OP_SHOULDER, ArmClutch, Edge, arm_scale_from_tpose, base_twist, draw_text, head_targets, quat_xyzw_to_mat,
    render_pose, rot_z, xr_pose_to_robot, yaw_of,
)

CMD_HZ = 30.0          # cmd_vel / upper-body rate
INFO_HZ = 10.0         # rendered info panels
PANEL_DIST = 1.5       # m from the head
POSE_PX = 512
CONTROLS_PX = (640, 400)


class FrameBox:
    """Latest frame (or stereo pair) for one panel, handed from ROS to the XR loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None
        self.version = 0

    def put(self, frame) -> None:
        with self._lock:
            self._frame, self.version = frame, self.version + 1

    def get(self):
        with self._lock:
            return self.version, self._frame


def rgba(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate((img, np.full(img.shape[:2] + (1,), 255, np.uint8)), axis=2))


def video_reader(path: str, box: FrameBox, eye: tuple[int, int], stereo: bool) -> None:
    """Loop a video into ``box``: side-by-side (width 2x height*aspect) -> stereo pair,
    otherwise the same frame for both eyes."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        halves = (rgb[:, : w // 2], rgb[:, w // 2:]) if w >= 2 * h * eye[0] / eye[1] * 0.9 else (rgb, rgb)
        l, r = (rgba(cv2.resize(x, eye, interpolation=cv2.INTER_AREA)) for x in halves)
        box.put((l, r) if stereo else l)
        time.sleep(1.0 / fps)


def _rot_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


class TeleopBridge(Node):
    """ROS side: images / joint states in, cmd_vel / upper-body joints out, plus the per-step mapping."""

    def __init__(self, ns: str, boxes: dict[str, FrameBox], image_topics: dict[str, str], ik,
                 arm_scale: float, poser, arm_src: str, zed_topics: tuple[str, str] | None) -> None:
        super().__init__("k1_xr_teleop")
        self.ns, self.boxes, self.ik, self.poser, self.arm_src = ns, boxes, ik, poser, arm_src
        reliable = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                              reliability=QoSReliabilityPolicy.RELIABLE)   # ~1.5 MB images vs 212 KB rmem_max
        for name, topic in image_topics.items():
            self.create_subscription(Image, topic, lambda m, b=boxes[name]: self._on_image(m, b), reliable)
        self._eyes: dict[str, tuple] = {}                                  # side -> (stamp, rgba)
        if zed_topics:
            for side, topic in zip(("left", "right"), zed_topics):
                self.create_subscription(Image, topic, lambda m, s=side: self._on_eye(m, s), reliable)
        self.create_subscription(JointState, f"/{ns}/joint_states", self._on_joints, 10)
        self.pub_vel = self.create_publisher(Twist, f"/{ns}/cmd_vel", 10)
        self.pub_upper = self.create_publisher(JointState, f"/{ns}/teleop/upper_body_cmd", 10)
        self.pub_goal = self.create_publisher(PoseArray, f"/{ns}/teleop/ik_goals", 10)
        self.pub_status = self.create_publisher(String, f"/{ns}/teleop/status", 10)
        self.pub_smpl = self.create_publisher(Float32MultiArray, f"/{ns}/teleop/smpl_pose", 10)

        self.joints: dict[str, float] = {}
        self.paused = False
        self.arms_on = False
        self.head_on = True
        self.heading = 0.0                                  # operator yaw = robot "straight ahead"
        self.clutch = {s: ArmClutch(scale=arm_scale, absolute=True, op_shoulder=OP_SHOULDER[s],
                                    reach=ik.arm_length(s) + 0.02) for s in ("left", "right")}
        self.arm_scale_calibrated = False
        self.q: dict[str, np.ndarray] = {}
        self.ik_err = {"left": None, "right": None}
        self.btn = {k: Edge() for k in ("A", "B", "X", "Y", "LS")}
        self.op: dict[str, np.ndarray | None] = {"head": None, "left": None, "right": None}
        self.goals: dict[str, np.ndarray] = {}
        self.head_q: dict[str, float] = {}
        self.cmd = (0.0, 0.0, 0.0, 0.0)
        self._last_cmd_t = 0.0
        self._last_status_t = 0.0
        self.recenter_request = True
        self.stereo_pairs = 0

    # ── ROS callbacks ──
    @staticmethod
    def _decode(msg: Image) -> np.ndarray | None:
        ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
        if ch is None:
            return None
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // ch, ch)[:, :msg.width]
        if msg.encoding.startswith("bgr"):
            img = img[..., [2, 1, 0] + ([3] if ch == 4 else [])]
        return rgba(img) if ch == 3 else np.ascontiguousarray(img)

    def _on_image(self, msg: Image, box: FrameBox) -> None:
        img = self._decode(msg)
        if img is not None:
            box.put(img)

    def _on_eye(self, msg: Image, side: str) -> None:
        """Pair left/right by header stamp: a pair from two different sim steps would
        tear the stereo depth."""
        img = self._decode(msg)
        if img is None:
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._eyes[side] = (stamp, img)
        other = self._eyes.get("right" if side == "left" else "left")
        if other is not None and other[0] == stamp:
            self.boxes["zed"].put((self._eyes["left"][1], self._eyes["right"][1]))
            self.stereo_pairs += 1
            self._eyes.clear()

    def _on_joints(self, msg: JointState) -> None:
        self.joints.update(zip(msg.name, (float(p) for p in msg.position)))

    # ── per-step mapping ──
    def step(self, out: dict, now: float, dt: float) -> None:
        from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex as CI, HeadPoseIndex as HI

        left, right, head = out["controller_left"], out["controller_right"], out["head"]
        if not right.is_none and self.btn["B"](float(right[CI.SECONDARY_CLICK])):
            self.paused = not self.paused
            self.get_logger().info("PAUSED" if self.paused else "resumed")
        if not left.is_none and self.btn["X"](float(left[CI.PRIMARY_CLICK])):
            self.recenter_request = True
        if not left.is_none and self.btn["Y"](float(left[CI.SECONDARY_CLICK])):
            self.head_on = not self.head_on
        toggle_arms = not right.is_none and self.btn["A"](float(right[CI.PRIMARY_CLICK]))

        head_yaw = 0.0
        if not head.is_none and bool(head[HI.IS_VALID]):
            head_T = xr_pose_to_robot(head[HI.POSITION], head[HI.ORIENTATION])
            head_yaw = yaw_of(head_T[:3, :3])
            self.op["head"] = head_T
            if self.recenter_request:
                self.heading = head_yaw
        for side, ctrl in (("left", left), ("right", right)):
            self.op[side] = self._grip(ctrl)
        if not left.is_none and self.btn["LS"](float(left[CI.THUMBSTICK_CLICK])):
            self._calibrate_tpose()
        if self.poser is not None:
            self.poser.push(now, self.op["head"], self.op["left"], self.op["right"])
        if toggle_arms and not self.paused:
            self._toggle_arms(left, right, head_yaw)

        if now - self._last_cmd_t < 1.0 / CMD_HZ:
            return
        cmd_dt, self._last_cmd_t = now - self._last_cmd_t, now

        sticks = [0.0, 0.0, 0.0, 0.0]
        if not left.is_none:
            sticks[0:2] = float(left[CI.THUMBSTICK_X]), float(left[CI.THUMBSTICK_Y])
        if not right.is_none:
            sticks[2:4] = float(right[CI.THUMBSTICK_X]), float(right[CI.THUMBSTICK_Y])
        self.cmd = (0.0, 0.0, 0.0, 0.0) if self.paused else base_twist(*sticks)
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z, tw.angular.z = self.cmd
        self.pub_vel.publish(tw)
        if self.paused:
            self._status(now)
            return

        names, pos = [], []
        if self.head_on and self.op["head"] is not None:
            self.head_q = head_targets(self.op["head"], self.heading)
            names += list(self.head_q)
            pos += list(self.head_q.values())
        if self.arms_on and self.arm_src == "ik":
            n, p = self._arms(min(cmd_dt, 0.1))
            names += n
            pos += p
        if names:
            msg = JointState(name=names, position=pos)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_upper.publish(msg)
        self._status(now)

    def _calibrate_tpose(self) -> None:
        if any(self.op[k] is None for k in ("head", "left", "right")):
            self.get_logger().warn("T-pose calibration needs head + both controllers tracked")
            return
        scale = arm_scale_from_tpose(self.op["head"], {s: self.op[s] for s in ("left", "right")},
                                     self.ik.arm_length("left"))
        if 0.3 < scale < 1.5:
            for c in self.clutch.values():
                c.scale = scale
            self.arm_scale_calibrated = True
        self.get_logger().info(f"T-pose calibration: arm scale {scale:.2f}"
                               + ("" if self.arm_scale_calibrated else " (implausible, kept default)"))
        if self.poser is not None:
            self.poser.calibrate(self.op["head"], self.op["left"], self.op["right"])

    def _grip(self, ctrl):
        from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex as CI

        if ctrl.is_none or not bool(ctrl[CI.GRIP_IS_VALID]):
            return None
        return xr_pose_to_robot(ctrl[CI.GRIP_POSITION], ctrl[CI.GRIP_ORIENTATION])

    def _toggle_arms(self, left, right, head_yaw: float) -> None:
        if self.arms_on:
            self.arms_on = False
            for c in self.clutch.values():
                c.release()
            self.get_logger().info("arms: hold")
            return
        if not self.joints:
            self.get_logger().warn(f"arms: no /{self.ns}/joint_states yet")
            return
        if self.arm_src == "soma":                 # soma_retarget_node reads arms_on from /teleop/status
            self.arms_on = True
            self.get_logger().info("arms: follow (soma)")
            return
        for side, ctrl in (("left", left), ("right", right)):
            self.q[side] = np.array([self.joints.get(n, 0.0) for n in self.ik.names[side]])
            pose = self._grip(ctrl)
            if pose is None:
                continue
            self.clutch[side].shoulder = self.ik.shoulder_pos(side, self.joints)
            self.clutch[side].engage(pose, head_yaw, self.ik.fk(side, self.joints))
        self.arms_on = any(c.engaged for c in self.clutch.values())
        self.get_logger().info(f"arms: follow {[s for s, c in self.clutch.items() if c.engaged]}")

    def _arms(self, dt: float) -> tuple[list[str], list[float]]:
        """Clutch goals -> IK; always returns all 8 arm joints (hold where not tracked)."""
        names, pos, goals = [], [], PoseArray()
        goals.header.stamp = self.get_clock().now().to_msg()
        goals.header.frame_id = "Trunk"
        for side in ("left", "right"):
            pose = self.op[side]
            if pose is not None and self.clutch[side].engaged:
                head = self.op["head"]
                goal = self.clutch[side].goal(pose, dt, head_pos=None if head is None else head[:3, 3])
                self.goals[side] = goal
                state = dict(self.joints)
                state.update(zip(self.ik.names[side], self.q[side]))
                self.q[side], self.ik_err[side] = self.ik.solve(side, goal, state)
            names += self.ik.names[side]
            pos += [float(v) for v in self.q[side]]
            g = self.goals.get(side)
            if g is not None:
                p = Pose()
                p.position.x, p.position.y, p.position.z = (float(v) for v in g[:3, 3])
                (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w) = _rot_to_quat_xyzw(g[:3, :3])
                goals.poses.append(p)
        if len(goals.poses) == 2:                  # sim pairs poses with (left, right) by order
            self.pub_goal.publish(goals)
        return names, pos

    # ── info panels ──
    def pose_image(self) -> np.ndarray:
        poses = self.ik.link_poses(self.joints)
        frames = [(poses[f"{s}_hand_tip"], 0.05) for s in ("left", "right")]
        if self.arms_on:
            frames += [(g, 0.10) for g in self.goals.values()]
        img = render_pose(self.ik.skeleton(poses), frames, (), POSE_PX)
        err = " ".join(f"{s[0].upper()} {e * 100:.1f}cm" for s, e in self.ik_err.items() if e is not None)
        return draw_text(img, [("ROBOT POSE (trunk frame)", (235, 235, 235)),
                               ("small axes: hand tips   big axes: IK goals", (200, 200, 200)),
                               (f"arm follow {'ON' if self.arms_on else 'off'} ({self.arm_src})   IK err {err or '-'}",
                                (255, 220, 0) if self.arms_on else (200, 200, 200))])

    def operator_image(self) -> np.ndarray:
        head, frames, links, lines = self.op["head"], [], [], []
        if head is not None:
            Rh = rot_z(-yaw_of(head[:3, :3]))
            anchor = self.ik.link_poses(self.joints)["Head_2"][:3, 3]
            H = np.eye(4)
            H[:3, :3], H[:3, 3] = Rh @ head[:3, :3], anchor
            frames.append((H, 0.1))
            for side, rgb in (("left", (0, 220, 220)), ("right", (255, 80, 255))):
                c = self.op[side]
                if c is None:
                    lines.append((f"{side:5s} controller: no tracking", (255, 80, 80)))
                    continue
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = Rh @ c[:3, :3], anchor + Rh @ (c[:3, 3] - head[:3, 3])
                frames.append((T, 0.08))
                links.append((anchor, T[:3, 3], rgb))
                d = T[:3, 3] - anchor
                lines.append((f"{side:5s} fwd {d[0]:+.2f} left {d[1]:+.2f} up {d[2]:+.2f} m", rgb))
        else:
            lines.append(("no head tracking", (255, 80, 80)))
        img = render_pose(np.zeros((0, 2, 3)), frames, links, POSE_PX)
        return draw_text(img, [("YOUR INPUT (head + controllers)", (235, 235, 235))] + lines)

    def controls_image(self) -> np.ndarray:
        w, h = CONTROLS_PX
        img = np.full((h, w, 4), (18, 18, 24, 255), np.uint8)
        state = ("PAUSED - B to resume", (255, 80, 80)) if self.paused else ("RUNNING", (80, 255, 120))
        vx, vy, vz, wz = self.cmd
        hq = self.head_q
        frames = "  ".join(f"{k} {b.version}" for k, b in self.boxes.items() if k in ("zed", "chase", "yolo"))
        return draw_text(img, [
            ("CONTROLS  (K1 suspended, fixed base)", (235, 235, 235)),
            ("Left stick   move base fwd/back, strafe", (200, 200, 200)),
            ("Right stick  X turn, Y up/down", (200, 200, 200)),
            (f"A  arm follow: {'ON' if self.arms_on else 'off'}  ({self.arm_src}, scale "
             f"{self.clutch['left'].scale:.2f}{'' if self.arm_scale_calibrated else ' default'})",
             (255, 220, 0) if self.arms_on else (200, 200, 200)),
            ("B  pause / resume", (200, 200, 200)),
            ("X  re-centre panels + head straight ahead", (200, 200, 200)),
            (f"Y  head follow: {'ON' if self.head_on else 'off'}", (200, 200, 200)),
            ("L-stick click  T-pose calibration", (200, 200, 200)),
            ("", (0, 0, 0)),
            state,
            (f"base  vx {vx:+.2f} vy {vy:+.2f} vz {vz:+.2f} wz {wz:+.2f}", (235, 235, 235)),
            (f"head  yaw {hq.get('AAHead_yaw', 0.0):+.2f} pitch {hq.get('Head_pitch', 0.0):+.2f}", (235, 235, 235)),
            (f"frames {frames}", (150, 150, 150)),
        ], dy=24)

    def publish_smpl(self) -> None:
        r = self.poser.result if self.poser else None
        if r and "smpl" in r:
            self.pub_smpl.publish(Float32MultiArray(data=[float(v) for v in r["smpl"].reshape(-1)]))

    def _status(self, now: float) -> None:
        if now - self._last_status_t < 0.5:
            return
        self._last_status_t = now
        self.pub_status.publish(String(data=json.dumps({
            "paused": self.paused, "arms": self.arms_on, "arm_source": self.arm_src, "head": self.head_on,
            "cmd_vel": [round(v, 3) for v in self.cmd],
            "ik_err_m": {s: None if e is None else round(e, 4) for s, e in self.ik_err.items()},
            "frames": {k: b.version for k, b in self.boxes.items()}, "stereo_pairs": self.stereo_pairs,
        })))


def build_pipeline():
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HeadSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner

    controllers, head = ControllersSource(name="controllers"), HeadSource(name="head")
    return OutputCombiner({"controller_left": controllers.output(ControllersSource.LEFT),
                           "controller_right": controllers.output(ControllersSource.RIGHT),
                           "head": head.output("head")})


def panel_placement(viz, position, orientation_xyzw, size, angle_deg: float, dy: float = 0.0):
    """Level panel PANEL_DIST from the head, ``angle_deg`` left of where you look (yaw only),
    centred ``dy`` m above eye height minus 0.1 m, facing you."""
    pos = np.asarray(position, dtype=np.float64)
    fwd = quat_xyzw_to_mat(orientation_xyzw) @ np.array([0.0, 0.0, -1.0])
    yaw = math.atan2(-fwd[0], -fwd[2]) + math.radians(angle_deg)   # about +y (OpenXR up)
    center = pos + PANEL_DIST * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)]) + np.array([0.0, dy - 0.1, 0.0])
    return viz.QuadLayerPlacement(
        viz.Pose3D(tuple(center), (math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)), size)  # wxyz


def main() -> int:
    from isaacteleop.cloudxr import CloudXRLauncher

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ns", default="k1_0")
    ap.add_argument("--arm-src", choices=("ik", "soma"), default="ik",
                    help="ik: controller positions -> K1 arm IK; soma: HMD-Poser -> soma_retarget_node (A still toggles)")
    ap.add_argument("--display", choices=("xr", "window"), default="xr")
    ap.add_argument("--video", help="loop this video on the ZED panel instead of the sim (side-by-side = stereo)")
    ap.add_argument("--eye-size", default="960x540", help="per-eye stream size (must match the sim)")
    ap.add_argument("--chase-size", default="640x360")
    ap.add_argument("--yolo-size", default="960x540", help="'' disables the YOLO panel")
    ap.add_argument("--hmd-poser", default="auto", choices=("auto", "on", "off"),
                    help="HMD-Poser body estimate (auto: on for --arm-src soma if the weights exist)")
    ap.add_argument("--arm-scale", type=float, default=0.65,
                    help="K1 arm / your arm until a T-pose calibration measures it (K1 0.37 m / ~0.56 m)")
    CloudXRLauncher.add_launcher_arguments(ap)
    args = ap.parse_args()

    import cupy as cp
    import isaacteleop.viz as viz
    from isaacteleop.oxr import OpenXRSessionHandles
    from isaacteleop.retargeting_engine.tensor_types import HeadPoseIndex as HI
    from isaacteleop.teleop_session_manager import (
        TeleopSession, TeleopSessionConfig, get_required_oxr_extensions_from_pipeline,
    )

    from k1_teleop.k1_arm_ik import ArmIK

    ns, xr = args.robot_ns, args.display == "xr"
    size = lambda s: tuple(int(v) for v in s.split("x"))   # noqa: E731
    ew, eh = size(args.eye_size)
    # name -> (pixels WxH, width m, degrees left of gaze, ROS image topic, metres up)
    panels = {"zed": ((ew, eh), 1.6, 0.0, None, 0.0),
              "chase": (size(args.chase_size), 1.0, 48.0, f"/{ns}/chase/image_raw", 0.0),
              "controls": (CONTROLS_PX, 0.9, 0.0, None, -1.0),
              "pose": ((POSE_PX, POSE_PX), 0.7, -38.0, None, -1.0),
              "operator": ((POSE_PX, POSE_PX), 0.7, 38.0, None, -1.0)}
    if args.yolo_size:
        panels["yolo"] = (size(args.yolo_size), 1.0, -48.0, f"/{ns}/yolo/image_raw", 0.0)
    boxes = {name: FrameBox() for name in panels}
    if args.video:
        threading.Thread(target=video_reader, args=(args.video, boxes["zed"], (ew, eh), True), daemon=True).start()

    rclpy.init()
    poser = None
    from k1_teleop.hmd_poser.live import JOINTS, WEIGHTS

    if args.hmd_poser == "on" or (args.hmd_poser == "auto" and args.arm_src == "soma"
                                  and WEIGHTS.exists() and JOINTS.exists()):
        from k1_teleop.hmd_poser.live import HmdPoserLive

        poser = HmdPoserLive()
        poser.start()
    elif args.arm_src == "soma":
        print("WARNING: --arm-src soma without HMD-Poser: nothing publishes the SMPL pose")
    ik = ArmIK()
    bridge = TeleopBridge(ns, boxes, {n: p[3] for n, p in panels.items() if p[3]}, ik, args.arm_scale, poser,
                          args.arm_src, None if args.video else (f"/{ns}/zed/left/image_raw", f"/{ns}/zed/right/image_raw"))
    if poser is not None:
        poser.on_result = bridge.publish_smpl
    pipeline = build_pipeline() if xr else None

    launch = CloudXRLauncher.launch_context(args, host_client=True) if xr else None
    launcher = launch.__enter__() if launch else None
    try:
        cfg = viz.VizSessionConfig()
        cfg.mode = viz.DisplayMode.kXr if xr else viz.DisplayMode.kWindow
        if xr:
            cfg.required_extensions = get_required_oxr_extensions_from_pipeline(pipeline)
            cfg.xr_system_wait_seconds = 3600          # wait for the Quest to connect
        else:
            cfg.window_width, cfg.window_height = 1600, 900
        vs = viz.VizSession.create(cfg)
        layers, sizes = {}, {}
        for name, ((w, h), width_m, angle, _src, dy) in panels.items():
            sizes[name] = (width_m, width_m * h / w)
            lcfg = viz.QuadLayerConfig()
            lcfg.name, lcfg.resolution = name, viz.Resolution(w, h)
            lcfg.stereo = name == "zed" and xr               # window mode shows the left eye
            if xr:
                lcfg.placement = panel_placement(viz, (0.0, 1.5, 0.0), (0.0, 0.0, 0.0, 1.0), sizes[name], angle, dy)
            layers[name] = vs.add_quad_layer(lcfg)
        bridge.get_logger().info(f"Televiz up ({args.display}), arm source {args.arm_src}, panels "
                                 f"{ {n: p[3] or ('stereo ZED' if n == 'zed' else 'rendered') for n, p in panels.items()} }")

        shown = {name: 0 for name in panels}
        session_cm = (TeleopSession(TeleopSessionConfig(
            app_name="K1XrTeleop", pipeline=pipeline,
            oxr_handles=OpenXRSessionHandles(*vs.get_oxr_handles()))) if xr else None)
        session = session_cm.__enter__() if session_cm else None
        try:
            t_prev = t_info = time.monotonic()
            while rclpy.ok() and not vs.should_close():
                try:
                    rclpy.spin_once(bridge, timeout_sec=0.0)
                except KeyboardInterrupt:             # Ctrl-C: leave cleanly (zero cmd_vel below)
                    break
                now = time.monotonic()
                if session is not None:
                    if launcher is not None:
                        launcher.health_check()
                    out = session.step()
                    bridge.step(out, now, now - t_prev)
                    head = out["head"]
                    if bridge.recenter_request and not head.is_none and bool(head[HI.IS_VALID]):
                        for name, layer in layers.items():
                            layer.set_placement(panel_placement(viz, head[HI.POSITION], head[HI.ORIENTATION],
                                                                sizes[name], panels[name][2], panels[name][4]))
                        bridge.recenter_request = False
                t_prev = now
                if now - t_info >= 1.0 / INFO_HZ:
                    t_info = now
                    boxes["pose"].put(bridge.pose_image())
                    boxes["operator"].put(bridge.operator_image())
                    boxes["controls"].put(bridge.controls_image())
                for name, layer in layers.items():
                    version, frame = boxes[name].get()
                    if frame is None or version == shown[name]:
                        continue
                    shown[name] = version
                    w, h = panels[name][0]
                    pair = frame if isinstance(frame, tuple) else (frame,)
                    if all(f.shape[:2] == (h, w) for f in pair):
                        if name == "zed" and xr:
                            layer.submit(cp.asarray(pair[0]), cp.asarray(pair[-1]))
                        else:
                            layer.submit(cp.asarray(pair[0]))
                    elif version == 1:
                        bridge.get_logger().error(f"{name}: stream is {pair[0].shape[1]}x{pair[0].shape[0]}, "
                                                  f"panel is {w}x{h}")
                vs.render()                             # paced by xrWaitFrame in XR
        finally:
            print(f"frames shown per panel {shown}, stereo pairs {bridge.stereo_pairs}", flush=True)
            if rclpy.ok():
                bridge.pub_vel.publish(Twist())         # never leave the base moving
            if session_cm:
                session_cm.__exit__(None, None, None)
            vs.destroy()
    finally:
        if launch:
            launch.__exit__(None, None, None)
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
