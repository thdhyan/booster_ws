"""Pure mapping from Quest 3 state to K1 commands (no ROS, no isaacteleop), so
it is unit-testable on the laptop. Frames, ``Edge``, ``ArmClutch`` and the
panel drawing are copied from G1_sim ``teleop/teleop_mapping.py`` (9c3c429).

Frames
------
OpenXR stage space (what Isaac Teleop reports): x right, y up, z backward
(the user looks down -z), quaternions XYZW. Robot / ROS: x forward, y left,
z up. ``XR_TO_ROBOT`` maps one onto the other.

Base (fixed-base phase)
-----------------------
The K1 hangs in the air on a fixed root joint; the thumbsticks move that root
(``base_twist``): left stick = forward/strafe, right stick X = turn, right
stick Y = up/down. Same stick convention as Isaac Teleop's
``LocomotionRootCmdRetargeter``, so the walking policy can take over later.

Head
----
``head_targets``: the headset's yaw relative to the operator heading (stored
at re-centre) and its pitch -> ``AAHead_yaw`` (URDF name), ``Head_pitch``, clamped.

Arms ("clutch" teleop)
----------------------
When arm follow is switched on we store, per side, the controller pose C0, the
robot's current hand pose H0 (trunk frame) and the operator's heading.
Position ``absolute`` (default): the hand relative to the operator's shoulder
(estimated from the headset, ``OP_SHOULDER``), times ``scale`` = K1 arm /
operator arm, placed at the robot's shoulder. K1 arms have no wrist, so the
IK only tracks the position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Rows: robot x/y/z expressed in XR axes. robot x = -xr z, y = -xr x, z = xr y.
XR_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# Suspended-base speed limits (m/s, rad/s) at full stick deflection.
BASE_LIMITS = {"vx": 0.5, "vy": 0.3, "vz": 0.2, "wz": 0.8}

# URDF limits (K1_22dof-ZED.urdf), rad.
HEAD_LIMITS = {"AAHead_yaw": (-1.0, 1.0), "Head_pitch": (-0.349, 0.855)}

# Operator shoulder from the headset (eyes), heading frame (x fwd, y left, z up), m.
OP_SHOULDER = {"left": np.array([-0.10, 0.18, -0.25]), "right": np.array([-0.10, -0.18, -0.25])}
GRIP_TO_WRIST = 0.08            # m, controller grip centre -> operator wrist


def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def xr_pose_to_robot(position, orientation_xyzw) -> np.ndarray:
    """4x4 pose in robot-convention axes (still the XR room origin)."""
    T = np.eye(4)
    T[:3, :3] = XR_TO_ROBOT @ quat_xyzw_to_mat(orientation_xyzw) @ XR_TO_ROBOT.T
    T[:3, 3] = XR_TO_ROBOT @ np.asarray(position, dtype=np.float64)
    return T


def yaw_of(R: np.ndarray) -> float:
    """Heading of a robot-convention rotation (angle of its x axis in the ground plane)."""
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def base_twist(lx: float, ly: float, rx: float, ry: float, deadzone: float = 0.05) -> tuple[float, float, float, float]:
    """Thumbsticks (each -1..1, up/right positive) -> (vx, vy, vz, wz) in the base frame."""
    def dz(v: float) -> float:
        v = float(v)
        return 0.0 if not math.isfinite(v) or abs(v) < deadzone else max(-1.0, min(1.0, v))
    lx, ly, rx, ry = dz(lx), dz(ly), dz(rx), dz(ry)
    return (ly * BASE_LIMITS["vx"], -lx * BASE_LIMITS["vy"], ry * BASE_LIMITS["vz"], -rx * BASE_LIMITS["wz"])


def head_targets(head: np.ndarray, heading: float) -> dict[str, float]:
    """Headset pose (robot axes) -> K1 head joints. ``heading``: operator yaw that
    means "robot looks straight ahead". Positive Head_pitch looks down."""
    f = head[:3, 0]                                             # headset forward
    yaw = wrap(math.atan2(f[1], f[0]) - heading)
    pitch = math.atan2(-f[2], math.hypot(f[0], f[1]))
    out = {}
    for name, v in (("AAHead_yaw", yaw), ("Head_pitch", pitch)):
        lo, hi = HEAD_LIMITS[name]
        out[name] = float(min(max(v, lo), hi))
    return out


class Edge:
    """Rising-edge detector for a button value (click fields are 0/1 floats)."""

    def __init__(self) -> None:
        self._prev = False

    def __call__(self, value: float) -> bool:
        pressed = value > 0.5
        rose = pressed and not self._prev
        self._prev = pressed
        return rose


@dataclass
class ArmClutch:
    max_speed: float = 0.5      # m/s, goal position rate limit
    reach: float = 0.45         # m, max goal distance from the shoulder
    scale: float = 1.0          # operator arm -> robot arm
    absolute: bool = False      # position from the operator's arm pose (see module doc)
    op_shoulder: np.ndarray = field(default_factory=lambda: np.zeros(3))   # OP_SHOULDER[side]
    shoulder: np.ndarray = field(default_factory=lambda: np.zeros(3))  # trunk frame
    engaged: bool = False
    _C0: np.ndarray | None = None
    _H0: np.ndarray | None = None
    _R_head: np.ndarray | None = None  # operator heading at engage (robot axes)
    _last: np.ndarray | None = None

    def engage(self, controller: np.ndarray, head_yaw: float, wrist: np.ndarray) -> None:
        self._C0, self._H0 = controller.copy(), wrist.copy()
        self._R_head = rot_z(head_yaw)
        self._last = wrist.copy()
        self.engaged = True

    def release(self) -> None:
        self.engaged = False

    def goal(self, controller: np.ndarray, dt: float, head_pos: np.ndarray | None = None) -> np.ndarray:
        """IK goal (trunk frame) for the current controller pose (``head_pos``:
        headset position, needed by ``absolute``)."""
        assert self.engaged and self._C0 is not None
        Rh = self._R_head
        dR = Rh.T @ (controller[:3, :3] @ self._C0[:3, :3].T) @ Rh         # rotation delta, heading frame
        goal = np.eye(4)
        goal[:3, :3] = dR @ self._H0[:3, :3]
        if self.absolute and head_pos is not None:
            op_shoulder = head_pos + Rh @ self.op_shoulder
            pos = self.shoulder + self.scale * Rh.T @ (controller[:3, 3] - op_shoulder)
        else:
            pos = self._H0[:3, 3] + self.scale * Rh.T @ (controller[:3, 3] - self._C0[:3, 3])
        off = pos - self.shoulder
        if np.linalg.norm(off) > self.reach:
            pos = self.shoulder + off * (self.reach / np.linalg.norm(off))
        step = pos - self._last[:3, 3]
        lim = self.max_speed * max(dt, 1e-3)
        if np.linalg.norm(step) > lim:
            pos = self._last[:3, 3] + step * (lim / np.linalg.norm(step))
        goal[:3, 3] = pos
        self._last = goal
        return goal


def arm_scale_from_tpose(head: np.ndarray, controllers: dict[str, np.ndarray], robot_arm_m: float) -> float:
    """Robot arm / operator arm, the operator standing in a T-pose (arms out sideways)."""
    R = rot_z(yaw_of(head[:3, :3]))
    human = [np.linalg.norm(c[:3, 3] - (head[:3, 3] + R @ OP_SHOULDER[side])) - GRIP_TO_WRIST
             for side, c in controllers.items()]
    return robot_arm_m / float(np.mean(human))


def _line_px(img, a, b, color) -> None:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    us = np.linspace(a[0], b[0], n).astype(int)
    vs = np.linspace(a[1], b[1], n).astype(int)
    for du, dv in ((0, 0), (1, 0), (0, 1)):                      # 2 px thick
        u, v = us + du, vs + dv
        ok = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
        img[v[ok], u[ok], :3] = color


def perspective(eye, target, size: int, fov_deg: float = 70.0):
    """Pinhole projector (z up): returns ``project(points) -> (uv, in_front)``."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    c, focal = size / 2, size / 2 / math.tan(math.radians(fov_deg / 2))

    def project(q):
        d = np.atleast_2d(q) - eye
        zc = d @ f
        ok = zc > 0.2
        zc = np.where(ok, zc, 1.0)
        return np.stack((c + focal * (d @ right) / zc, c - focal * (d @ up) / zc), axis=1), ok
    return project


AXIS_RGB = ((255, 60, 60), (60, 255, 60), (80, 140, 255))  # x red, y green, z blue


POSE_EYE = np.array([1.4, -1.0, 0.9])      # K1 trunk frame, front-right, above
MESH_RGB = np.array([205.0, 170.0, 145.0])


def _draw_mesh(img, project, verts: np.ndarray, faces: np.ndarray) -> None:
    """Shaded mesh as depth-sorted splats (vertices + face centroids, far to near):
    no rasteriser, a few ms for SMPL-X's 10k vertices / 21k faces."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], n)
    pts = np.concatenate([verts, tri.mean(1)])
    nrm = np.concatenate([vn, n])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    to_eye = POSE_EYE - pts
    dist = np.linalg.norm(to_eye, axis=1)
    shade = 0.3 + 0.7 * np.abs((nrm * to_eye).sum(1)) / dist            # two-sided Lambert, light at the eye
    uv, ok = project(pts)
    order = np.argsort(-dist[ok])                                        # far first, near overwrites
    u, v = uv[ok][order, 0].astype(int), uv[ok][order, 1].astype(int)
    col = (shade[ok][order, None] * MESH_RGB).astype(np.uint8)
    import cv2

    h, w = img.shape[:2]
    layer = np.zeros((h, w, 3), np.uint8)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        uu, vv = u + du, v + dv
        m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        layer[vv[m], uu[m]] = col[m]
    layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # fill splat gaps
    hit = layer.any(-1)
    img[hit, :3] = layer[hit]


def render_pose(skeleton: np.ndarray, frames: list[tuple[np.ndarray, float]],
                links: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = (), size: int = 512,
                mesh: tuple[np.ndarray, np.ndarray] | None = None, ground_z: float = -0.55) -> np.ndarray:
    """Robot skeleton (white) seen from the front-right, plus coordinate triads
    ``(T, axis_length_m)``, coloured segments ``(a, b, rgb)`` and an optional shaded
    ``mesh`` (vertices, faces) under them, all in the trunk frame."""
    img = np.full((size, size, 4), (18, 18, 24, 255), np.uint8)
    project = perspective(POSE_EYE, (0.05, 0.0, 0.0), size, fov_deg=45)
    ground = np.array([[x, y, ground_z] for x in np.arange(-0.6, 0.61, 0.15) for y in np.arange(-0.6, 0.61, 0.15)])
    uv, ok = project(ground)
    for (u, v), k in zip(uv.astype(int), ok):
        if k and 0 <= u < size and 0 <= v < size:
            img[v, u, :3] = 90
    if mesh is not None:
        _draw_mesh(img, project, *mesh)
    if len(skeleton):
        a, va = project(skeleton[:, 0])
        b, vb = project(skeleton[:, 1])
        for i in np.flatnonzero(va & vb):
            _line_px(img, a[i], b[i], 235)
    for a3, b3, rgb in links:
        (a, b), ok = project(np.array([a3, b3]))
        if ok.all():
            _line_px(img, a, b, rgb)
    for T, length in frames:
        pts = np.array([T[:3, 3]] + [T[:3, 3] + length * T[:3, k] for k in range(3)])
        uv, ok = project(pts)
        if ok.all():
            for k in range(3):
                _line_px(img, uv[0], uv[k + 1], AXIS_RGB[k])
    return img


def draw_text(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], x: int = 12, y0: int = 26,
              dy: int = 26, scale: float = 0.6) -> np.ndarray:
    """Overlay text lines (RGB colours) with OpenCV's Hershey font."""
    import cv2

    rgb = np.ascontiguousarray(img[..., :3])
    for i, (text, color) in enumerate(lines):
        cv2.putText(rgb, text, (x, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    img[..., :3] = rgb
    return img
