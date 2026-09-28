#!/usr/bin/env python3
"""Motion feasibility gate — K1.

Answers the question "is this retargeted motion safe to put on the robot?"
*before* anything touches hardware. SOMA Retargeter's own docs say its output
is kinematic and that its IK optimizer "does not prove physical feasibility or
controller performance" — so this script is that missing check.

Consumes the Booster motion CSV contract:
    col 0-2   root position  x, y, z            [m]
    col 3-6   root quaternion x, y, z, w        [xyzw]
    col 7-28  22 joint positions                [rad]
Sampled at 50 Hz (fps is not stored in the file).

Checks, in order of how likely they are to be the thing that breaks the robot:

  1. joint_limits      does any joint exceed the K1 URDF range?
  2. joint_velocity    does any joint exceed URDF max velocity?
  3. joint_accel       is the motion smoother than the actuators can track?
  4. torque_estimate   does the PD torque implied by tracking stay under effort?
  5. foot_penetration  do feet sink below the floor plane?
  6. foot_float        do planted feet hover?
  7. foot_slip         do feet slide while supposed to be planted?
  8. root_height       does the pelvis leave the K1's realisable band?
  9. double_support   is there enough time with both feet down to be catchable?
 10. com_margin        is the CoM over the support polygon when only one foot
                       is down? (catches statically-unrecoverable poses)

Exit code 0 = pass, 1 = fail, 2 = could not evaluate (bad input).

Usage:
  python3 scripts/motion_feasibility_gate.py motion.csv
  python3 scripts/motion_feasibility_gate.py motion.csv --fps 50 --json report.json
  python3 scripts/motion_feasibility_gate.py motions/*.csv --strict

This is a kinematic screen, not a physics rollout. Passing it means the motion
is *worth simulating*, not that it is safe on hardware. Phase 4 of
docs/video_to_motion_plan.md (MuJoCo -> Gazebo -> Isaac) is still mandatory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict

import numpy as np

# --------------------------------------------------------------------------- #
# K1 model constants (verified against K1_22dof.urdf)
# --------------------------------------------------------------------------- #
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(REPO, "src/k1_description/assets/robots/K1/K1_22dof.urdf")

# K1 standing root (Trunk) height. BOOSTER_K1_CFG init_state z=0.57.
ROOT_HEIGHT_NOMINAL = 0.57
ROOT_HEIGHT_MIN = 0.45   # below this the robot is on its way down
ROOT_HEIGHT_MAX = 0.75   # above this is a jump/flight phase, not a dance

# A sole within this height of the floor counts as planted.
FOOT_CONTACT_Z = 0.04

# Mass fraction per DoF, gross but adequate for a PD torque bound. A real gate
# would invert the full M matrix; this is a screening estimate.
MASS_PER_LEG_JOINT = 1.6   # kg equivalent per leg joint
MASS_PER_ARM_JOINT = 0.5
MASS_HEAD = 0.6
NOMINAL_GRAVITY = 9.81

# booster_assets.motions.K1_JOINT_NAMES — the CSV column order
JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
N_JOINTS = len(JOINT_NAMES)
N_COLS = 7 + N_JOINTS  # 7 root + 22 joints

FOOT_JOINTS = ["Left_Ankle_Roll", "Right_Ankle_Roll"]

# K1 MuJoCo model. Used for real forward kinematics so foot/CoM numbers are
# geometric truth, not a crude root-relative guess. The model's joint order
# matches K1_JOINT_NAMES exactly (verified: 7 free-joint qpos, then 22 hinges
# in the same sequence), and the sole box geom sits 0.038 m below each
# foot_link origin.
MJCF = os.path.join(REPO, "src/k1_description/assets/robots/K1/K1_22dof.xml")
SOLE_DROP = 0.038   # foot_link origin -> sole plane, from the K1 MJCF foot box

_FK = None  # lazily-built (model, data, foot body ids)


def _fk():
    """Build (and cache) the MuJoCo model used for foot/CoM forward kinematics."""
    global _FK
    if _FK is None:
        import mujoco  # imported lazily so the CSV-only checks work without it
        if not os.path.isfile(MJCF):
            raise FileNotFoundError(
                f"K1 MJCF not found: {MJCF}\n"
                f"Foot/CoM checks need it; pass --mjcf or run only the "
                f"joint-space checks.")
        model = mujoco.MjModel.from_xml_path(MJCF)
        data = mujoco.MjData(model)
        feet = []
        for side in ("left_foot_link", "right_foot_link"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side)
            if bid < 0:
                raise KeyError(f"K1 MJCF is missing body {side}")
            feet.append(bid)
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
        _FK = (model, data, feet, trunk)
    return _FK


def foot_and_com_world(root_pos, root_quat, joints):
    """Forward-kinematics foot sole centres and the whole-body CoM.

    Returns
    -------
    feet : [T, 2, 3]  sole centre of each foot in world frame
    com  : [T, 3]     mass-weighted centre of mass in world frame
    """
    import mujoco

    model, data, foot_bodies, trunk = _fk()
    nq = model.nq
    T = len(joints)

    # Verify the MJCF joint order matches our CSV column order before trusting FK.
    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(model.njnt)]
    csv_order = [n for n in jnames if n in JOINT_NAMES]
    if csv_order != JOINT_NAMES:
        raise KeyError(
            f"K1 MJCF joint order does not match K1_JOINT_NAMES.\n"
            f"  mjcf: {csv_order}\n  csv : {JOINT_NAMES}")

    qpos = np.zeros((T, nq))
    qpos[:, 0:3] = root_pos
    # CSV stores xyzw; MuJoCo free joints store wxyz.
    qpos[:, 3] = root_quat[:, 3]
    qpos[:, 4:7] = root_quat[:, 0:3]
    # hinge joints start at qpos index 7 in the same order as JOINT_NAMES
    qpos[:, 7:7 + N_JOINTS] = joints

    feet = np.zeros((T, 2, 3))
    com = np.zeros((T, 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        for s, b in enumerate(foot_bodies):
            # Sole is below the foot_link origin along the link's own -Z.
            feet[t, s] = (data.xpos[b][0], data.xpos[b][1],
                          data.xpos[b][2] - SOLE_DROP)
        com[t] = data.subtree_com[trunk]
    return feet, com


def joint_masses() -> np.ndarray:
    m = np.zeros(N_JOINTS)
    for i, n in enumerate(JOINT_NAMES):
        if "Hip" in n or "Knee" in n or "Ankle" in n:
            m[i] = MASS_PER_LEG_JOINT
        elif "Shoulder" in n or "Elbow" in n:
            m[i] = MASS_PER_ARM_JOINT
        else:
            m[i] = MASS_HEAD
    return m


# --------------------------------------------------------------------------- #
# URDF limits
# --------------------------------------------------------------------------- #
@dataclass
class Limits:
    lower: np.ndarray
    upper: np.ndarray
    effort: np.ndarray
    velocity: np.ndarray

    @staticmethod
    def from_urdf(path: str) -> "Limits":
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"K1 URDF not found: {path}\n"
                f"Expected at {URDF}. Override with --urdf.")
        root = ET.parse(path).getroot()
        by_name = {}
        for j in root.findall("joint"):
            if j.get("type") not in ("revolute", "continuous", "prismatic"):
                continue
            lim = j.find("limit")
            if lim is None:
                continue
            by_name[j.get("name")] = (
                float(lim.get("lower")), float(lim.get("upper")),
                float(lim.get("effort")), float(lim.get("velocity")),
            )
        missing = [n for n in JOINT_NAMES if n not in by_name]
        if missing:
            raise KeyError(f"URDF is missing expected K1 joints: {missing}")

        lo = np.array([by_name[n][0] for n in JOINT_NAMES])
        hi = np.array([by_name[n][1] for n in JOINT_NAMES])
        ef = np.array([by_name[n][2] for n in JOINT_NAMES])
        ve = np.array([by_name[n][3] for n in JOINT_NAMES])
        # Unbounded joints get a generous symmetric range.
        inf = 1e3
        lo = np.where(np.isfinite(lo), lo, -inf)
        hi = np.where(np.isfinite(hi), hi, inf)
        return Limits(lo, hi, ef, ve)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    worst: float | None = None
    worst_frame: int | None = None
    worst_joint: str | None = None
    severity: str = "fail"   # "fail" | "warn"


@dataclass
class Report:
    path: str
    fps: float
    n_frames: int
    duration_s: float
    checks: list = field(default_factory=list)

    @property
    def failures(self):
        return [c for c in self.checks if not c.passed and c.severity == "fail"]

    @property
    def warnings(self):
        return [c for c in self.checks if not c.passed and c.severity == "warn"]

    @property
    def ok(self):
        return not self.failures


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_motion(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (root_pos[T,3], root_quat_xyzw[T,4], joints[T,22])."""
    try:
        raw = np.loadtxt(path, delimiter=",", ndmin=2)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"cannot parse {path}: {e}") from e
    if raw.shape[1] < N_COLS:
        raise ValueError(
            f"{path}: expected >= {N_COLS} columns "
            f"(7 root + {N_JOINTS} joints), got {raw.shape[1]}")
    if raw.shape[0] < 2:
        raise ValueError(f"{path}: need at least 2 frames, got {raw.shape[0]}")
    if not np.all(np.isfinite(raw[:, :N_COLS])):
        bad = int(np.argmax(~np.isfinite(raw[:, :N_COLS]).all(axis=1)))
        raise ValueError(f"{path}: non-finite value at frame {bad}")
    return raw[:, 0:3], raw[:, 3:7], raw[:, 7:N_COLS]


def quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix. q: [...,4] -> [...,3,3]."""
    q = q_xyzw / np.linalg.norm(q_xyzw, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def foot_world_positions(root_pos, root_quat, joints):
    """Deprecated shim — use foot_and_com_world() (real FK)."""
    return foot_and_com_world(root_pos, root_quat, joints)[0]


def joint_masses() -> np.ndarray:
    m = np.zeros(N_JOINTS)
    for i, n in enumerate(JOINT_NAMES):
        if "Hip" in n or "Knee" in n or "Ankle" in n:
            m[i] = MASS_PER_LEG_JOINT
        elif "Shoulder" in n or "Elbow" in n:
            m[i] = MASS_PER_ARM_JOINT
        else:
            m[i] = MASS_HEAD
    return m


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_joint_limits(q, lim: Limits) -> Check:
    over_lo = np.maximum(lim.lower - q, 0.0)
    over_hi = np.maximum(q - lim.upper, 0.0)
    excess = np.maximum(over_lo, over_hi)
    f, j = np.unravel_index(int(np.argmax(excess)), excess.shape)
    worst = float(excess[f, j])
    if worst <= 1e-6:
        return Check("joint_limits", True,
                     f"all {N_JOINTS} joints within URDF range")
    return Check("joint_limits", False,
                 f"'{JOINT_NAMES[j]}' exceeds URDF limit by {worst:.3f} rad "
                 f"(value {q[f, j]:+.3f}, range "
                 f"[{lim.lower[j]:+.3f}, {lim.upper[j]:+.3f}])",
                 worst=worst, worst_frame=int(f), worst_joint=JOINT_NAMES[j])


def check_joint_velocity(q, lim: Limits, fps: float) -> Check:
    dt = 1.0 / fps
    dq = np.diff(q, axis=0) / dt
    ratio = np.abs(dq) / np.maximum(lim.velocity, 1e-6)
    f, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    worst = float(ratio[f, j])
    if worst <= 1.0:
        return Check("joint_velocity", True,
                     f"peak |dq| = {np.abs(dq).max():.2f} rad/s "
                     f"(within URDF max velocity)")
    return Check("joint_velocity", worst <= 1.25,
                 f"'{JOINT_NAMES[j]}' reaches {np.abs(dq[f, j]):.2f} rad/s = "
                 f"{worst:.2f}x the URDF limit "
                 f"({lim.velocity[j]:.1f} rad/s) — motion is too fast to track",
                 worst=worst, worst_frame=int(f), worst_joint=JOINT_NAMES[j],
                 severity="fail" if worst > 1.25 else "warn")


def check_joint_accel(q, fps: float, a_max=80.0) -> Check:
    dt = 1.0 / fps
    dq = np.diff(q, axis=0, n=2) / dt          # [T-2, 22]
    ddq = np.diff(dq, axis=0) / dt             # [T-3, 22]
    ratio = np.abs(ddq) / a_max
    f, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    worst = float(ratio[f, j])
    if worst <= 1.0:
        return Check("joint_accel", True,
                     f"peak |ddq| = {np.abs(ddq).max():.1f} rad/s^2 "
                     f"(within {a_max:.0f})")
    return Check("joint_accel", worst <= 2.0,
                 f"'{JOINT_NAMES[j]}' reaches {np.abs(ddq[f, j]):.1f} rad/s^2 "
                 f"= {worst:.2f}x the assumed {a_max:.0f} rad/s^2 limit — "
                 f"expect tracking lag and foot slip",
                 worst=worst, worst_frame=int(f), worst_joint=JOINT_NAMES[j],
                 severity="fail" if worst > 2.0 else "warn")


def check_torque_estimate(q, fps: float, lim: Limits, kp=60.0, kd=4.0) -> Check:
    """tau ~ I*ddq + kd*m*dq bound using a per-joint effective-inertia proxy."""
    dt = 1.0 / fps
    dq = np.diff(q, axis=0, n=2) / dt            # [T-2, 22]
    ddq = np.diff(dq, axis=0) / dt               # [T-3, 22]
    # dq is one frame longer than ddq — align on the trailing window so the
    # damping and inertia terms broadcast against each other.
    dq = dq[1:]                                   # [T-3, 22]
    m = joint_masses()
    inertia = m * (0.15 ** 2)                    # effective mass * link length^2
    tau = inertia[None, :] * ddq + kd * m[None, :] * dq
    ratio = np.abs(tau) / np.maximum(lim.effort, 1e-6)
    f, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    worst = float(ratio[f, j])
    if worst <= 1.0:
        return Check("torque_estimate", True,
                     f"peak |tau| = {np.abs(tau).max():.1f} Nm "
                     f"(within URDF effort)")
    return Check("torque_estimate", worst <= 1.3,
                 f"'{JOINT_NAMES[j]}' needs ~{abs(tau[f, j]):.1f} Nm = "
                 f"{worst:.2f}x the {lim.effort[j]:.0f} Nm effort limit — "
                 f"actuator will saturate and the pose will not be held",
                 worst=worst, worst_frame=int(f), worst_joint=JOINT_NAMES[j],
                 severity="fail" if worst > 1.3 else "warn")


def check_foot_penetration(feet) -> Check:
    pen = np.maximum(0.03 - feet[:, :, 2], 0.0)        # [T, 2] depth below floor
    frame, side_i = np.unravel_index(int(np.argmax(pen)), pen.shape)
    worst = float(pen[frame, side_i])
    if worst <= 0.02:
        return Check("foot_penetration", True,
                     f"max sole depth below floor = {worst*1000:.1f} mm")
    side = "left" if side_i == 0 else "right"
    return Check("foot_penetration", False,
                 f"{side} foot penetrates the floor by {worst*1000:.1f} mm at "
                 f"frame {frame} — retarget put the sole through the ground",
                 worst=worst, worst_frame=int(frame))


def check_foot_float(feet) -> Check:
    """Flag clips that spend long stretches with no foot on the ground."""
    z = feet[:, :, 2]
    both_air = (z[:, 0] > FOOT_CONTACT_Z) & (z[:, 1] > FOOT_CONTACT_Z)
    if not both_air.any():
        return Check("foot_float", True, "never fully airborne in a standing pose")
    frac = float(both_air.mean())
    f = int(np.argmax(both_air))
    return Check("foot_float", frac < 0.15,
                 f"both feet airborne for {frac*100:.0f}% of frames "
                 f"(first at frame {f}) — flight phases will not track",
                 worst=frac, worst_frame=f,
                 severity="fail" if frac > 0.35 else "warn")


def check_foot_slip(feet, fps: float, plant_thresh=0.15) -> Check:
    """Horizontal foot travel while the foot is low enough to be planted."""
    dt = 1.0 / fps
    vel = np.diff(feet, axis=0) / dt
    horiz = np.linalg.norm(vel[:, :, :2], axis=2)      # [T-1, 2]
    zmid = 0.5 * (feet[:-1, :, 2] + feet[1:, :, 2])
    planted = zmid < SOLE_DROP + 0.03
    slip = horiz * planted                             # [T-1, 2]
    frame, side_i = np.unravel_index(int(np.argmax(slip)), slip.shape)
    worst = float(slip[frame, side_i])
    side = "left" if side_i == 0 else "right"
    if worst <= 0.05:
        return Check("foot_slip", True,
                     f"peak planted-foot speed = {worst:.3f} m/s (well planted)")
    return Check("foot_slip", worst <= 0.20,
                 f"{side} foot slides {worst:.3f} m/s while planted near frame "
                 f"{frame} — retarget has no foot locking, expect visible "
                 f"skating (SOMA foot-plant post-processing fixes this)",
                 worst=worst, worst_frame=int(frame),
                 severity="fail" if worst > 0.30 else "warn")


def check_root_height(root_pos) -> Check:
    z = root_pos[:, 2]
    f = int(np.argmax(np.abs(z - ROOT_HEIGHT_NOMINAL)))
    worst = abs(float(z[f] - ROOT_HEIGHT_NOMINAL))
    lo_bad = z < ROOT_HEIGHT_MIN
    hi_bad = z > ROOT_HEIGHT_MAX
    if not lo_bad.any() and not hi_bad.any():
        return Check("root_height", True,
                     f"root z in [{z.min():.3f}, {z.max():.3f}] m "
                     f"(nominal {ROOT_HEIGHT_NOMINAL})")
    if lo_bad.any():
        f = int(np.argmin(z))
        return Check("root_height", False,
                     f"root drops to {z[f]:.3f} m at frame {f} "
                     f"(min {ROOT_HEIGHT_MIN}) — robot would be on the floor",
                     worst=abs(float(z[f] - ROOT_HEIGHT_NOMINAL)), worst_frame=f)
    f = int(np.argmax(z))
    return Check("root_height", False,
                 f"root rises to {z[f]:.3f} m at frame {f} "
                 f"(max {ROOT_HEIGHT_MAX}) — flight phase, not a standing dance",
                 worst=abs(float(z[f] - ROOT_HEIGHT_NOMINAL)), worst_frame=f,
                 severity="warn")


def check_double_support(feet, min_frac=0.25) -> Check:
    z = feet[:, :, 2]
    both = (z[:, 0] < FOOT_CONTACT_Z) & (z[:, 1] < FOOT_CONTACT_Z)
    one = (z[:, 0] < FOOT_CONTACT_Z) ^ (z[:, 1] < FOOT_CONTACT_Z)
    neither = ~(both | one)
    fb, fo, fn = (float(both.mean()), float(one.mean()), float(neither.mean()))
    if fn < 0.02 and fb >= min_frac:
        return Check("double_support", True,
                     f"both-feet {fb*100:.0f}% / one-foot {fo*100:.0f}% / "
                     f"airborne {fn*100:.0f}%")
    if fn >= 0.02:
        return Check("double_support", False,
                     f"airborne (no support) for {fn*100:.0f}% of frames — "
                     f"no foot on the ground means nothing to balance against",
                     worst=fn, severity="fail")
    return Check("double_support", False,
                 f"double support only {fb*100:.0f}% of frames "
                 f"(want >= {min_frac*100:.0f}%) — motion is single-support "
                 f"throughout, very hard for a tracker to keep upright",
                 worst=fb, severity="warn")


def check_com_margin(feet, com) -> Check:
    """CoM vs. support polygon, using the real MuJoCo centre of mass.

    Catches statically-unrecoverable single-support poses: when only one foot
    is down, the CoM's horizontal offset from that foot is the margin you have
    to accelerate the body before the foot would have to leave the ground.
    """
    z = feet[:, :, 2]
    planted = z < FOOT_CONTACT_Z
    single = planted[:, 0] ^ planted[:, 1]
    if not single.any():
        return Check("com_margin", True,
                     "always double support — CoM always inside support")
    sup = np.where(planted[:, 0:1], feet[:, 0:1, :2], feet[:, 1:2, :2])
    margin = np.linalg.norm(com[:, :2] - sup[:, 0, :], axis=1)
    ms = np.where(single, margin, -np.inf)
    frame = int(np.argmax(ms))
    worst = float(margin[frame])
    # This is a *tracking difficulty* signal, not a proof of infeasibility: a
    # tracking policy can recover by taking an extra step, so large offsets are
    # a warning rather than a hard fail. 0.30 m is ankle-roll authority,
    # 0.60 m needs a real recovery step, beyond that the pose is a fall.
    if worst <= 0.30:
        return Check("com_margin", True,
                     f"peak single-support CoM offset = {worst:.3f} m "
                     f"(within ankle authority)")
    return Check("com_margin", worst <= 0.60,
                 f"CoM is {worst:.3f} m from the planted foot at frame "
                 f"{frame} — needs a large recovery step to stay balanced",
                 worst=worst, worst_frame=frame,
                 severity="fail" if worst > 0.60 else "warn")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def evaluate(path: str, fps: float, lim: Limits) -> Report:
    root_pos, root_quat, joints = load_motion(path)
    T = joints.shape[0]
    rep = Report(path=path, fps=fps, n_frames=T, duration_s=T / fps)
    feet, com = foot_and_com_world(root_pos, root_quat, joints)

    rep.checks = [
        check_joint_limits(joints, lim),
        check_joint_velocity(joints, lim, fps),
        check_joint_accel(joints, fps),
        check_torque_estimate(joints, fps, lim),
        check_foot_penetration(feet),
        check_foot_float(feet),
        check_foot_slip(feet, fps),
        check_root_height(root_pos),
        check_double_support(feet),
        check_com_margin(feet, com),
    ]
    return rep


def print_report(rep: Report) -> None:
    print("=" * 74)
    print(f"MOTION FEASIBILITY GATE — {os.path.basename(rep.path)}")
    print(f"  {rep.n_frames} frames @ {rep.fps:g} Hz = {rep.duration_s:.2f} s")
    print("=" * 74)
    icon = {True: "  ok  ", False: " FAIL "}
    for c in rep.checks:
        tag = icon[c.passed]
        if not c.passed and c.severity == "warn":
            tag = " warn "
        print(f"[{tag}] {c.name:20s} {c.detail}")
    print("-" * 74)
    if rep.ok:
        n_warn = len(rep.warnings)
        print(f"RESULT: PASS" + (f"  ({n_warn} warning(s))" if n_warn else ""))
        print("Next: MuJoCo rollout -> Gazebo -> Isaac (Phase 4).")
        print("This gate is kinematic only; it does not prove the robot tracks.")
    else:
        print(f"RESULT: FAIL  ({len(rep.failures)} blocking issue(s))")
        print("Do not send this to hardware. Fix the retarget, then re-run:")
        for c in rep.failures:
            print(f"  - {c.name}: {c.detail}")
    print("=" * 74)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Screen retargeted K1 motion for physical feasibility.")
    p.add_argument("motions", nargs="+", help="motion CSV file(s)")
    p.add_argument("--fps", type=float, default=50.0,
                   help="CSV sample rate (Booster convention: 50)")
    p.add_argument("--urdf", default=URDF, help="K1 URDF for joint limits")
    p.add_argument("--json", default="", help="write a JSON report here")
    p.add_argument("--strict", action="store_true",
                   help="treat warnings as failures")
    args = p.parse_args(argv)

    try:
        lim = Limits.from_urdf(args.urdf)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    reports, rc = [], 0
    for path in args.motions:
        try:
            rep = evaluate(path, args.fps, lim)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {path}: {e}", file=sys.stderr)
            rc = 2
            continue
        print_report(rep)
        reports.append(rep)
        if not rep.ok or (args.strict and rep.warnings):
            rc = 1

    if args.json and reports:
        payload = {
            "generated_by": "scripts/motion_feasibility_gate.py",
            "note": "kinematic screen only — not a physics rollout",
            "reports": [asdict(r) for r in reports],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"\nwrote {args.json}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
