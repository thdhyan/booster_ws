"""The Booster K1 motion CSV contract, in one place.

Every producer (GMR, SOMA Retargeter) and every consumer (the feasibility
gate, the MuJoCo recorder, `booster_train`) has to agree on this, so it is
defined once here rather than copy-pasted into each script.

    col 0-2   root position  x, y, z        [m]
    col 3-6   root quaternion x, y, z, w    [xyzw]   <- Booster order
    col 7-28  22 joint positions            [rad]   K1_JOINT_NAMES order

Two facts that are easy to get wrong, both verified against real files rather
than assumed:

* **Quaternion order is xyzw, not wxyz.** GMR's `smplx_to_robot.py:162` writes
  ``qpos[3:7][[1,2,3,0]]``, reordering MuJoCo's scalar-first root quaternion
  into xyzw before saving. Some GMR entry points (`GMRResult`) publish wxyz, so
  this is path-dependent -- `QUAT_ORDER_NOTE` records the distinction.
* **Joint order is fixed by the robot, not by the producer.** GMR's
  `K1_serial.xml` declares its 22 hinge joints in exactly `K1_JOINT_NAMES`
  order, so `dof_pos` from GMR needs no reordering. A producer that *did* need
  reordering would produce plausible-looking but wrong motion, so
  :func:`verify_joint_order` re-checks it against the model at runtime.
"""
from __future__ import annotations

import os

import numpy as np

K1_JOINT_NAMES: list[str] = [
    # head (2)
    "AAHead_yaw", "Head_pitch",
    # left arm (4)
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll",
    "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    # right arm (4)
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll",
    "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    # left leg (6)
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    # right leg (6)
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

N_JOINTS = len(K1_JOINT_NAMES)          # 22
N_COLS = 7 + N_JOINTS                   # 29
N_LEG_JOINTS = 12
N_ARM_JOINTS = 8
N_HEAD_JOINTS = 2

QUAT_ORDER_NOTE = """\
Booster CSV wants root quat as xyzw.

  GMR smplx_to_robot.py  : saves xyzw (it reorders wxyz qpos via [[1,2,3,0]])
  GMR GMRResult          : publishes wxyz -- reorder before writing a CSV
  SOMA Retargeter        : converts Euler deg -> xyzw itself
"""

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_K1_URDF = os.path.join(_ROOT, "src", "k1_description", "assets",
                               "robots", "K1", "K1_22dof.urdf")
DEFAULT_K1_XML = os.path.join(_ROOT, "src", "k1_description", "assets",
                              "robots", "K1", "K1_22dof.xml")


def as_matrix(arr) -> np.ndarray:
    """Coerce motion data to a contiguous (N, 29) float64 array."""
    m = np.asarray(arr, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != N_COLS:
        raise ValueError(f"expected (N, {N_COLS}) motion, got {m.shape}")
    return np.ascontiguousarray(m)


def read_csv(path: str, default_fps: float = 50.0) -> tuple[np.ndarray, float]:
    """Read a Booster motion CSV. Returns (motion (N,29), fps).

    The CSV format itself has no header and no fps field, so fps comes from a
    `<name>.fps.txt` sidecar written by :func:`write_csv`, falling back to
    `default_fps`. Without the sidecar the rate is a guess, and a wrong rate
    makes every velocity/accel number wrong by the square of the error ratio,
    which is why the gate takes --fps explicitly.
    """
    m = np.loadtxt(path, delimiter=",", ndmin=2)
    fps = default_fps
    side = os.path.splitext(path)[0] + ".fps.txt"
    if os.path.exists(side):
        try:
            with open(side) as f:
                fps = float(f.read().split()[0])
        except (ValueError, IndexError, OSError):
            pass
    return as_matrix(m), fps


def write_csv(path: str, motion, fps: float | None = None) -> str:
    """Write a Booster motion CSV, validating the contract first."""
    m = as_matrix(motion)
    check = validate(m)
    if not check["ok"]:
        raise ValueError("refusing to write an invalid motion CSV: "
                         + "; ".join(check["errors"]))
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    np.savetxt(path, m, delimiter=",", fmt="%.8f")
    if fps:
        with open(os.path.splitext(path)[0] + ".fps.txt", "w") as f:
            f.write(f"{float(fps):.6f}\n")
    return path


def validate(motion) -> dict:
    """Structural checks on the CSV contract (not physics -- that's the gate)."""
    m = np.asarray(motion, dtype=np.float64)
    errors: list[str] = []
    if m.ndim != 2 or m.shape[1] != N_COLS:
        return {"ok": False, "errors": [f"shape {m.shape} != (N, {N_COLS})"],
                "n_frames": 0}
    if not np.isfinite(m).all():
        bad = int((~np.isfinite(m)).sum())
        errors.append(f"{bad} non-finite value(s)")
    qn = np.linalg.norm(m[:, 3:7], axis=1)
    dev = float(np.abs(qn - 1.0).max()) if len(qn) else 0.0
    if dev > 1e-3:
        errors.append(
            f"root quats not unit norm (max dev {dev:.2e}) -- "
            f"check xyzw vs wxyz; a scalar-first reading gives the same dev, "
            f"so verify the producer")
    return {"ok": not errors, "errors": errors, "n_frames": int(m.shape[0]),
            "quat_norm_max_dev": dev}


def verify_joint_order(xml_path: str = DEFAULT_K1_XML) -> list[str]:
    """Return the model's hinge-joint order, raising if it is not ours.

    Cheap insurance: a producer that emits joints in a different order yields
    motion that still looks plausible in a render, so this is the one check
    worth failing loudly on.
    """
    import mujoco as mj
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"K1 MJCF not found: {xml_path}")
    m = mj.MjModel.from_xml_path(xml_path)
    hinges = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i)
              for i in range(m.njnt)
              if m.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
    if hinges != K1_JOINT_NAMES:
        for i, (a, b) in enumerate(zip(hinges, K1_JOINT_NAMES)):
            if a != b:
                raise ValueError(
                    f"joint order mismatch at index {i}: model={a!r} "
                    f"expected={b!r}\nmodel  : {hinges}\nexpected: {K1_JOINT_NAMES}")
        raise ValueError(f"hinge count {len(hinges)} != {N_JOINTS}")
    return hinges


def joint_limits_from_urdf(urdf_path: str = DEFAULT_K1_URDF) -> dict:
    """Parse per-joint position limits from the K1 URDF (no ROS needed)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    lim: dict[str, tuple[float, float]] = {}
    for j in root.iter("joint"):
        name = j.get("name")
        if not name or j.get("type") not in ("revolute", "continuous"):
            continue
        lo = hi = None
        for l in j.iter("limit"):
            lo = float(l.get("lower")) if l.get("lower") is not None else -3.14
            hi = float(l.get("upper")) if l.get("upper") is not None else 3.14
        if name in K1_JOINT_NAMES:
            lim[name] = (lo, hi)
    missing = [n for n in K1_JOINT_NAMES if n not in lim]
    if missing:
        raise ValueError(f"URDF missing limits for {len(missing)} joints: {missing}")
    return lim
