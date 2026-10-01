#!/usr/bin/env python3
"""SOMA Retargeter CSV -> Booster K1 motion CSV (the booster_train contract).

SOMA writes per-robot CSVs in its own schema. For `booster_t1` (the 29-DoF
variant bundled with the repo) that schema is:

    Frame, root_translateX/Y/Z, root_rotateX/Y/Z, <29 joint columns>
      * root translation in CENTIMETRES
      * root rotation in DEGREES, Euler XYZ
      * joint angles in DEGREES
      * a leading integer Frame index column

The Booster contract (booster_assets.motions.K1_JOINT_NAMES, consumed by
`booster_train/scripts/csv_to_npz.py` and by scripts/motion_feasibility_gate.py)
is:

    root x, y, z                    [m]
    root quat x, y, z, w            [xyzw]   <- note the order
    22 joint positions              [rad]
    no Frame column, no header

So this converter does five things, all of which are silent-corruption risks if
skipped:

  1. drop the Frame index column
  2. cm -> m on the root translation
  3. Euler XYZ degrees -> quaternion, in **xyzw** order
  4. degrees -> radians on every joint
  5. 29-DoF T1 column set -> 22-DoF K1 column set
     (drop both 3-DoF wrists/hands, drop Waist, and add the "A" prefix that
      the K1 URDF uses on the shoulder joints)

Usage
  python scripts/soma_csv_to_k1_csv.py soma_t1.csv out_k1.csv
  python scripts/soma_csv_to_k1_csv.py soma_t1.csv out_k1.csv --root-z-offset 0.05
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

# Booster K1 joint order — the contract's column order after the 7 root columns.
K1_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

# SOMA's T1 column name -> K1 contract column name. Anything absent is dropped.
# The K1 URDF prefixes the shoulder joints with "A"; SOMA's T1 model does not.
SOMA_TO_K1 = {
    "AAHead_yaw": "AAHead_yaw",
    "Head_pitch": "Head_pitch",
    "Left_Shoulder_Pitch": "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll": "Left_Shoulder_Roll",
    "Left_Elbow_Pitch": "Left_Elbow_Pitch",
    "Left_Elbow_Yaw": "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch": "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll": "Right_Shoulder_Roll",
    "Right_Elbow_Pitch": "Right_Elbow_Pitch",
    "Right_Elbow_Yaw": "Right_Elbow_Yaw",
    "Left_Hip_Pitch": "Left_Hip_Pitch",
    "Left_Hip_Roll": "Left_Hip_Roll",
    "Left_Hip_Yaw": "Left_Hip_Yaw",
    "Left_Knee_Pitch": "Left_Knee_Pitch",
    "Left_Ankle_Pitch": "Left_Ankle_Pitch",
    "Left_Ankle_Roll": "Left_Ankle_Roll",
    "Right_Hip_Pitch": "Right_Hip_Pitch",
    "Right_Hip_Roll": "Right_Hip_Roll",
    "Right_Hip_Yaw": "Right_Hip_Yaw",
    "Right_Knee_Pitch": "Right_Knee_Pitch",
    "Right_Ankle_Pitch": "Right_Ankle_Pitch",
    "Right_Ankle_Roll": "Right_Ankle_Roll",
    # intentionally dropped: Left/Right_Wrist_Pitch, _Wrist_Yaw, _Hand_Roll, Waist
}

DEG = math.pi / 180.0


def euler_xyz_deg_to_quat_xyzw(rx: float, ry: float, rz: float):
    """Euler XYZ (degrees) -> quaternion in Booster's **xyzw** order."""
    cx, sx = math.cos(rx * DEG * 0.5), math.sin(rx * DEG * 0.5)
    cy, sy = math.cos(ry * DEG * 0.5), math.sin(ry * DEG * 0.5)
    cz, sz = math.cos(rz * DEG * 0.5), math.sin(rz * DEG * 0.5)
    # intrinsic X->Y->Z
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    w = cx * cy * cz + sx * sy * sz
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return (x / n, y / n, z / n, w / n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src", help="SOMA retargeter CSV")
    p.add_argument("dst", help="output K1 motion CSV")
    p.add_argument("--root-z-offset", type=float, default=0.0,
                   help="add this many metres to root z (floor-height matching)")
    p.add_argument("--ground-feet", action="store_true",
                   help="shift root z so the lowest foot rests on the floor "
                        "(uses K1 forward kinematics; overrides --root-z-offset)")
    p.add_argument("--no-header", action="store_true",
                   help="input has no header row (assume SOMA column order)")
    args = p.parse_args()

    with open(args.src) as f:
        header = f.readline().strip().split(",")
    raw = np.loadtxt(args.src, delimiter=",", skiprows=1, ndmin=2)
    if raw.shape[1] != len(header):
        raise RuntimeError(
            f"{args.src}: {raw.shape[1]} data columns but {len(header)} headers")

    col = {name: i for i, name in enumerate(header)}
    print(f"[soma->k1] {len(raw)} frames, {len(header)} source columns")

    # --- root: cm -> m, Euler deg -> quat xyzw ------------------------------
    pos = raw[:, [col["root_translateX"], col["root_translateY"],
                  col["root_translateZ"]]] * 0.01
    if args.root_z_offset:
        pos[:, 2] += args.root_z_offset
    rot = raw[:, [col["root_rotateX"], col["root_rotateY"],
                  col["root_rotateZ"]]]

    n = len(raw)
    quat = np.zeros((n, 4))
    for i in range(n):
        quat[i] = euler_xyz_deg_to_quat_xyzw(*rot[i])

    # --- joints: deg -> rad, rename, reorder, drop ------------------------
    src_cols, used_names = [], set()
    for k1_name in K1_JOINT_NAMES:
        src_name = next((s for s, d in SOMA_TO_K1.items() if d == k1_name), None)
        if src_name is None or src_name not in col:
            raise KeyError(f"no SOMA column maps to K1 joint {k1_name}")
        src_cols.append(col[src_name])
        used_names.add(src_name)
    joints = np.deg2rad(raw[:, src_cols])
    dropped = sorted(n for n in header[1:]
                     if not n.startswith("root_") and n not in used_names)
    print(f"[soma->k1] dropped {len(dropped)} source joints: {dropped}")

    # --- assemble: 3 pos + 4 quat + 22 joints = 29 columns ----------------
    out = np.concatenate([pos, quat, joints], axis=1)
    if out.shape[1] != 7 + len(K1_JOINT_NAMES):
        raise RuntimeError(f"expected {7 + len(K1_JOINT_NAMES)} cols, got {out.shape[1]}")

    if args.ground_feet:
        # SOMA solves IK against its own ground plane, so the resulting root
        # height generally does not match K1's leg length — the robot ends up
        # floating (or sunk) with no foot ever near z=0. Solve it with real
        # K1 forward kinematics rather than a hand-tuned constant.
        shift = _ground_offset(out, REPO_URDF)
        pos[:, 2] += shift
        out[:, 0:3] = pos
        print(f"[soma->k1] grounded: root z shifted by {shift:+.4f} m so the "
              f"lowest sole sits on the floor")
    elif args.root_z_offset:
        pos[:, 2] += args.root_z_offset
        out[:, 0:3] = pos

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)) or ".", exist_ok=True)
    np.savetxt(args.dst, out, delimiter=",")
    print(f"[soma->k1] wrote {args.dst}  {out.shape[0]} x {out.shape[1]}")
    print(f"[soma->k1] root z range [{pos[:,2].min():.3f}, {pos[:,2].max():.3f}] m")
    print(f"[soma->k1] root xy range x[{pos[:,0].min():.2f},{pos[:,0].max():.2f}] "
          f"y[{pos[:,1].min():.2f},{pos[:,1].max():.2f}] m")
    return 0


REPO_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src/k1_description/assets/robots/K1/K1_22dof.xml")
SOLE_DROP = 0.038   # foot_link origin -> sole plane, from the K1 MJCF foot box


def _sole_heights(out, mjcf):
    """Sole z of both feet per frame, via K1 forward kinematics."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
          for i in range(model.njnt)]
    order = [n for n in jn if n in K1_JOINT_NAMES]
    if order != K1_JOINT_NAMES:
        raise KeyError("K1 MJCF joint order does not match K1_JOINT_NAMES")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in ("left_foot_link", "right_foot_link")]
    nq = model.nq
    h = np.zeros((len(out), 2))
    for t in range(len(out)):
        data.qpos[0:3] = out[t, 0:3]
        data.qpos[3:7] = out[t, 3:7]          # xyzw, which is what MuJoCo wants
        data.qpos[7:7 + len(K1_JOINT_NAMES)] = out[t, 7:]
        mujoco.mj_kinematics(model, data)
        for s, b in enumerate(feet):
            h[t, s] = data.xpos[b][2] - SOLE_DROP
    return h


def _ground_offset(out, mjcf):
    """Vertical shift that puts the lowest sole on the floor, using the frames
    where the robot is genuinely standing (both feet within 8 cm of each other
    in height) rather than the global minimum, which a single bad frame can set."""
    h = _sole_heights(out, mjcf)
    lo = h.min(axis=1)
    # standing frames: the two feet are at similar heights
    standing = np.abs(h[:, 0] - h[:, 1]) < 0.08
    ref = lo[standing] if standing.any() else lo
    return float(-np.percentile(ref, 5))     # 5th pct tolerates a little sink


if __name__ == "__main__":
    sys.exit(main())
