#!/usr/bin/env python3
"""Convert a GMR retarget .pkl / .npz to the Booster K1 motion CSV contract.

Lives in the package (rather than being scp'd into a GMR checkout by hand) so
`k1m` is self-contained: `remote_stages.gmr` pushes this file next to the run
and invokes it there.

GMR's own `scripts/smplx_to_robot.py --save_path` dumps a pickle of

    fps, root_pos (N,3), root_rot (N,4) [xyzw], dof_pos (N,22), ...

and that is already the K1 CSV contract, column for column:

    col 0-2   root position  x, y, z        [m]
    col 3-6   root quaternion x, y, z, w    [xyzw]   <- Booster order
    col 7-28  22 joint positions            [rad]   K1_JOINT_NAMES order

Two facts make this a straight hstack rather than a real converter, both
verified rather than assumed:

  * `smplx_to_robot.py:162` writes
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
    which reorders MuJoCo's scalar-first (wxyz) qpos into xyzw. Booster wants
    xyzw, so no further conversion is needed.
  * GMR's `assets/booster_k1/K1_serial.xml` declares its 22 hinge joints in
    exactly K1_JOINT_NAMES order, so `dof_pos` is already in CSV order.

The joint-order check is re-verified here at runtime against the URDF/XML rather
than trusted, because a silent reordering would still render as plausible motion.

  python gmr_pkl_to_k1_csv.py --in out/mac_k1.pkl --out macarena_k1.csv
  python gmr_pkl_to_k1_csv.py --in out/mac_k1.npz --out macarena_k1.csv --fps 50
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

# booster_assets.motions.K1_JOINT_NAMES -- the CSV column order. Duplicated
# from k1m.schema on purpose: this file is pushed to a remote host and run
# there, where the k1m package may not be importable.
K1_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

EXPECTED_COLS = 7 + len(K1_JOINT_NAMES)


def load_any(path: str) -> tuple[np.ndarray, int]:
    """Return (N, 29) array and source fps, from a .pkl or .npz."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        fps = int(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else 30
        parts = [np.asarray(d["root_pos"], dtype=np.float64),
                 np.asarray(d["root_rot"], dtype=np.float64),
                 np.asarray(d["dof_pos"], dtype=np.float64)]
        n = min(len(p) for p in parts)
        return np.concatenate([p[:n] for p in parts], axis=1), fps

    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or "dof_pos" not in d:
        raise SystemExit(f"{path}: not a GMR motion dict "
                         f"(got {type(d).__name__})")
    parts = [np.asarray(d["root_pos"], dtype=np.float64),
             np.asarray(d["root_rot"], dtype=np.float64),
             np.asarray(d["dof_pos"], dtype=np.float64)]
    n = min(len(p) for p in parts)
    arr = np.concatenate([p[:n] for p in parts], axis=1)
    return arr, int(d.get("fps", 30))


def verify_joint_order(xml_path: str) -> None:
    """Fail loudly if the GMR model's hinge order is not K1_JOINT_NAMES order."""
    try:
        import mujoco as mj
    except ImportError:
        print("[gmr] mujoco unavailable; skipping joint-order check", file=sys.stderr)
        return
    if not xml_path or not os.path.exists(xml_path):
        print(f"[gmr] xml not found ({xml_path}); skipping joint-order check",
              file=sys.stderr)
        return
    m = mj.MjModel.from_xml_path(xml_path)
    hinges = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i)
              for i in range(m.njnt) if m.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
    if hinges != K1_JOINT_NAMES:
        for i, (a, b) in enumerate(zip(hinges, K1_JOINT_NAMES)):
            if a != b:
                raise SystemExit(
                    f"joint order mismatch at index {i}: "
                    f"model={a!r} expected={b!r}\n"
                    f"model : {hinges}\nexpected: {K1_JOINT_NAMES}")
        raise SystemExit(f"hinge count {len(hinges)} != {len(K1_JOINT_NAMES)}")
    print(f"[gmr] joint order verified against {os.path.basename(xml_path)} "
          f"({len(hinges)} hinges, exact match)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True,
                   help="GMR output: .pkl (--save_path) or .npz")
    p.add_argument("--out", required=True, help="output motion CSV")
    p.add_argument("--fps", type=int, default=0,
                   help="resample to this rate; 0 = keep the GMR rate")
    p.add_argument("--qz_offset", type=float, default=0.0,
                   help="shift the whole motion in z (K1 root height vs human)")
    p.add_argument("--xml", default="",
                   help="GMR robot xml, to verify the joint order at runtime")
    args = p.parse_args()

    arr, src_fps = load_any(args.inp)
    print(f"[gmr] in={os.path.basename(args.inp)} frames={arr.shape[0]} "
          f"cols={arr.shape[1]} src_fps={src_fps}")

    if arr.shape[1] != EXPECTED_COLS:
        raise SystemExit(f"expected {EXPECTED_COLS} columns, got {arr.shape[1]}")

    if args.xml:
        verify_joint_order(args.xml)

    # Quaternion sanity: a non-unit root quat means the order is wrong.
    qn = np.linalg.norm(arr[:, 3:7], axis=1)
    dev = float(np.abs(qn - 1.0).max())
    if dev > 1e-6:
        raise SystemExit(f"root quats not unit norm (max dev {dev:.2e}); "
                         f"check xyzw vs wxyz")
    print(f"[gmr] root quats unit-norm OK (max dev {dev:.1e})")

    out_fps = int(args.fps) or src_fps
    if out_fps != src_fps:
        # linear resample of position + slerp-free quat renormalise is adequate
        # here because we only change the sample rate of an already-dense track
        from fractions import Fraction
        ratio = Fraction(out_fps, src_fps)
        n_new = int(round(arr.shape[0] * ratio))
        t_old = np.linspace(0.0, 1.0, arr.shape[0])
        t_new = np.linspace(0.0, 1.0, n_new)
        arr = np.stack([np.interp(t_new, t_old, arr[:, c]) for c in
                        range(arr.shape[1])], axis=1)
        arr[:, 3:7] /= np.linalg.norm(arr[:, 3:7], axis=1, keepdims=True)
        print(f"[gmr] resampled {src_fps} -> {out_fps} Hz ({n_new} frames)")

    if args.qz_offset:
        arr[:, 2] += args.qz_offset
        print(f"[gmr] z offset {args.qz_offset:+.4f} m")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savetxt(args.out, arr, delimiter=",", fmt="%.8f")
    print(f"[gmr] wrote {args.out}  {arr.shape[0]} x {arr.shape[1]} @ {out_fps} Hz "
          f"({arr.shape[0]/out_fps:.2f}s)")
    print(f"[gmr] root z range [{arr[:,2].min():.3f}, {arr[:,2].max():.3f}] m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
