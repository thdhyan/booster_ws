#!/usr/bin/env python3
"""GMR driver: BVH (SOMA / LAFAN1) -> Booster K1 motion CSV.

GMR's own `scripts/bvh_to_robot.py` restricts `--robot` to a list that omits
`booster_k1`, even though `general_motion_retargeting/params.py` ships a full
`booster_k1` config (K1_serial.xml + ik_configs/smplx_to_k1.json). This driver
calls the library directly and writes the Booster motion CSV contract that
`booster_train` expects:

    col 0-2   root position  x, y, z        [m]
    col 3-6   root quaternion x, y, z, w    [xyzw]   <- Booster order
    col 7-28  22 joint positions            [rad]   K1_JOINT_NAMES order
    sampled at 50 Hz

Run inside the GMR checkout with an interpreter that has mink + mujoco + torch.

  python gmr_bvh_to_k1_csv.py --bvh dance.bvh --out dance.csv
  python gmr_bvh_to_k1_csv.py --bvh dance.bvh --out dance.csv --human_height 1.75
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# booster_assets.motions.K1_JOINT_NAMES — the CSV column order.
K1_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", required=True)
    p.add_argument("--out", required=True, help="output motion CSV")
    p.add_argument("--robot", default="booster_k1")
    p.add_argument("--format", default="lafan1", choices=["lafan1", "nokov"])
    p.add_argument("--human_height", type=float, default=0.0,
                   help="0 = read from the BVH file")
    p.add_argument("--fps", type=int, default=50, help="output sample rate")
    p.add_argument("--qz_offset", type=float, default=0.0,
                   help="shift the whole motion in z (K1 vs human floor height)")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--no_viewer", action="store_true", default=True)
    args = p.parse_args()

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, human_height = load_bvh_file(args.bvh, format=args.format)
    if args.human_height > 0:
        human_height = args.human_height
    print(f"[gmr] bvh={os.path.basename(args.bvh)} frames={len(frames)} "
          f"human_height={human_height:.3f} target={args.robot}")

    retargeter = GMR(src_human=f"bvh_{args.format}", tgt_robot=args.robot,
                     actual_human_height=human_height)

    rows = []
    n = len(frames)
    for i in range(n):
        data = frames[i]
        if args.loop:
            data = frames[i % n]
        retargeter.retarget(data)
        q = np.asarray(retargeter.rob_q, dtype=np.float64)
        # GMR exposes the root transform in the same layout MuJoCo uses:
        # (x, y, z, qx, qy, qz, qw)
        rows.append(q[:7].tolist() + q[7:7 + len(K1_JOINT_NAMES)].tolist())
        if (i + 1) % 100 == 0:
            print(f"  retargeted {i+1}/{n}", flush=True)

    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[1] != 7 + len(K1_JOINT_NAMES):
        raise RuntimeError(
            f"expected {7 + len(K1_JOINT_NAMES)} columns, got {arr.shape[1]}")

    # Resample to the requested output rate.
    src_fps = float(getattr(args, "src_fps", 0) or 30)
    step = max(1, int(round(src_fps / args.fps)))
    if step > 1:
        arr = arr[::step]
    if args.qz_offset:
        arr[:, 2] += args.qz_offset

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savetxt(args.out, arr, delimiter=",")
    print(f"[gmr] wrote {args.out}  {arr.shape[0]} frames x {arr.shape[1]} cols "
          f"@ {args.fps} Hz")
    print(f"[gmr] root z range [{arr[:, 2].min():.3f}, {arr[:, 2].max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
