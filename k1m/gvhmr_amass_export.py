"""GVHMR output -> AMASS-style SMPL-X npz, which is what GMR consumes.

GVHMR itself does not export AMASS: `tools/demo/demo.py` `torch.save`s a `pred`
dict to `outputs/demo/<clip>/hmr4d_results.pt`, whose `smpl_params_global` entry
holds the global-frame SMPL-X parameters. GMR wants the AMASS npz layout, so
this reshapes those params into it.

The layout matches AMASS: GVHMR's `endecoder.fk_v2` consumes `body_pose` of
shape `(B, L, 63)`, i.e. 21 joints, which is exactly AMASS `pose_body`. The
hand/eye/jaw blocks are absent from a body-only estimate and are written as
zeros, which is what AMASS does for mocap without finger capture.

Run on the host that has the GVHMR env:

    python gvhmr_amass_export.py --pred outputs/demo/tennis/hmr4d_results.pt \
        --out tennis_amass.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# SMPL-X: 21 body joints -> 63; 15 hand joints per side -> 45 each; jaw 1; eyes 2.
N_BODY = 21
N_HAND = 15
GENDER = "neutral"


def _get(d: dict, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def _f3(v, frames: int | None = None) -> np.ndarray:
    """Coerce a (F,3) or (1,F,3) or (F,1,3) parameter block to (F,3)."""
    a = np.asarray(v, dtype=np.float32)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3:
        a = a.reshape(-1, a.shape[-1])
    if frames is not None and a.shape[0] != frames:
        if a.shape[0] == 1:
            a = np.repeat(a, frames, axis=0)
        else:
            raise ValueError(f"frame mismatch: got {a.shape[0]}, expected {frames}")
    return a


def convert(pred_path: str, out_path: str, fps: int = 30) -> dict:
    import torch
    pred = torch.load(pred_path, "cpu", weights_only=False)
    if not isinstance(pred, dict) or "smpl_params_global" not in pred:
        keys = list(pred)[:8] if isinstance(pred, dict) else type(pred)
        raise SystemExit(
            f"{pred_path}: expected a GVHMR pred dict with "
            f"'smpl_params_global'; got {keys}")

    p = pred["smpl_params_global"]
    print(f"[gvhmr] smpl_params_global keys: {sorted(p)}")

    body = _f3(p["body_pose"])
    frames = body.shape[0]
    go = _f3(_get(p, "global_orient", "root_orient", default=np.zeros((frames, 3))),
             frames)
    tr = _f3(_get(p, "transl", "trans", "translation"), frames)
    betas = np.asarray(_get(p, "betas", default=np.zeros(16)),
                       dtype=np.float32).reshape(-1)
    # GVHMR may carry a reduced shape space; AMASS consumers expect at least 10.
    if betas.size < 10:
        betas = np.pad(betas, (0, 10 - betas.size))

    if body.shape[1] != N_BODY * 3:
        raise SystemExit(
            f"body_pose has {body.shape[1]} values = {body.shape[1] // 3} "
            f"joints; AMASS needs {N_BODY} ({N_BODY * 3} values). GVHMR's "
            f"joint set differs and needs an explicit index map.")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez(
        out_path,
        betas=betas.astype(np.float64),
        gender=GENDER,
        mocap_frame_rate=np.array([fps], dtype=np.float64),
        mocap_time_length=np.array([frames / fps], dtype=np.float64),
        num_betas=np.array([betas.size], dtype=np.int64),
        # AMASS stores the root rotation in axis-angle, not a 3x3 matrix.
        root_orient=go.astype(np.float64),
        trans=tr.astype(np.float64),
        pose_body=body.astype(np.float64),
        pose_jaw=np.zeros((frames, 3)),
        pose_eye=np.zeros((frames, 6)),
        pose_hand=np.zeros((frames, N_HAND * 3 * 2)),
        surface_model_type=np.array(["smplx"], dtype=object),
    )
    print(f"[gvhmr] wrote {out_path}  frames={frames} @ {fps} Hz "
          f"body={body.shape} betas={betas.size}")
    print(f"[gvhmr] trans z range [{tr[:, 2].min():.3f}, {tr[:, 2].max():.3f}] m")
    return {"frames": frames, "fps": fps, "out": out_path}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="GVHMR hmr4d_results.pt")
    p.add_argument("--out", required=True, help="AMASS npz to write")
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()
    convert(a.pred, a.out, a.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
