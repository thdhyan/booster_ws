#!/usr/bin/env python3
"""Offline export-parity test for the Track B frozen base (CPU only, no Kit).

Q: does the shipped TorchScript base (models/k1_*.pt) reproduce the runner
actor checkpoint on the SAME obs?

Result (2026-09-24, Run-10 model_2999, partial layout): BIT-EXACT — shipped
.pt == eager runner actor == fresh as_jit(), max|diff| = 0.0 on both canonical
(jitter-free) and partial-style jittered obs. The export path is exonerated;
environment-side differences are hunted IN-ENV, never in the export.

Layouts (blocker: the frozen base changed in Phase 1):
  --layout partial  68-dim obs  -> 14 targets  (legacy Run-11 partial base)
  --layout squat    236-dim obs -> 12 targets  (Phase-1 squat teacher; obs
                     order = lin 3 | ang 3 | grav 3 | cmd 4 | leg_pos 12 |
                     leg_vel 12 | last action 12 | height_scan 187)
  --obs_dim/--act_dim override the per-layout defaults.

Gates for automation: prints PARITY_TEST_DONE always, then
PARITY_MAX_DIFF=<x> and PARITY_OK / PARITY_FAIL (threshold 1e-4).

Usage:
  ~/Projects/IsaacLab-ea/.venv/bin/python scripts/push_export_parity.py \
      --shipped models/k1_partialctrl_base.pt \
      --ckpt /tmp/opencode/run10_final.pt [--ckpt ...] [--layout partial]

  # zz-bw chain (inside the Isaac image, via K1_TRAIN_SCRIPT):
  #   scripts/push_export_parity.py --shipped models/k1_squat_base.pt \
  #       --ckpt logs/rsl_rl/k1_squat_teacher/<run>/model_XXXX.pt --layout squat

(GPU box: run inside the Isaac image the same way; needs only torch+rsl_rl.)
"""
import argparse
import os

import torch

OBS_DIM, ACT_DIM = 68, 14
LAYOUT = "partial"
DEV = "cpu"


def build_actor(ckpt_path):
    from rsl_rl.models.mlp_model import MLPModel

    d = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    sd = d["actor_state_dict"]
    model = MLPModel(
        obs={"policy": torch.zeros(1, OBS_DIM, device=DEV)},
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        obs_set="actor",
        output_dim=ACT_DIM,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 1.0,
        },
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[parity] built runner actor from {ckpt_path} (iter={d.get('iter')})")
    return model


def make_obs(kind: str, n: int = 8) -> torch.Tensor:
    o = torch.zeros(n, OBS_DIM, device=DEV)
    o[:, 6:9] = torch.tensor([0.0, 0.0, -1.0])  # projected gravity
    g = torch.Generator().manual_seed(0)
    if LAYOUT == "squat":
        if kind == "squat_reset":
            o[:, 0:3] = (torch.rand(n, 3, generator=g) - 0.5) * 0.6      # lin vel
            o[:, 3:6] = (torch.rand(n, 3, generator=g) - 0.5) * 1.2      # ang vel
            o[:, 9:12] = torch.tensor([0.4, 0.0, 0.3]).repeat(n, 1)      # cmd vx/vy/wz
            o[:, 12] = 0.48                                              # H* (0.40-0.55 m)
            o[:, 13:25] = (torch.rand(n, 12, generator=g) - 0.5) * 0.4   # leg_pos_rel
            o[:, 25:37] = (torch.rand(n, 12, generator=g) - 0.5) * 3.0   # leg_vel
            o[:, 37:49] = (torch.rand(n, 12, generator=g) - 0.5) * 1.0   # last action
            o[:, 49:236] = (torch.rand(n, 187, generator=g) - 0.5) * 0.6 # height scan (clip +-1)
    elif kind == "partial_reset":
        o[:, 9:12] = torch.tensor([0.8, 0.0, 0.0]).repeat(n, 1)    # cmd like a walking env
        o[:, 12:24] = (torch.rand(n, 12, generator=g) - 0.5) * 0.4  # leg_pos_rel jitter (reset scale 0.5-1.5)
        o[:, 24:36] = (torch.rand(n, 12, generator=g) - 0.5) * 3.0  # leg_vel jitter
        o[:, 36:38] = (torch.rand(n, 2, generator=g) - 0.5) * 0.1   # head pos
        o[:, 38:46] = (torch.rand(n, 8, generator=g) - 0.5) * 1.0   # arm pos (randomized)
        o[:, 46:54] = (torch.rand(n, 8, generator=g) - 0.5) * 2.0   # arm vel
        o[:, 54:68] = (torch.rand(n, 14, generator=g) - 0.5) * 1.0  # last action
    return o


def describe(name, t):
    print(f"[parity]   {name:28s} absmax={t.abs().max().item():8.3f} "
          f"mean={t.mean().item():+8.4f} legs_absmax={t[:, :12].abs().max().item():.3f} "
          f"v0={t[0, :6].tolist()}")


def main():
    global OBS_DIM, ACT_DIM, LAYOUT
    p = argparse.ArgumentParser()
    p.add_argument("--shipped", default="models/k1_partialctrl_base.pt")
    p.add_argument("--ckpt", action="append", default=[],
                   help="runner checkpoint to compare against (repeatable)")
    p.add_argument("--layout", choices=["partial", "squat"], default="partial",
                   help="frozen-base obs layout: partial = Run-11 (68->14), "
                        "squat = Phase-1 teacher (236->12)")
    p.add_argument("--obs_dim", type=int, default=None, help="override layout default")
    p.add_argument("--act_dim", type=int, default=None, help="override layout default")
    p.add_argument("--threshold", type=float, default=1e-4,
                   help="max|diff| below which PARITY_OK is printed")
    args = p.parse_args()

    LAYOUT = args.layout
    OBS_DIM = args.obs_dim or (68 if LAYOUT == "partial" else 236)
    ACT_DIM = args.act_dim or (14 if LAYOUT == "partial" else 12)
    kinds = ("canonical", "partial_reset") if LAYOUT == "partial" else ("canonical", "squat_reset")

    torch.manual_seed(0)
    shipped = torch.jit.load(args.shipped, map_location=DEV)
    shipped.eval()
    print(f"[parity] shipped={args.shipped} layout={LAYOUT} obs={OBS_DIM} act={ACT_DIM}")
    ckpts = [c for c in args.ckpt if os.path.isfile(c)]
    missing = [c for c in args.ckpt if not os.path.isfile(c)]
    for m in missing:
        print(f"[parity] WARNING: ckpt not found, skipped: {m}")
    if not ckpts and os.path.isfile("/tmp/opencode/run10_final.pt"):
        ckpts = ["/tmp/opencode/run10_final.pt"]

    first = build_actor(ckpts[0]) if ckpts else None
    jit = torch.jit.script(first.as_jit()) if first else None
    jit.eval() if jit is not None else None

    max_diff = 0.0
    for kind in kinds:
        obs = make_obs(kind)
        print(f"[parity] obs={kind}:")
        out_ship = shipped(obs)
        describe("shipped .pt", out_ship)
        if first is not None:
            out_eager = first({"policy": obs})
            out_jit = jit(obs)
            describe("runner eager", out_eager)
            describe("runner fresh as_jit", out_jit)
            d1 = (out_ship - out_eager).abs().max().item()
            d2 = (out_ship - out_jit).abs().max().item()
            max_diff = max(max_diff, d1, d2)
            print(f"[parity]   max|ship-eager|={d1:.2e}  "
                  f"max|ship-jit|={d2:.2e}")
        # extra checkpoints: reference only
        for extra in ckpts[1:]:
            m = build_actor(extra)
            describe(f"{extra} eager", m({"policy": obs}))

    print("PARITY_TEST_DONE")
    if first is None:
        print(f"PARITY_MAX_DIFF=n/a (no reference ckpt)")
        print("PARITY_FAIL")
    else:
        print(f"PARITY_MAX_DIFF={max_diff:.3e}")
        print("PARITY_OK" if max_diff < args.threshold else "PARITY_FAIL")


main()
