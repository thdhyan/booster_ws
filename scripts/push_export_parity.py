#!/usr/bin/env python3
"""Offline export-parity test for the Track B frozen base (CPU only, no Kit).

Q: does models/k1_partialctrl_base.pt (shipped TorchScript) reproduce the
runner actor checkpoint on the SAME obs?

Result (2026-09-24, Run-10 model_2999): BIT-EXACT — shipped .pt == eager
runner actor == fresh as_jit(), max|diff| = 0.0 on both canonical (exact P6
reset) and partial-style jittered obs. The export path is exonerated; the
P6 fall is an env-side difference (the raw action scale ±5-6 is the policy's
normal output everywhere — it walks in the partial env with the same
outputs). Hunt the difference IN-ENV (dynamics/actuation/obs beyond step 0),
never in the export.

Usage:
  ~/Projects/IsaacLab-ea/.venv/bin/python scripts/push_export_parity.py \
      --shipped models/k1_partialctrl_base.pt \
      --ckpt /tmp/opencode/run10_final.pt [--ckpt /tmp/opencode/run11_final.pt]

(GPU box: run inside the Isaac image the same way; needs only torch+rsl_rl.)
"""
import argparse

import torch

OBS_DIM, ACT_DIM = 68, 14
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
    if kind == "partial_reset":
        g = torch.Generator().manual_seed(0)
        o[:, 9:12] = torch.tensor([0.8, 0.0, 0.0]).repeat(n, 1)    # cmd like a walking env
        o[:, 12:24] = (torch.rand(n, 12, generator=g) - 0.5) * 0.4  # leg_pos_rel jitter (reset scale 0.5-1.5)
        o[:, 24:36] = (torch.rand(n, 12, generator=g) - 0.5) * 3.0  # leg_vel jitter
        o[:, 36:38] = (torch.rand(n, 2, generator=g) - 0.5) * 0.1    # head pos
        o[:, 38:46] = (torch.rand(n, 8, generator=g) - 0.5) * 1.0    # arm pos (randomized)
        o[:, 46:54] = (torch.rand(n, 8, generator=g) - 0.5) * 2.0    # arm vel
        o[:, 54:68] = (torch.rand(n, 14, generator=g) - 0.5) * 1.0   # last action
    return o


def describe(name, t):
    print(f"[parity]   {name:28s} absmax={t.abs().max().item():8.3f} "
          f"mean={t.mean().item():+8.4f} legs_absmax={t[:, :12].abs().max().item():.3f} "
          f"v0={t[0, :6].tolist()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shipped", default="models/k1_partialctrl_base.pt")
    p.add_argument("--ckpt", action="append", default=[])
    args = p.parse_args()

    torch.manual_seed(0)
    shipped = torch.jit.load(args.shipped, map_location=DEV)
    shipped.eval()
    ckpts = args.ckpt or ["/tmp/opencode/run10_final.pt"]

    first = build_actor(ckpts[0]) if ckpts else None
    jit = torch.jit.script(first.as_jit()) if first else None
    jit.eval() if jit is not None else None

    for kind in ("canonical", "partial_reset"):
        obs = make_obs(kind)
        print(f"[parity] obs={kind}:")
        out_ship = shipped(obs)
        describe("shipped .pt", out_ship)
        if first is not None:
            out_eager = first({"policy": obs})
            out_jit = jit(obs)
            describe("runner eager", out_eager)
            describe("runner fresh as_jit", out_jit)
            print(f"[parity]   max|ship-eager|={(out_ship - out_eager).abs().max().item():.2e}  "
                  f"max|ship-jit|={(out_ship - out_jit).abs().max().item():.2e}")
        # extra checkpoints: reference only
        for extra in ckpts[1:]:
            m = build_actor(extra)
            describe(f"{extra} eager", m({"policy": obs}))

    print("PARITY_TEST_DONE")


main()
