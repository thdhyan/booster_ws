#!/usr/bin/env python3
"""Convert one rsl-rl Gaussian PPO checkpoint from scalar to log std.

Older checkpoints store ``distribution.std_param`` directly, which can cross
zero during a long PPO run.  New configs use ``distribution.log_std_param``.
Only the actor distribution key changes; optimizer parameter ordering is
unchanged because the distribution owns the same single learnable parameter.
"""
import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
args = parser.parse_args()

src, dst = Path(args.src), Path(args.dst)
ckpt = torch.load(src, map_location="cpu", weights_only=False)
state = ckpt["actor_state_dict"]
if "distribution.std_param" in state:
    std = state.pop("distribution.std_param").float().clamp_min(1.0e-3)
    state["distribution.log_std_param"] = torch.log(std)
elif "distribution.log_std_param" not in state:
    raise KeyError("checkpoint has neither scalar nor log std parameter")

if not all(torch.isfinite(v).all() for v in ckpt["actor_state_dict"].values()):
    raise ValueError("actor checkpoint contains non-finite tensors")
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, dst)
print(f"PPO_LOG_STD_MIGRATION=OK src={src} dst={dst} iter={ckpt.get('iter')}")
