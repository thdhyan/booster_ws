"""Report P3 checkpoint inventory: iteration, distribution std parameterisation, weight health."""
import glob
import os
import sys

import torch

pattern = sys.argv[1] if len(sys.argv) > 1 else "logs/rsl_rl/p3_head_track/*/model_*.pt"
rows = []
for path in glob.glob(pattern):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    actor = ckpt["actor_state_dict"]
    std_keys = [k for k in actor if "std" in k]
    std_val = {k: float(actor[k].float().mean()) for k in std_keys}
    bad = [k for grp in ("actor_state_dict", "critic_state_dict")
           for k, v in ckpt[grp].items() if not torch.isfinite(v).all()]
    rows.append((
        float(os.path.getmtime(path)),
        os.path.basename(os.path.dirname(path)),
        os.path.basename(path),
        ckpt.get("iter"),
        std_keys,
        std_val,
        bad or "finite",
    ))

for _, run, name, it, keys, vals, bad in sorted(rows):
    print(f"{run:38s} {name:16s} iter={it} std={keys} mean={vals} {bad}")
