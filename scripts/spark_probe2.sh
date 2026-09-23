#!/bin/bash
# Probe2 (container-side): map cartpole/Ant env ids -> registered RL config entry points.
set +e
/isaac-sim/python.sh - <<'PY' 2>&1 | tail -40
import gymnasium as gym
import isaaclab_tasks
ids = sorted(k for k in gym.envs.registry if "Cartpole" in k or "Ant" in k)
print("ALL_IDS:", ids)
for k in ids:
    try:
        s = gym.spec(k)
        cfgs = [a for a in dir(s) if a.endswith("_cfg_entry_point")]
        print(k, "->", cfgs)
    except Exception as e:
        print(k, "SPEC_ERR", repr(e))
print("PROBE2_DONE")
PY
echo "PROBE2_RC=$?"
