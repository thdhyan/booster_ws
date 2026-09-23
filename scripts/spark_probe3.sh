#!/bin/bash
# Probe3 (container-side): how does this Isaac Lab image wire RL agent configs to tasks?
# Answers: where train.py's "Could not find configuration" reads entry points from,
# whether rsl_rl configs exist at all, and what Isaac-Cartpole-v0 registers.
set +e
echo "== scripts/reinforcement_learning:"
ls /workspace/isaaclab/scripts/reinforcement_learning/
echo "== Isaac-Cartpole-v0 registration:"
grep -rn "Isaac-Cartpole-v0" /workspace/isaaclab/source/isaaclab_tasks --include=*.py | head -5
echo "== rsl_rl_cfg_entry_point in cartpole dir:"
grep -rn "rsl_rl_cfg_entry_point" /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/cartpole/ 2>/dev/null | head -5
echo "== error origin:"
grep -rn "Could not find configuration for the environment" /workspace/isaaclab/source --include=*.py | head -3
echo "== python introspection:"
/isaac-sim/python.sh - <<'PY' 2>&1 | tail -45
import gymnasium as gym
import isaaclab_tasks

spec = gym.spec("Isaac-Cartpole-v0")
print("VARS:", {k: str(v)[:90] for k, v in vars(spec).items()})

try:
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    keys = sorted(cs.store.keys())
    print("CS_TOTAL:", len(keys))
    print("CS_RL_KEYS:", [k for k in keys if "ppo" in k.lower() or "rsl" in k.lower()
                          or "rl_games" in k.lower() or "a2c" in k.lower()][:40])
except Exception as e:
    print("CS_FAIL", repr(e))

try:
    import inspect
    from isaaclab_tasks.utils.hydra import register_task
    src = inspect.getsource(register_task)
    print("REGISTER_TASK_SRC:")
    print("\n".join(src.splitlines()[:55]))
except Exception as e:
    print("INTROSPECT_FAIL", repr(e))
PY
echo "PROBE3_RC=$?"
