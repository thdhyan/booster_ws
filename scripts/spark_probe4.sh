#!/bin/bash
# Quick batch probe (NO Kit boot, ~60 s): install all four of our extensions,
# import every module of our stack, and instantiate env cfgs against the image's
# isaaclab. Surfaces ALL module-level version drift in one pass instead of one
# failure per S1 cycle.
set +e
PY=/isaac-sim/python.sh
REPO=/workspace/booster_ws
cd "$REPO" || { echo "PROBE4_CD_FAIL"; exit 1; }

echo "== installs:"
for p in isaac_tasks/booster_train_ref/source/booster_train \
         isaac_tasks/k1_velocity \
         src/k1_description/assets \
         isaac_tasks/k1_kicking \
         isaac_tasks/k1_head_tracking; do
  "$PY" -m pip install --no-deps --no-build-isolation -q -e "$REPO/$p" > /tmp/pip_$$.log 2>&1
  rc=$?
  [ "$rc" -ne 0 ] && tail -4 /tmp/pip_$$.log
  echo "pip $p rc=$rc"
done

"$PY" - <<'PYEOF'
import importlib
import traceback

results = []

def probe(label, fn):
    try:
        fn()
        results.append((label, "OK", ""))
    except Exception as e:
        tb = " | ".join(
            ln.strip() for ln in traceback.format_exc().splitlines()[-4:]
        )
        results.append((label, "FAIL", f"{type(e).__name__}: {e} @ {tb}"))

# kicking / head_tracking are source-dir (non-pip) trees: probe via sys.path.
import sys
sys.path.insert(0, "/workspace/booster_ws/isaac_tasks/k1_kicking/source")
sys.path.insert(0, "/workspace/booster_ws/isaac_tasks/k1_head_tracking/source")

mods = [
    "booster_train",
    "booster_train.assets.robots.booster",
    "booster_train.tasks",
    "k1_velocity.tasks.velocity",
    "k1_velocity.tasks.velocity.velocity_env_cfg",
    "k1_velocity.tasks.basic",
    "k1_velocity.tasks.basic.basic_env_cfg",
    "k1_velocity.tasks.basic.basic_env_student",
    "k1_velocity.tasks.kick",
    "k1_velocity.tasks.kick.mdp",
    "k1_kicking.tasks.kicking.kicking_env_cfg",
    "k1_head_tracking.tasks.head_tracking.head_tracking_env_cfg",
]
for m in mods:
    probe(f"import {m}", lambda m=m: importlib.import_module(m))

def inst_student_cfg():
    from k1_velocity.tasks.basic import basic_env_cfg as b
    b.K1BasicTeacherEnvCfg()
probe("cfg K1BasicTeacherEnvCfg()", inst_student_cfg)

def inst_velocity_cfg():
    from k1_velocity.tasks.velocity import velocity_env_cfg as v
    for name in dir(v):
        obj = getattr(v, name)
        if isinstance(obj, type) and name.endswith("EnvCfg"):
            obj()
            return
    raise RuntimeError("no EnvCfg class found")
probe("cfg velocity EnvCfg (first *EnvCfg)", inst_velocity_cfg)

for label, status, msg in results:
    print(f"[{status}] {label}" + (f" -> {msg}" if msg else ""))
print(f"PROBE4_FAILS={sum(1 for _, s, _ in results if s == 'FAIL')}")
print("PROBE4_DONE")
PYEOF
echo "PROBE4_RC=$?"
