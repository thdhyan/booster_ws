#!/bin/bash
# S0 (container-side): inventory + Kit headless boot + physics step probe on DGX Spark GB10.
# Runs INSIDE nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 via: --entrypoint bash IMG /s0.sh
set +e

# -- resolve a python that actually has isaaclab (entrypoint PATH not available with --entrypoint bash)
PY=""
for C in /isaac-sim/python.sh python3 python /opt/conda/bin/python /usr/local/bin/python3; do
  P="$C"
  case "$C" in
    /*) [ -x "$C" ] || { echo "skip(abs) $C"; continue; } ;;
    *) P="$(command -v "$C" 2>/dev/null || true)"; [ -n "$P" ] || { echo "skip(path) $C"; continue; } ;;
  esac
  if "$P" -c "import isaaclab" >/dev/null 2>&1; then PY="$P"; break; fi
  echo "no-isaaclab: $P"
done
echo "PY_RESOLVED=$PY"
[ -n "$PY" ] || { echo "PY_NOT_FOUND"; exit 1; }

echo "=== A1 LAYOUT"
ls /
echo "--- /workspace:"; ls /workspace 2>/dev/null
echo "--- isaaclab.sh:"; find / -maxdepth 5 -name isaaclab.sh 2>/dev/null | head -3
echo "--- list_envs.py:"; find / -maxdepth 8 -name list_envs.py 2>/dev/null | head -3
echo "--- rsl_rl train scripts:"; find / -maxdepth 9 -path "*rsl_rl*" -name "*.py" 2>/dev/null | grep -E "train|cli" | head -6

echo "=== A2 VERSIONS"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))" 2>&1 | tail -2
"$PY" -m pip list 2>/dev/null | grep -iE "^(isaaclab|isaacsim|rsl|warp|torch|gymnasium)" | head -30

echo "=== A3 ENV_IDS"
"$PY" - <<'PY' 2>&1 | tail -6
import gymnasium as gym
try:
    import isaaclab_tasks
    print("isaaclab_tasks import OK")
except Exception as e:
    print("isaaclab_tasks FAIL", repr(e))
ids = [k for k in gym.envs.registry if "Cartpole" in k or "Ant" in k]
print("sample ids:", sorted(ids)[:12])
PY

echo "=== A4 KIT_BOOT + PHYSICS_STEP"
timeout 900 "$PY" - <<'PY' 2>&1 | tail -80
import os, traceback
print("A4_START cwd=", os.getcwd(), "euid=", os.geteuid())
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("KIT_BOOT_OK")
try:
    from isaaclab.sim import SimulationContext, SimulationCfg
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0))
    print("SIM_CTX_OK")
    import omni.usd
    from pxr import UsdGeom, UsdPhysics, Gf
    stage = omni.usd.get_context().get_stage()
    print("STAGE_OK", stage is not None)
    gnd = UsdGeom.Cube.Define(stage, "/World/Ground")
    gnd.AddTranslateOp().Set(Gf.Vec3f(0, 0, -1))
    gnd.AddScaleOp().Set(Gf.Vec3f(100, 100, 1))
    UsdPhysics.CollisionAPI.Apply(gnd.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.AddTranslateOp().Set(Gf.Vec3f(0, 0, 5))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    print("CUBES_DEFINED")
    sp = stage.GetPrimAtPath("/physicsScene")
    zattr = cube.GetPrim().GetAttribute("xformOp:translate")
    if sp and sp.IsValid():
        g = sp.GetAttribute("physxScene:enableGPUDynamics")
        print("GPU_DYN_VALUE_BEFORE", g.Get() if g else "no-attr")
        gm = sp.GetAttribute("physics:gravityMagnitude")
        print("GRAVITY_MAG", gm.Get() if gm else "no-attr")
    print("SIM_API:", [a for a in dir(sim) if any(k in a.lower() for k in ("pause", "play", "reset", "step"))])
    for fn in ("reset", "play"):
        try:
            getattr(sim, fn)()
            print(fn.upper() + "_OK")
        except Exception:
            print(fn.upper() + "_FAIL:", traceback.format_exc().splitlines()[-1])
    for i in range(120):
        sim.step(render=False)
        if (i + 1) % 30 == 0:
            print("z@%d =" % (i + 1), zattr.Get()[2])
    print("STEP_COUNT:", sim.get_physics_step_count())
    print("IS_PLAYING:", sim.is_playing())
    za = zattr.Get()[2]
    print("STAGE_A_Z_AFTER_120 =", za)
    if sp and sp.IsValid():
        g = sp.GetAttribute("physxScene:enableGPUDynamics")
        print("GPU_DYN_VALUE_END", g.Get() if g else "no-attr")
    print("PHYSICS_MOTION_OK" if za < 4.0 else "PHYSICS_NO_MOTION (z=%.3f)" % za)
    print("PHYSICS_PROBE_DONE")
except Exception:
    print("PHYSICS_FAIL_TRACE:")
    traceback.print_exc()
try:
    app.close()
except Exception:
    print("CLOSE_FAIL_TRACE:")
    traceback.print_exc()
print("A4_DONE")
PY
echo "=== A5 CARTPOLE_RSL_RL"
# Isaac-Cartpole-v0 is the canonical manager-based env; its rsl_rl_cfg_entry_point
# lives in spec.kwargs (camera variants like Depth-v0 only wire rl_games).
CID="Isaac-Cartpole-v0"
echo "CARTPOLE_ID=$CID"
if [ -n "$CID" ]; then
  timeout 600 "$PY" /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task "$CID" --num_envs 16 --headless --max_iterations 3 2>&1 | tail -70
  echo "A5_RC=${PIPESTATUS[0]}"
else
  echo "A5_NO_ENV_ID"
fi
echo "S0_CONTAINER_RC=$?"
