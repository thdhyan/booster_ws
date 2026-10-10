"""Probe v3: the push scene build fails, but the CFG is provably fine.

Probe v2 already ruled out the two obvious explanations:
  * the container's resolver branch order is correct (SensorBaseCfg at 947 precedes
    AssetBaseCfg at 981);
  * isinstance(cfg.scene.height_scanner, SensorBaseCfg) is True for BOTH the velocity
    scene that works and the push scene that fails.

So the cfg object is well-typed and the chain is in the right order, yet building the
env raises

    ValueError: Unknown asset config type for height_scanner: RayCasterCfg(...)

which can only mean the object reaching the `raise` is not the one in the cfg. This
dumps the resolver source around the raise, the type of every scene field, and then
bisects: build the env as-is, and if that fails, build it again with ground_patch
removed. If removing ground_patch fixes it, the patch insertion is the cause and the
resolver is being handed something other than the declared cfg.
"""
import inspect
import traceback

import isaaclab.scene.interactive_scene as isc

SRC = inspect.getsourcefile(isc)
src = open(SRC).read().splitlines()
print("PROBE3 file:", SRC)

# the exact region that decides asset dispatch, including the raise
start = next(i for i, l in enumerate(src) if "isinstance(asset_cfg, TerrainImporterCfg)" in l)
print("PROBE3 resolver dispatch region:")
for i in range(start, min(start + 90, len(src))):
    line = src[i]
    if ("isinstance(asset_cfg" in line or "raise ValueError" in line
            or "_extras" in line or "_sensors[" in line or "for asset_name" in line
            or "continue" in line):
        print(f"PROBE3   {i+1}: {line.strip()[:100]}")

import isaaclab_tasks  # noqa: F401,E402
import booster_train.tasks  # noqa: F401,E402
import k1_velocity.tasks.velocity  # noqa: F401,E402
import k1_velocity.tasks.push  # noqa: F401,E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

cfg = load_cfg_from_registry("Isaac-Push-SG-K1-Play-v0", "env_cfg_entry_point")
print("PROBE3 scene fields:")
for name, val in vars(cfg.scene).items():
    if name.startswith("_") or callable(val) or val is None:
        continue
    print(f"PROBE3   {name}: {type(val).__name__}")

import gymnasium as gym  # noqa: E402


def attempt(label, mutate=None):
    c = load_cfg_from_registry("Isaac-Push-SG-K1-Play-v0", "env_cfg_entry_point")
    c.scene.num_envs = 2
    if mutate is not None:
        mutate(c)
    try:
        e = gym.make("Isaac-Push-SG-K1-Play-v0", cfg=c).unwrapped
        print(f"PROBE3 {label}: BUILD_OK")
        e.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE3 {label}: {type(exc).__name__}: {str(exc)[:110]}")
        tb = traceback.format_exc().splitlines()
        for line in tb[-6:]:
            print("PROBE3   tb:", line.strip()[:110])
        return False


from isaaclab.app import AppLauncher  # noqa: E402
app = AppLauncher({"headless": True}).app

ok = attempt("as-is")
if not ok:
    def drop_patch(c):
        for attr in ("ground_patch",):
            if hasattr(c.scene, attr):
                delattr(type(c.scene), attr)
                print(f"PROBE3 removed scene.{attr}")
    ok = attempt("without ground_patch", drop_patch)

print("PROBE3_DONE")
app.close()
