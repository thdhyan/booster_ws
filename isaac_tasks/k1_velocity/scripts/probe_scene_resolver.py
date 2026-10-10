"""Probe: why does the push scene build fail?

Runs INSIDE the container and prints:
  * the container's own resolver dispatch region (the file and branch order actually
    loaded, not the one in the laptop source tree);
  * the type of every field on the push scene cfg;
  * the type of the same field on the velocity scene, which is known to work;
  * a bisect: build the play env as-is, then again with scene.ground_patch deleted.

ORDERING -- this is the part that matters and it cost three failed runs. Isaac/Carbonite
requires SimulationApp to be instantiated BEFORE any isaaclab import; importing first
crashes Kit during startup, which shows up only as

    [crash] 'appState' = 'startup'
    [crash] 'UptimeSeconds' = '1'

with a full crash dump and no Python traceback. An earlier version of this probe did the
imports at module level and created the launcher at the bottom, so it never got as far as
printing anything. play_record.py gets this right; mirror it exactly.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe the push scene resolver.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_known_args()[0]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- everything below runs after SimulationApp exists -------------------------
import inspect  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.scene.interactive_scene as isc  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import booster_train.tasks  # noqa: F401,E402
import k1_velocity.tasks.velocity  # noqa: F401,E402
import k1_velocity.tasks.push  # noqa: F401,E402
from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

SRC = inspect.getsourcefile(isc)
src = open(SRC).read().splitlines()
print("PROBE file:", SRC)

start = next(i for i, l in enumerate(src) if "isinstance(asset_cfg, TerrainImporterCfg)" in l)
print("PROBE resolver dispatch region:")
for i in range(start, min(start + 90, len(src))):
    line = src[i]
    if ("isinstance(asset_cfg" in line or "raise ValueError" in line
            or "self._extras" in line or "self._sensors[" in line
            or "for asset_name" in line or line.strip().startswith("continue")):
        print(f"PROBE   {i+1}: {line.strip()[:100]}")

for task in ("Isaac-Velocity-Rough-K1-v0", "Isaac-Push-SG-K1-Play-v0"):
    try:
        c = load_cfg_from_registry(task, "env_cfg_entry_point")
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE {task}: cfg load failed {type(exc).__name__}: {exc}")
        continue
    print(f"PROBE {task}: scene fields")
    for name, val in vars(c.scene).items():
        if name.startswith("_") or callable(val) or val is None:
            continue
        mark = ""
        if isinstance(val, SensorBaseCfg):
            mark = "  <-- SensorBaseCfg"
        print(f"PROBE   {name}: {type(val).__name__}{mark}")


def attempt(label, mutate=None):
    c = load_cfg_from_registry("Isaac-Push-SG-K1-Play-v0", "env_cfg_entry_point")
    c.scene.num_envs = 2
    if mutate is not None:
        mutate(c)
    try:
        e = gym.make("Isaac-Push-SG-K1-Play-v0", cfg=c).unwrapped
        print(f"PROBE {label}: BUILD_OK")
        e.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE {label}: {type(exc).__name__}: {str(exc)[:120]}")
        for line in traceback.format_exc().splitlines()[-5:]:
            print("PROBE   tb:", line.strip()[:120])
        return False


ok = attempt("as-is")
if not ok:
    def drop_patch(c):
        if hasattr(type(c.scene), "ground_patch"):
            delattr(type(c.scene), "ground_patch")
            print("PROBE removed scene.ground_patch")
    ok = attempt("without ground_patch", drop_patch)
if not ok:
    def drop_scanner(c):
        if hasattr(type(c.scene), "height_scanner"):
            delattr(type(c.scene), "height_scanner")
            print("PROBE removed scene.height_scanner")
    ok = attempt("without height_scanner", drop_scanner)

print("PROBE_DONE")
simulation_app.close()
