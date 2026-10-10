"""Probe: why does the push scene's height_scanner fall through the resolver?

Runs INSIDE the container and prints, from the container's own IsaacLab:
  * the branch order of the scene resolver's isinstance chain
  * the MRO of the height_scanner cfg actually built by K1PushSceneCfg
  * whether isinstance() agrees with the resolver's SensorBaseCfg class
  * the same for the v3 velocity scene, which is known to work

The v3-vs-SG comparison is the point: both declare the same RayCasterCfg, so if one
resolves and the other does not, the difference is in how the cfg object was built, not
in the resolver.
"""
import inspect
import sys
import traceback

import isaaclab.scene.interactive_scene as isc

SRC = inspect.getsourcefile(isc)
print("PROBE isaaclab scene:", SRC)
try:
    src = open(SRC).read()
    order = [(i + 1, l.strip()) for i, l in enumerate(src.splitlines())
             if "isinstance(asset_cfg" in l]
    print("PROBE resolver branch order:")
    for ln, l in order:
        print(f"PROBE   {ln}: {l[:90]}")
except Exception as exc:
    print("PROBE could not read resolver:", exc)

try:
    from isaaclab.sensors import RayCasterCfg
    from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
    print("PROBE RayCasterCfg module:", RayCasterCfg.__module__)
    print("PROBE SensorBaseCfg module:", SensorBaseCfg.__module__)
    print("PROBE RayCasterCfg is SensorBaseCfg:", issubclass(RayCasterCfg, SensorBaseCfg))
    print("PROBE MRO:", [c.__name__ for c in RayCasterCfg.__mro__][:6])
except Exception as exc:
    print("PROBE sensor import failed:", type(exc).__name__, exc)
    traceback.print_exc()

# now build the actual scene cfg and inspect the instance the resolver sees
try:
    import isaaclab_tasks  # noqa: F401
    import booster_train.tasks  # noqa: F401
    import k1_velocity.tasks.velocity  # noqa: F401
    import k1_velocity.tasks.push  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    for task in ("Isaac-Velocity-Rough-K1-v0", "Isaac-Push-SG-K1-Play-v0"):
        try:
            cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
            hs = cfg.scene.height_scanner
            print(f"PROBE {task}: height_scanner type={type(hs).__name__} module={type(hs).__module__}")
            print(f"PROBE {task}: is SensorBaseCfg ->", isinstance(hs, __import__(
                "isaaclab.sensors.sensor_base_cfg", fromlist=["SensorBaseCfg"]).SensorBaseCfg))
        except Exception as exc:
            print(f"PROBE {task}: FAILED {type(exc).__name__}: {exc}")
except Exception as exc:
    print("PROBE cfg load failed:", type(exc).__name__, exc)

print("PROBE_DONE")