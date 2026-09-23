#!/bin/bash
# Probe6: distinguish WHY the image contact sensor reports only ('Trunk',):
#   H1 = PhysxContactReportAPI applied only to the root body (Isaac 6.0-era importer bug)
#   H2 = bodies below the spawn are USD instance proxies (default walk can't reach them)
# Runs once per INSTANCEABLE=1/0 (0 = candidate fix) and prints per-body flags plus an
# exact replication of the image sensor's body walk (default vs instance-proxy traversal).
set +e
PY=/isaac-sim/python.sh
REPO=/workspace/booster_ws
cd "$REPO" || exit 1

for p in isaac_tasks/booster_train_ref/source/booster_train \
         isaac_tasks/k1_velocity \
         src/k1_description/assets; do
  "$PY" -m pip install --no-deps --no-build-isolation -q -e "$REPO/$p" > /dev/null 2>&1
done

echo "== image activate_contact_sensors application:"
grep -rn "PhysxContactReportAPI\|activate_contact_sensors" /workspace/isaaclab/source/isaaclab/isaaclab/sim/ 2>/dev/null | head -12
echo "== image sensor _initialize_impl walk (293-345):"
sed -n '293,345p' /workspace/isaaclab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py

echo "== MI=$INSTANCEABLE probe:"
"$PY" -u - <<'PYEOF' 2>&1 | grep -E "^(==|MI=|MATCH|BODY|WALK|SENSOR_|P6|Traceback|  File|[A-Za-z]*Error)" | head -140
import argparse
import dataclasses
import os
import re

MI = os.environ.get("INSTANCEABLE", "1") == "1"
print(f"MI={MI}")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
if "headless" not in getattr(AppLauncher, "_APPLAUNCHER_CFG_INFO", {}):
    parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)
app = app_launcher.app

import omni.usd
import isaaclab.sim as sim_utils
from pxr import Usd, UsdPhysics

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from booster_train.assets.robots.booster import BOOSTER_K1_CFG

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))

spawn0 = BOOSTER_K1_CFG.spawn
try:
    spawn = spawn0.replace(make_instanceable=MI)
except AttributeError:
    spawn = dataclasses.replace(spawn0, make_instanceable=MI)


@configclass
class P6Scene(InteractiveSceneCfg):
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", spawn=spawn
    )
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CylinderCfg(
            radius=1.0,
            height=0.1,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
    )
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=["/World/ground"],
    )


scene = InteractiveScene(P6Scene(num_envs=1, env_spacing=2.5))
sim.reset()
scene.update(0.005)

robot = scene["robot"]
sensor = scene["contact_forces"]
print("SENSOR_", sensor.body_names)

stage = omni.usd.get_context().get_stage()
body_names = set(robot.body_names)

# Per-body flags over the proxy-inclusive prim universe.
seen = set()
for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if p.GetName() not in body_names or "/Robot" not in str(p.GetPath()):
        continue
    key = str(p.GetPath())
    if key in seen:
        continue
    seen.add(key)
    schemas = p.GetAppliedSchemas()
    print(
        f"BODY {key} rb={p.HasAPI(UsdPhysics.RigidBodyAPI)} "
        f"cr={'PhysxContactReportAPI' in schemas} proxy={p.IsInstanceProxy()} inst={p.IsInstance()}"
    )

# Exact replication of the image sensor walk (default vs instance-proxy traversal).
from isaaclab.sim.utils.queries import resolve_matching_prims_from_source

matches = resolve_matching_prims_from_source("{ENV_REGEX_NS}/Robot")
name_pat = re.compile(".*")
for root, expr in matches:
    print(f"MATCH root={root.GetPath()} expr={expr} inst={root.IsInstance()}")
    default_hits = [
        p.GetName()
        for p in Usd.PrimRange(root)
        if name_pat.fullmatch(p.GetName()) and "PhysxContactReportAPI" in p.GetAppliedSchemas()
    ]
    proxy_hits = [
        p.GetName()
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if name_pat.fullmatch(p.GetName()) and "PhysxContactReportAPI" in p.GetAppliedSchemas()
    ]
    print("WALK_DEFAULT", default_hits)
    print("WALK_PROXIES", proxy_hits)
print("P6DONE")
app.close()
PYEOF
echo "PROBE6_RC=$?"
