#!/bin/bash
# Probe5 v2: build an InteractiveScene exactly like training (robot + ground +
# contact_forces) and print: articulation body_names, sensor body_names, and the
# full prim tree under /Robot with RigidBodyAPI flags. Explains sensor=('Trunk',).
# python -u so a crash can't hide buffered output.
set +e
PY=/isaac-sim/python.sh
REPO=/workspace/booster_ws
cd "$REPO" || exit 1

for p in isaac_tasks/booster_train_ref/source/booster_train \
         isaac_tasks/k1_velocity \
         src/k1_description/assets; do
  "$PY" -m pip install --no-deps --no-build-isolation -q -e "$REPO/$p" > /dev/null 2>&1
done

echo "== image contact sensor body selection:"
sed -n '90,140p' /workspace/isaaclab/source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor.py

"$PY" -u - <<'PYEOF' 2>&1 | grep -E "^(==|PRIM|P5|URDF|Traceback|  File|[A-Za-z]*Error)" | head -120
import argparse

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

print("== scene build")
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))


@configclass
class P5Scene(InteractiveSceneCfg):
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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


scene = InteractiveScene(P5Scene(num_envs=1, env_spacing=2.5))
sim.reset()
scene.update(0.005)

robot = scene["robot"]
sensor = scene["contact_forces"]
print("P5ROBOT_BODIES", robot.body_names)
print("P5ROBOT_JOINTS", robot.joint_names[:6], "...", len(robot.joint_names))
print("P5SENSOR_BODY_NAMES", sensor.body_names)

stage = omni.usd.get_context().get_stage()
for p in Usd.PrimRange(stage.GetPseudoRoot()):
    path = str(p.GetPath())
    if "/Robot" not in path:
        continue
    rb = p.HasAPI(UsdPhysics.RigidBodyAPI)
    print(f"PRIM {path} type={p.GetTypeName()} rb={rb}")
print("P5DONE")
app.close()
PYEOF
echo "PROBE5_RC=$?"
