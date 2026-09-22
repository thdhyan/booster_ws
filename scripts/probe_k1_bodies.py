"""Probe: spawn K1, dump articulation body/joint names + contact sensor bodies.

Ground truth for velocity_env_cfg.py fixes.
Run from workspace root with venv-isaac python.
"""

from __future__ import annotations

import argparse
import re
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe K1 articulation names")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from booster_train.assets.robots.booster import BOOSTER_K1_CFG  # noqa: E402


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


def main():
    scene_cfg = ProbeSceneCfg(num_envs=1, env_spacing=2.5)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    scene = InteractiveScene(scene_cfg)

    # activate physics + lazy-initialize assets/sensors (as ManagerBasedEnv._init_sim does)
    sim.reset()
    scene.update(dt=0.005)

    robot = scene["robot"]
    sensor = scene["contact"]

    print("=" * 60)
    print(f"ARTICULATION BODY NAMES ({len(robot.body_names)}):")
    print(sorted(robot.body_names))
    print("=" * 60)
    print(f"ARTICULATION JOINT NAMES ({len(robot.joint_names)}):")
    print(sorted(robot.joint_names))

    try:
        scene.reset()
        print("[OK] scene.reset()")
    except Exception as e:
        print(f"[WARN] scene.reset() failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 60)
    print(f"hasattr actuators: {hasattr(robot, 'actuators')}")
    print(f"type(robot): {type(robot)}")
    print(f"robot.__class__.__module__: {robot.__class__.__module__}")

    try:
        print(f"sensor.body_names ({len(sensor.body_names)}): {sorted(sensor.body_names)}")
    except Exception as e:
        print(f"[WARN] sensor.body_names failed: {e}")

    print("=" * 60)

    from pxr import UsdPhysics  # noqa: E402

    import omni.usd  # noqa: E402

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")

    def walk(prim, depth=0, max_depth=4):
        rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        cl = prim.HasAPI(UsdPhysics.CollisionAPI)
        flags = f"{'[RB]' if rb else ''}{'[CL]' if cl else ''}"
        print("  " * depth + f"- {prim.GetName()} [{prim.GetTypeName()}] {flags}")
        if depth < max_depth:
            for child in prim.GetChildren():
                walk(child, depth + 1, max_depth)

    print("PRIM TREE under /World/envs/env_0/Robot:")
    if robot_prim.IsValid():
        walk(robot_prim)
    else:
        print("  INVALID — searching:")
        for p in stage.Traverse():
            if "Robot" in str(p.GetPath()) and p.GetDepth() < 6:
                print(f"  {p.GetPath()} [{p.GetTypeName()}]")

    simulation_app.close()


if __name__ == "__main__":
    main()
