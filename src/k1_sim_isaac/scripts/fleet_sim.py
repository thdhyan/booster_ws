#!/usr/bin/env python3
"""Isaac Sim multi-K1 fleet with namespaced ROS2 endpoints.

N K1 robots in one InteractiveScene; per-robot ROS endpoints identical to the
Gazebo/MuJoCo fleets:
    /{ns}/joint_states   (sensor_msgs/JointState, 50 Hz)
    /{ns}/joint_commands (k1_interfaces/JointCommand) -> position targets

ROS stack: uses the isaacsim-bundled jazzy rclpy via env re-exec — this process
NEVER sources /opt/ros, so no RMW/env double-init (single DDS stack per process,
interop with system nodes happens over loopback DDS discovery).

Usage (venv-isaac python):
    python src/k1_sim_isaac/scripts/fleet_sim.py --n_robots 2 --headless
"""
import argparse
import os
import sys

CORE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "..",
)
# Resolve isaacsim ros2 core path robustly from site-packages
import site  # noqa: E402

def _ros_core_path():
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        cand = os.path.join(sp, "isaacsim", "exts", "isaacsim.ros2.core")
        if os.path.isdir(cand):
            return cand
    return None


def _ensure_bundled_ros_env():
    """Re-exec once with the bundled jazzy overlay prepended (before Kit starts)."""
    if os.environ.get("K1_BUNDLED_ROS_READY") == "1":
        return
    core = _ros_core_path()
    if core is None:
        print("[fleet] isaacsim.ros2.core not found — continuing without ROS")
        os.environ["K1_BUNDLED_ROS_READY"] = "0"
        return
    overlay = os.path.join(core, "jazzy", "rclpy")
    libs = os.pathsep.join([
        os.path.join(core, "jazzy", "lib"),
        os.path.join(core, "lib"),
    ])
    os.environ["PYTHONPATH"] = overlay + os.pathsep + os.environ.get("PYTHONPATH", "")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = libs + (os.pathsep + ld if ld else "")
    os.environ["AMENT_PREFIX_PATH"] = os.path.join(core, "jazzy")
    os.environ.setdefault("ROS_DOMAIN_ID", "77")
    os.environ["K1_BUNDLED_ROS_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_bundled_ros_env()

parser = argparse.ArgumentParser()
parser.add_argument("--n_robots", type=int, default=2)
parser.add_argument("--headless", action="store_true", default=False)
args, unknown = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

launcher_args = argparse.Namespace(**vars(args))
app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from k1_velocity.tasks.velocity.velocity_env_cfg import K1_ARTICULATION_CFG  # noqa: E402

try:
    from k1_interfaces.msg import JointCommand
except Exception:
    from sensor_msgs.msg import JointState as JointCommand  # degraded fallback

JOINT_ORDER = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


@configclass
class FleetSceneCfg(InteractiveSceneCfg):
    def __init_subclass__(cls):  # not used; kept simple below
        pass


def build_scene(n):
    cfg = FleetSceneCfg(num_envs=n, env_spacing=3.0)
    cfg.robot = {
        f"robot_{i}": K1_ARTICULATION_CFG.replace(
            prim_path=f"{{ENV_REGEX_NS}}/Robot_{i}",
            init_pos=(3.0 * i, 0.0, 0.57),
        )
        for i in range(n)
    }
    cfg.ground = sim_utils.GroundPlaneCfg()
    cfg.light = sim_utils.DomeLightCfg(intensity=600.0)
    return cfg


def main():
    rclpy.init()
    node = Node("k1_isaac_fleet")

    scene_cfg = build_scene(args.n_robots)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(0.005)

    robots = {f"k1_{i}": scene[f"robot_{i}"] for i in range(args.n_robots)}
    pubs, subs = {}, {}

    for ns, art in robots.items():
        idx = [art.find_joints(j)[0][0] for j in JOINT_ORDER]
        pubs[ns] = {
            "js": node.create_publisher(JointState, f"/{ns}/joint_states", 10),
            "idx": idx,
            "art": art,
        }

        def make_cb(namespace, articulation, joint_idx):
            def cb(msg):
                q_des = {n: p for n, p in zip(msg.joint_names, msg.positions)}
                if q_des:
                    ids, _ = articulation.find_joints(list(q_des.keys()))
                    vals = list(q_des.values())
                    articulation.set_joint_position_target(vals, ids)
            return cb

        subs[ns] = node.create_subscription(
            JointCommand, f"/{ns}/joint_commands", make_cb(ns, art, idx), 10
        )
        node.get_logger().info(f"[{ns}] isaac fleet endpoint ready")

    rate = node.create_rate(50)
    step = 0
    try:
        while simulation_app.is_running():
            sim.step(render=False)
            scene.update(0.005)
            step += 1
            if step % 4 == 0:  # 200Hz physx -> 50 Hz publish
                stamp = node.get_clock().now().to_msg()
                for ns, d in pubs.items():
                    art = d["art"]
                    m = JointState()
                    m.header.stamp = stamp
                    m.name = JOINT_ORDER
                    pos = art.data.joint_pos[0]
                    vel = art.data.joint_vel[0]
                    m.position = [float(pos[i]) for i in d["idx"]]
                    m.velocity = [float(vel[i]) for i in d["idx"]]
                    d["js"].publish(m)
            rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
