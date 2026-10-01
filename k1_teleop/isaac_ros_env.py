"""Use Isaac Sim's bundled Jazzy rclpy from an Isaac Lab venv process (same
re-exec as ``src/k1_sim_isaac/scripts/fleet_sim.py``): the process never sources
/opt/ros, so there is one DDS stack per process and it talks to system nodes
over normal DDS discovery. Call ``ensure()`` before anything imports rclpy.
"""
import os
import site
import sys


def _ros_core_path():
    roots = [os.path.join(sp, "isaacsim") for sp in site.getsitepackages() + [site.getusersitepackages()]]
    roots.append(os.environ.get("ISAAC_PATH", "/isaac-sim"))          # isaac-lab container layout
    for root in roots:
        cand = os.path.join(root, "exts", "isaacsim.ros2.core")
        if os.path.isdir(cand):
            return cand
    return None


def ensure() -> bool:
    """Re-exec once with the bundled overlay prepended; returns False if Isaac Sim's
    ROS 2 extension is missing (no ROS then)."""
    if os.environ.get("K1_BUNDLED_ROS_READY") == "1":
        return True
    if os.environ.get("K1_BUNDLED_ROS_READY") == "0":
        return False
    core = _ros_core_path()
    if core is None:
        print("[ros-env] isaacsim.ros2.core not found - continuing without ROS")
        os.environ["K1_BUNDLED_ROS_READY"] = "0"
        return False
    os.environ["PYTHONPATH"] = os.path.join(core, "jazzy", "rclpy") + os.pathsep + os.environ.get("PYTHONPATH", "")
    libs = os.pathsep.join([os.path.join(core, "jazzy", "lib"), os.path.join(core, "lib")])
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = libs + (os.pathsep + ld if ld else "")
    os.environ["AMENT_PREFIX_PATH"] = os.path.join(core, "jazzy")
    os.environ["K1_BUNDLED_ROS_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
