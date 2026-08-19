#!/usr/bin/env python3
"""
Isaac Sim 6.0.1 standalone script for K1 simulation.

Run this with the Isaac Sim Python (not system Python):
  cd ~/Projects/IsaacLab/isaac6
  ./python.sh ~/Projects/booster_ws/src/k1_sim_isaac/scripts/isaac_standalone.py \
      --robot_ns k1_0 --usd_path /path/to/k1.usd

This script:
1. Opens Isaac Sim
2. Loads K1 USD asset
3. Sets up ROS2 bridge (ActionGraph/OmniGraph) for joint states + commands
4. Runs simulation loop

IMPORTANT: Source Isaac Sim env BEFORE running. Never mix with system ROS2 shell.
"""
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_ns', default='k1_0')
    parser.add_argument('--usd_path', default='', help='Path to K1 USD file')
    parser.add_argument('--headless', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()

    # Isaac Sim launch
    try:
        from isaacsim import SimulationApp
    except ImportError:
        print("ERROR: Run this script with Isaac Sim Python, not system Python.")
        print("  cd ~/Projects/IsaacLab/isaac6 && ./python.sh <this_script>")
        sys.exit(1)

    sim_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

    # TODO: load K1 USD, set up OmniGraph ROS2 bridge
    # Stub: just run simulation loop
    from omni.isaac.core import World
    world = World(stage_units_in_meters=1.0)
    world.reset()

    print(f"Isaac Sim running [robot_ns={args.robot_ns}]")
    print("TODO: load K1 USD + ROS2 OmniGraph bridge")

    while sim_app.is_running():
        world.step(render=True)

    sim_app.close()

if __name__ == '__main__':
    main()
