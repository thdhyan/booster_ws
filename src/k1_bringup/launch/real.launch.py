#!/usr/bin/env python3
"""
Real robot bringup — PLACEHOLDER.
Usage: ros2 launch k1_bringup real.launch.py robot_ns:=k1_0 sdk_ip:=192.168.1.100

WARNING: Ensure booster_robotics_sdk is built and robot is powered on.
         Run in a safe, open area. Verify joint limits before enabling policy.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_ns', default_value='k1_0'),
        DeclareLaunchArgument('sdk_ip', default_value='192.168.1.100'),
        # TODO Phase 4: add k1_control sdk_bridge_node, k1_locomotion, k1_wbc
    ])
