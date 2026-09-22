#!/usr/bin/env python3
"""
Isaac Sim bringup for K1.
Launches the Isaac Sim standalone script via the isaac6 Python.

IMPORTANT: Isaac Sim ROS2 and system ROS2 must NOT be sourced in same shell.
This launch file runs the Isaac script in a subprocess that sources isaac6 env.

Usage: ros2 launch k1_sim_isaac sim_isaac.launch.py robot_ns:=k1_0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

ISAAC6_ROOT = os.path.expanduser('~/Projects/IsaacLab/isaac6')
ISAAC_PYTHON = os.path.join(ISAAC6_ROOT, 'python.sh')

def generate_launch_description():
    robot_ns = LaunchConfiguration('robot_ns')
    headless = LaunchConfiguration('headless')
    usd_path = LaunchConfiguration('usd_path')

    return LaunchDescription([
        DeclareLaunchArgument('robot_ns', default_value='k1_0'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('usd_path', default_value='',
            description='Path to K1 USD file (from booster_assets)'),

        ExecuteProcess(
            cmd=[
                ISAAC_PYTHON,
                PathJoinSubstitution([FindPackageShare('k1_sim_isaac'), 'scripts', 'isaac_standalone.py']),
                '--robot_ns', robot_ns,
                '--usd_path', usd_path,
            ],
            output='screen',
            additional_env={'ROS_DOMAIN_ID': '0'},
        ),
    ])
