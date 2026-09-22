#!/usr/bin/env python3
"""
Launch file for visualizing the K1 robot in RViz2.
- Runs robot_state_publisher to publish TF from URDF
- Runs joint_state_publisher_gui to manually set joint angles
- Launches RViz2 for visualization
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from pathlib import Path


def generate_launch_description():
    # Declare launch argument for robot namespace
    robot_ns = DeclareLaunchArgument(
        'robot_ns',
        default_value='k1_0',
        description='Namespace prefix for robot links and joints'
    )

    # Get package share directory
    pkg_share = FindPackageShare('k1_description')

    # Path to URDF xacro file
    urdf_file = PathJoinSubstitution([pkg_share, 'urdf', 'k1.urdf.xacro'])

    # Path to RViz config (optional, will create default if not found)
    rviz_config = PathJoinSubstitution([pkg_share, 'config', 'view_robot.rviz'])

    # robot_state_publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': open(Path(__file__).parent.parent / 'urdf' / 'k1.urdf.xacro').read(),
            'frame_prefix': LaunchConfiguration('robot_ns') + '/'
        }],
        remappings=[
            ('robot_description', 'robot_description'),
        ]
    )

    # joint_state_publisher_gui node
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        parameters=[],
        remappings=[
            ('robot_description', 'robot_description'),
        ]
    )

    # RViz2 node
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        robot_ns,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2,
    ])
