from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_ns_arg = DeclareLaunchArgument(
        'robot_ns',
        default_value='k1_0',
        description='Robot namespace'
    )
    sim_mode_arg = DeclareLaunchArgument(
        'sim_mode',
        default_value='true',
        description='Run in simulation mode (true) or real robot mode (false)'
    )

    robot_ns = LaunchConfiguration('robot_ns')
    sim_mode = LaunchConfiguration('sim_mode')

    # Choose bridge based on sim_mode
    bridge_node = Node(
        package='k1_control',
        executable='sim_bridge' if sim_mode.perform(None) == 'true' else 'sdk_bridge',
        name='control_bridge',
        namespace=robot_ns,
        output='screen',
        parameters=[
            {'robot_ns': robot_ns},
        ]
    )

    return LaunchDescription([
        robot_ns_arg,
        sim_mode_arg,
        bridge_node,
    ])
