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
    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        default_value='models/k1_velocity_policy.pt',
        description='Path to TorchScript policy file'
    )

    robot_ns = LaunchConfiguration('robot_ns')
    policy_path = LaunchConfiguration('policy_path')

    locomotion_node = Node(
        package='k1_locomotion',
        executable='locomotion_node',
        name='locomotion',
        namespace=robot_ns,
        output='screen',
        parameters=[
            {'robot_ns': robot_ns},
            {'policy_path': policy_path},
        ]
    )

    return LaunchDescription([
        robot_ns_arg,
        policy_path_arg,
        locomotion_node,
    ])
