#!/usr/bin/env python3
"""
Sim Bridge Node — passthrough JointCommand -> ros2_control (Gazebo/Isaac).

In sim mode, JointCommand msgs are forwarded directly to the
gz_ros2_control / Isaac OmniGraph joint command topics.

Topics:
  Sub: /{robot_ns}/joint_commands  (k1_interfaces/JointCommand)
  Pub: /{robot_ns}/k1_joint_controller/commands  (std_msgs/Float64MultiArray)

Params:
  robot_ns: str = 'k1_0'
  sim_backend: str = 'gazebo'  # 'gazebo' or 'isaac'
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# K1 leg joint order (must match ros2_control.yaml)
LEG_JOINTS = [
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw',
    'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw',
    'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll',
]


class SimBridgeNode(Node):
    def __init__(self):
        super().__init__('sim_bridge_node')
        ns = self.declare_parameter('robot_ns', 'k1_0').value
        backend = self.declare_parameter('sim_backend', 'gazebo').value
        self._ns = ns
        self._cmd_pub = self.create_publisher(
            Float64MultiArray,
            f'/{ns}/k1_joint_controller/commands', 10)
        self.get_logger().info(f'SimBridgeNode ready [ns={ns}, backend={backend}]')


def main(args=None):
    rclpy.init(args=args)
    node = SimBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
