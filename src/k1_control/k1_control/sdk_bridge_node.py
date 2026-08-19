#!/usr/bin/env python3
"""
SDK Bridge Node — bridges ROS2 JointCommand -> booster_robotics_sdk -> real K1 robot.

Topics:
  Sub: /{robot_ns}/joint_commands  (k1_interfaces/JointCommand)
  Pub: /{robot_ns}/joint_states    (sensor_msgs/JointState)

Params:
  robot_ns: str = 'k1_0'
  sdk_ip: str = '192.168.1.100'   # robot IP for SDK
  control_freq: float = 50.0      # Hz

NOTE: booster_robotics_sdk uses fastDDS internally (not ROS2 DDS).
      This node wraps the SDK Python/C++ API.
      SDK not yet integrated — stub raises NotImplementedError.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
# from k1_interfaces.msg import JointCommand  # uncomment after k1_interfaces built


class SdkBridgeNode(Node):
    def __init__(self):
        super().__init__('sdk_bridge_node')
        ns = self.declare_parameter('robot_ns', 'k1_0').value
        self.declare_parameter('sdk_ip', '192.168.1.100')
        self.declare_parameter('control_freq', 50.0)

        self._ns = ns
        self._joint_state_pub = self.create_publisher(
            JointState, f'/{ns}/joint_states', 10)
        # self._cmd_sub = self.create_subscription(
        #     JointCommand, f'/{ns}/joint_commands', self._cmd_cb, 10)
        self.get_logger().info(f'SdkBridgeNode ready [ns={ns}] — SDK stub, not connected')

    def _cmd_cb(self, msg):
        raise NotImplementedError('SDK integration pending — see booster_robotics_sdk')


def main(args=None):
    rclpy.init(args=args)
    node = SdkBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
