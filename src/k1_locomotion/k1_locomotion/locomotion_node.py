#!/usr/bin/env python3
"""
Locomotion Node — runs trained RSL-RL velocity policy for K1.

Loads a TorchScript policy (models/k1_velocity_policy.pt) and runs
inference at 50 Hz, building observations from joint_states + cmd_vel.

Observation vector (72-dim, 10-step history = 720-dim input):
  [0:3]   base_lin_vel (estimated or from sim)
  [3:6]   base_ang_vel
  [6:9]   projected_gravity
  [9:21]  leg joint positions (12) — default offset removed
  [21:33] leg joint velocities (12)
  [33:36] velocity commands (vx, vy, yaw_rate)
  [36:48] last action (12)
  [48:72] (reserved / height scan placeholder)

Action: 12 leg joint position offsets from default pose.

Topics:
  Sub: /{robot_ns}/joint_states      (sensor_msgs/JointState)
  Sub: /{robot_ns}/cmd_vel           (geometry_msgs/Twist)
  Pub: /{robot_ns}/joint_commands    (k1_interfaces/JointCommand)
  Pub: /{robot_ns}/policy_obs        (k1_interfaces/PolicyObs) [debug]

Params:
  robot_ns: str = 'k1_0'
  policy_path: str = 'models/k1_velocity_policy.pt'
  control_freq: float = 50.0
  obs_history_len: int = 10
  action_scale: float = 0.25
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np
import threading

# K1 leg joints in policy order
LEG_JOINTS = [
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw',
    'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw',
    'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll',
]

# Default standing pose (radians) — tune after booster_assets URDF review
K1_DEFAULT_LEG_POS = np.zeros(12, dtype=np.float32)

OBS_DIM = 72
HISTORY_LEN = 10


class LocomotionNode(Node):
    def __init__(self):
        super().__init__('locomotion_node')
        ns = self.declare_parameter('robot_ns', 'k1_0').value
        policy_path = self.declare_parameter('policy_path', 'models/k1_velocity_policy.pt').value
        freq = self.declare_parameter('control_freq', 50.0).value
        self._action_scale = self.declare_parameter('action_scale', 0.25).value
        self._ns = ns
        self._lock = threading.Lock()

        # State
        self._cmd_vel = np.zeros(3, dtype=np.float32)  # vx, vy, yaw
        self._joint_pos = np.zeros(12, dtype=np.float32)
        self._joint_vel = np.zeros(12, dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._obs_history = np.zeros((HISTORY_LEN, OBS_DIM), dtype=np.float32)
        self._policy = None

        # Load policy
        try:
            import torch
            self._policy = torch.jit.load(policy_path)
            self._policy.eval()
            self.get_logger().info(f'Policy loaded: {policy_path}')
        except Exception as e:
            self.get_logger().warn(f'Policy not loaded ({e}) — running in passthrough mode')

        # Subs/pubs
        self.create_subscription(JointState, f'/{ns}/joint_states', self._js_cb, 10)
        self.create_subscription(Twist, f'/{ns}/cmd_vel', self._cmd_cb, 10)
        # self._cmd_pub = self.create_publisher(JointCommand, f'/{ns}/joint_commands', 10)

        self.create_timer(1.0 / freq, self._control_loop)
        self.get_logger().info(f'LocomotionNode ready [ns={ns}, freq={freq}Hz]')

    def _cmd_cb(self, msg: Twist):
        with self._lock:
            self._cmd_vel[:] = [msg.linear.x, msg.linear.y, msg.angular.z]

    def _js_cb(self, msg: JointState):
        with self._lock:
            for i, jname in enumerate(LEG_JOINTS):
                if jname in msg.name:
                    idx = msg.name.index(jname)
                    self._joint_pos[i] = msg.position[idx]
                    self._joint_vel[i] = msg.velocity[idx] if msg.velocity else 0.0

    def _build_obs(self) -> np.ndarray:
        """Build 72-dim observation vector (placeholders for base vel/gravity)."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        # [0:3] base_lin_vel — placeholder (use IMU/estimator when available)
        # [3:6] base_ang_vel — placeholder
        # [6:9] projected_gravity — placeholder [0, 0, -1] nominal
        obs[6:9] = [0.0, 0.0, -1.0]
        obs[9:21] = self._joint_pos - K1_DEFAULT_LEG_POS
        obs[21:33] = self._joint_vel
        obs[33:36] = self._cmd_vel
        obs[36:48] = self._last_action
        return obs

    def _control_loop(self):
        with self._lock:
            obs = self._build_obs()
        # Shift history and append
        self._obs_history[:-1] = self._obs_history[1:]
        self._obs_history[-1] = obs

        if self._policy is None:
            return  # passthrough / no policy loaded

        try:
            import torch
            obs_tensor = torch.from_numpy(self._obs_history.flatten()).unsqueeze(0)
            with torch.no_grad():
                action = self._policy(obs_tensor).squeeze(0).numpy()
            self._last_action[:] = action
            # TODO: publish JointCommand with positions = default + action_scale * action
        except Exception as e:
            self.get_logger().error(f'Policy inference error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LocomotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
