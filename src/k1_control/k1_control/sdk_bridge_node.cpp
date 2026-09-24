// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024, Booster Robotics K1 ROS2 Workspace
//
// SDK Bridge Node — C++ implementation for real K1 robot control
// Bridges ROS2 JointCommand -> booster_robotics_sdk LowCmd (FastDDS)

#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <k1_interfaces/msg/joint_command.hpp>

#include <booster/idl/b1/LowCmd.h>
#include <booster/idl/b1/LowState.h>
#include <booster/idl/b1/MotorCmd.h>
#include <booster/idl/b1/MotorState.h>
#include <booster/robot/b1/b1_api_const.hpp>
#include <booster/robot/channel/channel_publisher.hpp>
#include <booster/robot/channel/channel_subscriber.hpp>
#include <booster/robot/channel/channel_factory.hpp>

using namespace std::chrono_literals;
using booster::robot::b1::JointIndexK1;
using booster::robot::b1::kJointCntK1;
using booster::robot::b1::kTopicJointCtrl;
using booster::robot::b1::kTopicLowState;

namespace k1_control
{

class SdkBridgeNode : public rclcpp::Node
{
public:
  SdkBridgeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("sdk_bridge_node", options)
  {
    // Parameters
    ns_ = this->declare_parameter<std::string>("robot_ns", "k1_0");
    sdk_ip_ = this->declare_parameter<std::string>("sdk_ip", "192.168.1.100");
    control_freq_ = this->declare_parameter<double>("control_freq", 50.0);
    network_interface_ = this->declare_parameter<std::string>("network_interface", "wlan0");

    // ROS2 interfaces
    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      "/" + ns_ + "/joint_states", 10);
    joint_cmd_sub_ = this->create_subscription<k1_interfaces::msg::JointCommand>(
      "/" + ns_ + "/joint_commands", 10,
      std::bind(&SdkBridgeNode::jointCmdCallback, this, std::placeholders::_1));

    // Initialize booster SDK channel factory (FastDDS)
    // Domain ID 0, network interface from parameter
    booster::robot::ChannelFactory::Instance()->Init(0, network_interface_);

    // LowCmd publisher (to robot)
    low_cmd_pub_ = std::make_shared<booster::robot::ChannelPublisher<booster_interface::msg::LowCmd>>(
      kTopicJointCtrl);
    low_cmd_pub_->InitChannel();

    // LowState subscriber (from robot)
    low_state_sub_ = std::make_shared<booster::robot::ChannelSubscriber<booster_interface::msg::LowState>>(
      kTopicLowState);
    low_state_sub_->InitChannel(
      [this](const void* msg) { this->lowStateCallback(static_cast<const booster_interface::msg::LowState*>(msg)); });

    // Initialize LowCmd message with 22 motors (K1)
    low_cmd_msg_.cmd_type(booster_interface::msg::CmdType::PARALLEL);
    for (size_t i = 0; i < kJointCntK1; ++i) {
      booster_interface::msg::MotorCmd motor_cmd;
      low_cmd_msg_.motor_cmd().push_back(motor_cmd);
    }

    // Joint name mapping: ROS (URDF) -> SDK (JointIndexK1)
    // K1 has 22 DoF: 2 head + 10 arm + 10 leg (no waist on K1)
    initJointMapping();

    // Control timer
    control_dt_ = 1.0 / control_freq_;
    control_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(control_dt_),
      std::bind(&SdkBridgeNode::controlLoop, this));

    RCLCPP_INFO(this->get_logger(),
      "SdkBridgeNode ready [ns=%s, sdk_ip=%s, freq=%.1f Hz, iface=%s]",
      ns_.c_str(), sdk_ip_.c_str(), control_freq_, network_interface_.c_str());
  }

private:
  // Joint mapping: URDF names (policy order) -> SDK JointIndexK1
  // Policy controls 12 leg joints in this exact order:
  // Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw,
  // Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll,
  // Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw,
  // Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
  void initJointMapping()
  {
    // K1 leg joints (12) in policy order -> SDK indices
    // Note: K1 has no waist, and ankle uses CrankUp/CrankDown (4-bar linkage)
    // Mapping based on b1_api_const.hpp JointIndexK1 enum
    joint_map_ = {
      {"Left_Hip_Pitch",     static_cast<uint8_t>(JointIndexK1::kLeftHipPitch)},     // 10
      {"Left_Hip_Roll",      static_cast<uint8_t>(JointIndexK1::kLeftHipRoll)},      // 11
      {"Left_Hip_Yaw",       static_cast<uint8_t>(JointIndexK1::kLeftHipYaw)},       // 12
      {"Left_Knee_Pitch",    static_cast<uint8_t>(JointIndexK1::kLeftKneePitch)},    // 13
      {"Left_Ankle_Pitch",   static_cast<uint8_t>(JointIndexK1::kCrankUpLeft)},      // 14 (maps to crank up)
      {"Left_Ankle_Roll",    static_cast<uint8_t>(JointIndexK1::kCrankDownLeft)},    // 15 (maps to crank down)
      {"Right_Hip_Pitch",    static_cast<uint8_t>(JointIndexK1::kRightHipPitch)},    // 16
      {"Right_Hip_Roll",     static_cast<uint8_t>(JointIndexK1::kRightHipRoll)},     // 17
      {"Right_Hip_Yaw",      static_cast<uint8_t>(JointIndexK1::kRightHipYaw)},      // 18
      {"Right_Knee_Pitch",   static_cast<uint8_t>(JointIndexK1::kRightKneePitch)},   // 19
      {"Right_Ankle_Pitch",  static_cast<uint8_t>(JointIndexK1::kCrankUpRight)},     // 20
      {"Right_Ankle_Roll",   static_cast<uint8_t>(JointIndexK1::kCrankDownRight)},   // 21
    };

    // PD gains per joint (from booster actuator specs, Nm/rad & Nms/rad)
    // These should match the sim_bridge and training
    kp_map_ = {
      {static_cast<uint8_t>(JointIndexK1::kLeftHipPitch),   30.2},
      {static_cast<uint8_t>(JointIndexK1::kLeftHipRoll),    21.4},
      {static_cast<uint8_t>(JointIndexK1::kLeftHipYaw),     17.8},
      {static_cast<uint8_t>(JointIndexK1::kLeftKneePitch),  60.4},
      {static_cast<uint8_t>(JointIndexK1::kCrankUpLeft),    35.7},
      {static_cast<uint8_t>(JointIndexK1::kCrankDownLeft),  35.7},
      {static_cast<uint8_t>(JointIndexK1::kRightHipPitch),  30.2},
      {static_cast<uint8_t>(JointIndexK1::kRightHipRoll),   21.4},
      {static_cast<uint8_t>(JointIndexK1::kRightHipYaw),    17.8},
      {static_cast<uint8_t>(JointIndexK1::kRightKneePitch), 60.4},
      {static_cast<uint8_t>(JointIndexK1::kCrankUpRight),   35.7},
      {static_cast<uint8_t>(JointIndexK1::kCrankDownRight), 35.7},
    };

    kd_map_ = {
      {static_cast<uint8_t>(JointIndexK1::kLeftHipPitch),   30.2 * 0.12},
      {static_cast<uint8_t>(JointIndexK1::kLeftHipRoll),    21.4 * 0.12},
      {static_cast<uint8_t>(JointIndexK1::kLeftHipYaw),     17.8 * 0.12},
      {static_cast<uint8_t>(JointIndexK1::kLeftKneePitch),  4.8},
      {static_cast<uint8_t>(JointIndexK1::kCrankUpLeft),    4.3},
      {static_cast<uint8_t>(JointIndexK1::kCrankDownLeft),  4.3},
      {static_cast<uint8_t>(JointIndexK1::kRightHipPitch),  30.2 * 0.12},
      {static_cast<uint8_t>(JointIndexK1::kRightHipRoll),   21.4 * 0.12},
      {static_cast<uint8_t>(JointIndexK1::kRightHipYaw),    17.8 * 0.12},
      {static_cast<uint8_t>(JointIndexK1::kRightKneePitch), 4.8},
      {static_cast<uint8_t>(JointIndexK1::kCrankUpRight),   4.3},
      {static_cast<uint8_t>(JointIndexK1::kCrankDownRight), 4.3},
    };

    // Default joint positions (standing pose, all zeros for legs)
    default_pos_.fill(0.0f);

    // Last commanded positions (for safety interpolation)
    last_pos_.fill(0.0f);
    first_cmd_received_ = false;
  }

  void jointCmdCallback(const k1_interfaces::msg::JointCommand::SharedPtr msg)
  {
    if (msg->joint_names.size() != msg->positions.size()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "joint_names/positions size mismatch (%zu vs %zu)", msg->joint_names.size(), msg->positions.size());
      return;
    }

    // Map named joints to SDK indices
    for (size_t i = 0; i < msg->joint_names.size(); ++i) {
      const auto & name = msg->joint_names[i];
      auto it = joint_map_.find(name);
      if (it != joint_map_.end()) {
        uint8_t sdk_idx = it->second;
        target_pos_[sdk_idx] = static_cast<float>(msg->positions[i]);
        has_target_[sdk_idx] = true;
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "Unknown joint '%s' in command", name.c_str());
      }
    }
    first_cmd_received_ = true;
  }

  void lowStateCallback(const booster_interface::msg::LowState * msg)
  {
    if (!msg) return;

    // Publish joint states to ROS2
    sensor_msgs::msg::JointState js_msg;
    js_msg.header.stamp = this->now();
    js_msg.header.frame_id = "base_link";

    // K1 joint names in URDF order (22 DoF)
    static const std::array<std::string, 22> joint_names = {
      "AAHead_yaw", "Head_pitch",
      "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
      "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
      "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
      "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
      "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
      "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"
    };

    js_msg.name = std::vector<std::string>(joint_names.begin(), joint_names.end());
    js_msg.position.resize(22);
    js_msg.velocity.resize(22);
    js_msg.effort.resize(22);

    // Extract motor states (22 motors for K1)
    const auto & motor_states = msg->motor_state_parallel();
    for (size_t i = 0; i < std::min<size_t>(kJointCntK1, motor_states.size()); ++i) {
      js_msg.position[i] = motor_states[i].q();
      js_msg.velocity[i] = motor_states[i].dq();
      js_msg.effort[i] = motor_states[i].tau_est();
    }

    joint_state_pub_->publish(js_msg);
  }

  void controlLoop()
  {
    if (!first_cmd_received_) {
      // No command yet - hold default position with zero torque
      for (size_t i = 0; i < kJointCntK1; ++i) {
        low_cmd_msg_.motor_cmd()[i].q(default_pos_[i]);
        low_cmd_msg_.motor_cmd()[i].dq(0.0f);
        low_cmd_msg_.motor_cmd()[i].kp(0.0f);
        low_cmd_msg_.motor_cmd()[i].kd(0.0f);
        low_cmd_msg_.motor_cmd()[i].tau(0.0f);
      }
    } else {
      // Safety: rate-limit position changes
      const float max_joint_velocity = 2.0f;  // rad/s
      const float max_joint_delta = max_joint_velocity * control_dt_;

      for (size_t i = 0; i < kJointCntK1; ++i) {
        float target = has_target_[i] ? target_pos_[i] : default_pos_[i];
        
        // Clamp position change rate
        float delta = target - last_pos_[i];
        delta = std::clamp(delta, -max_joint_delta, max_joint_delta);
        float cmd_pos = last_pos_[i] + delta;
        last_pos_[i] = cmd_pos;

        // Fill LowCmd
        low_cmd_msg_.motor_cmd()[i].q(cmd_pos);
        low_cmd_msg_.motor_cmd()[i].dq(0.0f);  // velocity feedforward
        low_cmd_msg_.motor_cmd()[i].kp(kp_map_[i]);
        low_cmd_msg_.motor_cmd()[i].kd(kd_map_[i]);
        low_cmd_msg_.motor_cmd()[i].tau(0.0f);  // torque feedforward
        low_cmd_msg_.motor_cmd()[i].weight(1.0f);  // full weight
        low_cmd_msg_.motor_cmd()[i].mode(0x01);    // position control mode
      }
    }

    // Send to robot via FastDDS
    low_cmd_pub_->Write(&low_cmd_msg_);
  }

  // Parameters
  std::string ns_;
  std::string sdk_ip_;
  double control_freq_;
  std::string network_interface_;
  double control_dt_;

  // ROS2 interfaces
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Subscription<k1_interfaces::msg::JointCommand>::SharedPtr joint_cmd_sub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  // Booster SDK (FastDDS)
  std::shared_ptr<booster::robot::ChannelPublisher<booster_interface::msg::LowCmd>> low_cmd_pub_;
  std::shared_ptr<booster::robot::ChannelSubscriber<booster_interface::msg::LowState>> low_state_sub_;
  booster_interface::msg::LowCmd low_cmd_msg_;

  // Joint mapping
  std::unordered_map<std::string, uint8_t> joint_map_;
  std::array<float, kJointCntK1> target_pos_{};
  std::array<bool, kJointCntK1> has_target_{};
  std::array<float, kJointCntK1> default_pos_;
  std::array<float, kJointCntK1> last_pos_;
  std::unordered_map<uint8_t, float> kp_map_;
  std::unordered_map<uint8_t, float> kd_map_;
  bool first_cmd_received_;
};

}  // namespace k1_control

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(k1_control::SdkBridgeNode)