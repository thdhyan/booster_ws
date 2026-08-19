cmake_minimum_required(VERSION 3.8)
project(k1_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointCommand.msg"
  "msg/LocomotionCommand.msg"
  "msg/RobotStatus.msg"
  "msg/PolicyObs.msg"
  "srv/SetGaitMode.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_package()
