# High-level velocity command + gait override
std_msgs/Header header
geometry_msgs/Twist twist  # cmd_vel
uint8 gait_mode    # 0=walk 1=stand 2=recovery
bool policy_active # true = policy running, false = hold position
