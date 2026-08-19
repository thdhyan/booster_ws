# Commanded joint targets for K1 (subset or all 22 joints)
# joint_names must match booster_assets exactly
std_msgs/Header header
string[] joint_names
float64[] positions   # rad
float64[] velocities  # rad/s (optional, zeros ok)
float64[] efforts     # Nm (optional, zeros ok)
uint8 control_mode    # 0=position 1=velocity 2=effort
