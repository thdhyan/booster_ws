# Robot health and state summary
std_msgs/Header header
float32 battery_voltage
float32 battery_percent
bool[] foot_contact       # [left, right]
uint8 fault_code          # 0=ok
string fault_description
uint8 control_mode        # 0=idle 1=stand 2=locomotion 3=sdk
