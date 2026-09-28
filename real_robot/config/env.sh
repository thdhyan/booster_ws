# Template environment for the K1 rig. Copy to env.sh and edit.
#   cp config/env.sh config/env.sh.local && $EDITOR config/env.sh.local
#   source config/env.sh.local
#
# env.sh.local is gitignored so credentials and per-site IPs stay local.

# --- robot network ---------------------------------------------------------
# Default is the robot's own hotspot subnet. Change per site.
ROBOT_HOST="${ROBOT_HOST:-192.168.1.100}"
ROBOT_USER="${ROBOT_USER:-robot}"
ROBOT_PORT="${ROBOT_PORT:-22}"
ROBOT_BAG_DIR="${ROBOT_BAG_DIR:-/home/robot/bags}"

# SSH key with access to the robot. Test before walking:
#   ssh -i "$SSH_KEY" "$ROBOT_USER@$ROBOT_HOST" 'echo ok'
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# --- local storage ---------------------------------------------------------
LOCAL_BAG_DIR="${LOCAL_BAG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/bags}"

# --- ROS 2 -----------------------------------------------------------------
# Uncomment one, matching the machine you are on.
# source /opt/ros/humble/setup.bash      # dl
# source /opt/ros/jazzy/setup.bash       # laptop
# source /home/thakk100/Projects/booster_ws/install/setup.bash   # colcon workspace
