#!/usr/bin/env bash
# Record the K1 head stereo pair + proprioception ON THE ROBOT, detached, so the
# laptop can be unplugged for the walk.
#
# Usage (on the robot):
#   setsid nohup bash k1_stereo_record.sh [minutes] [mode] > ~/k1_bags/last_stereo.log 2>&1 &
#   mode = rect  (default) rectified pair ~8 Hz + raw left 30 Hz
#   mode = raw   additionally raw right at 30 Hz (suspected to stall the X5 -- see below)
#
# QoS: the camera publishers are RELIABLE + VOLATILE; default rosbag2 subscribers
# are silently rejected, hence the explicit overrides file.
#
# Suspected stall trigger (2026-09-28, A2): the pair survived >3 min with the
# rectified pair + raw/rgb subscribed, then died seconds after raw/right/rgb and
# raw/combine/rgb were added. raw/combine is never recorded here.
set -o pipefail

MINUTES="${1:-15}"
MODE="${2:-rect}"
OUTROOT="${K1_BAG_ROOT:-$HOME/k1_bags}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUTROOT/k1_stereo_$STAMP"
mkdir -p "$OUTROOT"

set +u
source /opt/ros/humble/setup.bash
source /opt/booster/BoosterRos2Interface/install/setup.bash 2>/dev/null
source /opt/booster/BoosterRos2/install/setup.bash 2>/dev/null

CAM=(
  /boostercamera/head/rgb
  /boostercamera/head/right/rgb
  /boostercamera/head/rgb/camera_info
  /boostercamera/head/right/rgb/camera_info
  /boostercamera/head/raw/rgb
  /boostercamera/head/raw/rgb/camera_info
  /boostercamera/head/raw/right/rgb/camera_info
)
[ "$MODE" = "raw" ] && CAM+=(/boostercamera/head/raw/right/rgb)

STATE=(/low_state /joint_states /odometer_state /tf /tf_static /fall_down /remote_controller_state /joy)

QOS="$OUTROOT/qos_camera_$STAMP.yaml"
: > "$QOS"
for t in "${CAM[@]}"; do
  printf '%s:\n  reliability: reliable\n  durability: volatile\n  history: keep_last\n  depth: 10\n' "$t" >> "$QOS"
done

echo "out=$OUT minutes=$MINUTES mode=$MODE uptime=$(cut -d' ' -f1 /proc/uptime)s"
echo "topics: ${CAM[*]} ${STATE[*]}"

# 60 s splits: a stall or crash loses at most one minute.
ros2 bag record -o "$OUT" --max-bag-duration 60 \
  --qos-profile-overrides-path "$QOS" "${CAM[@]}" "${STATE[@]}" &
PID=$!
echo "rosbag pid=$PID"

sleep $((MINUTES * 60))
kill -INT "$PID" 2>/dev/null
for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

ros2 bag info "$OUT" 2>/dev/null | grep -E "Duration|Messages|Topic:"
du -sh "$OUT"
echo "DONE $OUT"
