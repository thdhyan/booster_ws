#!/usr/bin/env bash
# Record a K1 walk on the robot itself, so the bag survives without the laptop.
#
# WHY ON THE ROBOT: the K1 sits on a campus WiFi that blocks DDS multicast
# between hosts, so a laptop-side recorder discovers nothing. Recording locally
# also removes the network as a failure mode mid-walk.
#
# Verified on robot A2 (10.37.11.3) 2026-09-29: this records
#   /low_state /joint_states /odometer_state /tf /tf_static /fall_down   ~500 Hz
#   /booster_video_stream                                                 10 Hz JPEG
#   /remote_controller_state /joy                             controller input
#   .../camera_info x2                                        intrinsics (one-shot)
#
# Usage:  ./k1_walk_record.sh [minutes] [segment_seconds]
# Stop:   Ctrl-C  (finishes the current segment cleanly, then stops)
set -o pipefail

MINUTES="${1:-30}"
SEGMENT="${2:-300}"
OUTROOT="${K1_BAG_ROOT:-$HOME/k1_bags}"

# --- workspaces -------------------------------------------------------------
# booster_interface lives in BoosterRos2Interface, NOT BoosterRos2. Without
# this, rosbag2 silently drops every booster_interface topic ("unknown type")
# and you get a bag full of TF and nothing else. This is the #1 gotcha.
#
# `set -u` is deliberately NOT enabled: ROS setup.bash reads unset variables
# (AMENT_TRACE_SETUP_FILES) and aborts under it.
set +u
source /opt/ros/humble/setup.bash
source /opt/booster/BoosterRos2Interface/install/setup.bash 2>/dev/null
source /opt/booster/BoosterRos2/install/setup.bash 2>/dev/null
set -u

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUTROOT/k1_walk_$STAMP"
mkdir -p "$OUT"
LOG="$OUT/recorder.log"

# Topics that must actually contain data. If a bag has 0 for one of these the
# run is not good, and we say so loudly rather than letting it pass.
# camera_info is listed as critical on purpose: without calibration cuVSLAM will
# not initialise and nvblox silently warps, so a bag without it is not usable for
# SLAM even though the walk data in it is fine.
CRITICAL="/low_state /joint_states /odometer_state /tf /booster_video_stream /boostercamera/head/raw/rgb/camera_info"

TOPICS=(
  /low_state
  /joint_states
  /odometer_state
  /tf
  /tf_static
  /fall_down
  /remote_controller_state
  /joy
  /booster_video_stream
  /boostercamera/head/raw/rgb/camera_info
  /boostercamera/head/raw/right/rgb/camera_info
)

echo "=== K1 walk recorder ==="
echo "out      : $OUT"
echo "duration : ${MINUTES} min, in ${SEGMENT}s segments"
echo "disk free: $(df -h --output=avail / | tail -1 | tr -d ' ') on /"
echo

DEADLINE=$(( $(date +%s) + MINUTES * 60 ))
SEG=0
FAILED=()

summarise() {   # $1 = bag dir
  local bag="$1"
  local info; info="$(ros2 bag info "$bag" 2>/dev/null)"
  local missing=""
  for t in $CRITICAL; do
    # Count: N for this topic, 0 if the topic is absent from the listing.
    local c; c="$(echo "$info" | grep -F "Topic: $t |" | grep -oE 'Count: [0-9]+' | grep -oE '[0-9]+')"
    [ -z "$c" ] && c=0
    [ "$c" -eq 0 ] && missing="$missing $t"
  done
  if [ -n "$missing" ]; then
    echo "  [FAIL] no messages for:$missing"
    FAILED+=("seg$SEG:$missing")
  else
    echo "  [ok] all critical topics present"
  fi
  echo "  $(echo "$info" | grep -E 'Messages:|Duration:' | tr -s ' ' | tr '\n' ' ')"
  du -sh "$bag" 2>/dev/null | awk '{print "  size:",$1}'
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  SEG=$((SEG+1))
  BAG="$OUT/seg$(printf '%03d' $SEG)"
  echo "--- segment $SEG -> $BAG"
  ros2 bag record -o "$BAG" "${TOPICS[@]}" >>"$LOG" 2>&1 &
  BGPID=$!

  # Sleep in short slices so Ctrl-C is responsive and the deadline is honoured.
  LEFT=$SEGMENT
  while [ $LEFT -gt 0 ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 1 &
    wait $! 2>/dev/null
    LEFT=$((LEFT-1))
    kill -0 $BGPID 2>/dev/null || break
  done

  kill -INT $BGPID 2>/dev/null
  # Give rosbag2 time to flush its cache to sqlite before moving on.
  for _ in $(seq 1 15); do
    kill -0 $BGPID 2>/dev/null || break
    sleep 1
  done
  kill -9 $BGPID 2>/dev/null
  wait $BGPID 2>/dev/null

  summarise "$BAG"

  LEFTDISK=$(df --output=avail / | tail -1 | tr -d ' ')
  if [ "$LEFTDISK" -lt 20000000 ]; then
    echo "  [ABORT] less than 20 GB free -- stopping to protect the robot"
    break
  fi
done

echo
echo "=== summary ==="
echo "bags in $OUT:"
ls -1 "$OUT" 2>/dev/null | grep '^seg' | while read -r b; do
  printf '  %s  ' "$b"; du -sh "$OUT/$b" 2>/dev/null | awk '{print $1}'
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "PROBLEMS:"; printf '  %s\n' "${FAILED[@]}"
else
  echo "all segments carried every critical topic"
fi
echo
echo "Retrieve with:"
echo "  scp -r booster@10.37.11.3:$OUT ."
echo "and verify with:"
echo "  python3 real_robot/verify_bag.py <bag>"
