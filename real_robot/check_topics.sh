#!/usr/bin/env bash
# Pre-flight: is the rig actually publishing what a map needs?
#
#   ./check_topics.sh
#
# Reports rates, the TF tree, and whether camera calibration is present.
# Calibration is checked explicitly because cuVSLAM rejects uncalibrated frames
# and NVBloX silently produces a warped reconstruction without it.
set -uo pipefail

if ! ros2 topic list >/dev/null 2>&1; then
  echo "[check] ros2 unreachable — source an environment first:"
  echo "         source /opt/ros/humble/setup.bash   (or your config/env.sh)"
  exit 3
fi

echo "=================== TOPICS ==================="
ros2 topic list | sed 's/^/  /'

fail=0
note() { echo "  $1"; }

echo
echo "=================== RATES ==================="
for t in /tf /imu/data /odom /joint_states; do
  if ros2 topic list 2>/dev/null | grep -qx "$t"; then
    r=$(timeout 6 ros2 topic hz "$t" 2>/dev/null | grep -oE 'average rate: [0-9.]+' | grep -oE '[0-9.]+')
    if [ -n "$r" ]; then note "$(printf '%-18s %8s Hz  %s' "$t" "$r" "$( [ "${r%.*}" -ge 1 ] 2>/dev/null && echo OK || echo 'SLOW/0')")"
    else note "$(printf '%-18s %8s' "$t" "no rate")  <-- stalled or no data"; fi
  else
    note "$(printf '%-18s %8s' "$t" "MISSING")"; fail=1
  fi
done

echo
echo "=================== CAMERA ==================="
cams=$(ros2 topic list 2>/dev/null | grep -E 'image_raw|camera_info' | cut -d/ -f1-2 | sort -u)
if [ -z "$cams" ]; then
  note "no camera topics"
  fail=1
else
  for c in $cams; do
    base="/$c"
    for kind in color depth; do
      t="${base}/${kind}/image_raw"
      ros2 topic list 2>/dev/null | grep -qx "$t" && \
        note "$(printf '%-42s %s' "$t" "$(timeout 5 ros2 topic hz "$t" 2>/dev/null | grep -oE 'average rate: [0-9.]+' | grep -oE '[0-9.]+' || echo 'no rate')")"
    done
    ci="${base}/color/camera_info"
    if ros2 topic list 2>/dev/null | grep -qx "$ci"; then
      note "camera_info present for $base"
    else
      note "NO camera_info for $base  <-- SLAM will reject this rig"
      fail=1
    fi
  done
fi

echo
echo "=================== TF TREE ==================="
if ros2 topic list 2>/dev/null | grep -qx /tf; then
  frames=$(timeout 6 ros2 topic echo /tf --once 2>/dev/null | grep -c "frame_id" || echo 0)
  note "tf frames in one message: $frames"
  [ "$frames" -lt 1 ] && { note "tf message empty -- static tree only?"; }
  if command -v timeout >/dev/null; then
    echo "  --- frames seen over 5s ---"
    timeout 5 ros2 topic echo /tf 2>/dev/null | grep -oE "child_frame_id: '[^']*'" \
      | sort -u | head -12 | sed 's/^/    /' || note "  (no frames observed — run a rosbag to inspect)"
  fi
else
  note "/tf MISSING — no map can be built without a TF tree"; fail=1
fi

echo
if [ $fail -eq 0 ]; then echo "[check] OK — ready to record"; else echo "[check] PROBLEMS FOUND (see above)"; fi
exit $fail
