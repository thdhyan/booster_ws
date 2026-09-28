#!/usr/bin/env bash
# One-shot session: pre-flight -> record -> verify -> pull from robot -> report.
#
#   ./record_session.sh walk_fast --duration 120
#
# The pre-flight deliberately refuses to record without TF and IMU. A bag that
# lacks them cannot be used to build a map, and on a real robot the disk cost of
# re-recording is much higher than the cost of waiting for a topic to appear.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
case "${1:-}" in -h|--help) sed -n '2,8p' "$0"; exit 0 ;; esac
NAME="${1:-session}"; shift || true
DURATION=120
TOPICS_FILE="$HERE/config/topics_walk.yaml"
LOCAL_BAG_DIR="${LOCAL_BAG_DIR:-$HERE/bags}"
NO_PULL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --topics-file) TOPICS_FILE="$2"; shift 2 ;;
    --no-pull) NO_PULL=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BAG="$LOCAL_BAG_DIR/${NAME}_${STAMP}"

echo "=============================================================="
echo " K1 session: $NAME"
echo " bag: $BAG"
echo "=============================================================="

# ---- pre-flight -----------------------------------------------------------
echo "[session] pre-flight..."
TF_OK=0; IMU_OK=0; VIS_OK=0
if ros2 topic list >/dev/null 2>&1; then
  LIST=$(ros2 topic list)
  echo "$LIST" | grep -qE '^/tf$' && TF_OK=1
  echo "$LIST" | grep -qE '^/imu' && IMU_OK=1
  echo "$LIST" | grep -qE 'image_raw' && VIS_OK=1
  echo "[session] tf=$TF_OK imu=$IMU_OK visual=$VIS_OK"
  for t in /imu/data /odom; do
    r=$(ros2 topic hz "$t" 2>/dev/null | head -2 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    [ -n "$r" ] && echo "[session]   $t ~$r Hz"
  done
else
  echo "[session] WARNING: ros2 not reachable (no daemon / not sourced)"
fi

if [ "$TF_OK" != "1" ] || [ "$IMU_OK" != "1" ]; then
  echo "[session] ABORT: TF and IMU are both required for a usable bag."
  echo "[session]   tf=$TF_OK imu=$IMU_OK — fix the bridge, or pass a topics file"
  echo "[session]   that excludes them deliberately."
  exit 3
fi
[ "$VIS_OK" == "1" ] || echo "[session] WARNING: no visual topic found; recording anyway"

# ---- record ---------------------------------------------------------------
echo "[session] recording ${DURATION}s ..."
"$HERE/rosbag_record.sh" --out "$BAG" --duration "$DURATION" --topics-file "$TOPICS_FILE"
RC=$?
[ $RC -ne 0 ] && { echo "[session] recording failed rc=$RC"; exit $RC; }

# ---- report ---------------------------------------------------------------
SIZE=$(du -sh "$BAG" 2>/dev/null | cut -f1)
echo "[session] recorded $SIZE -> $BAG"
python3 "$HERE/verify_bag.py" "$BAG"

if [ "$NO_PULL" = "0" ] && [ -n "${ROBOT_HOST:-}" ]; then
  echo "[session] pulling from robot $ROBOT_HOST ..."
  LOCAL_BAG_DIR="$LOCAL_BAG_DIR" "$HERE/sftp_sync.sh" pull --filter "$(basename "$BAG")" \
    && echo "[session] sync OK" || echo "[session] sync FAILED (bag is still local)"
fi

echo "=============================================================="
echo " done: $NAME  ($SIZE)"
echo "=============================================================="
