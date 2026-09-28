#!/usr/bin/env bash
# Record a rosbag2 from the K1 rig: TF, IMU, visual, odometry, joints.
#
#   ./rosbag_record.sh --out bags/walk_fast --duration 120
#   ./rosbag_record.sh --topics-file config/topics_walk.yaml --duration 300
#
# Works on both ROS 2 Humble (dl) and Jazzy (laptop). The two distros differ in
# the `ros2 bag record` CLI -- Jazzy wants -o/--output and defaults to mcap,
# Humble takes a positional path and defaults to sqlite3, and --max-chunk-size
# only exists on one of them -- so capabilities are probed rather than assumed.
#
# Clean shutdown on Ctrl-C so the bag metadata is finalized; a bag that was
# killed rather than closed will not replay.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=""
DURATION=0                 # 0 = until Ctrl-C
TOPICS_FILE=""
COMPRESSION=auto           # auto | none | message
MAX_CACHE=536870912         # 512 MB writer cache
MAX_BAG_SIZE=0              # 0 = no split (Jazzy -b; Humble has no equivalent)
STORAGE=""
PRESET="fastwrite"          # Jazzy storage preset
EXTRA=()

usage() { sed -n '2,15p' "$0"; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --out)          OUT="$2"; shift 2 ;;
    --duration)     DURATION="$2"; shift 2 ;;
    --topics-file)  TOPICS_FILE="$2"; shift 2 ;;
    --max-cache-size) MAX_CACHE="$2"; shift 2 ;;
    --max-bag-size) MAX_BAG_SIZE="$2"; shift 2 ;;
    --storage)      STORAGE="$2"; shift 2 ;;
    --preset)       PRESET="$2"; shift 2 ;;
    --compression)  COMPRESSION="$2"; shift 2 ;;
    --no-compression) COMPRESSION="none"; shift ;;
    -h|--help)      usage ;;
    *)              EXTRA+=("$1"); shift ;;
  esac
done
[ -n "$OUT" ] || usage
mkdir -p "$(dirname "$OUT")"

command -v ros2 >/dev/null || { echo "[rec] ros2 not on PATH — source a ROS 2 setup.bash"; exit 3; }

# ---- capability probe -----------------------------------------------------
REC_HELP=$(ros2 bag record --help 2>&1)
has_opt() { grep -q -- "$1" <<< "$REC_HELP"; }
HAS_OUTPUT_OPT=0; has_opt "--output" && HAS_OUTPUT_OPT=1
HAS_PRESET=0;    has_opt "--storage-preset-profile" && HAS_PRESET=1
HAS_CHUNK=0;     has_opt "--max-chunk-size" && HAS_CHUNK=1
HAS_COMPMODE=0;  has_opt "--compression-mode" && HAS_COMPMODE=1
[ -n "$STORAGE" ] || { has_opt "mcap" && STORAGE="mcap" || STORAGE="sqlite3"; }

if [ -n "$TOPICS_FILE" ]; then
  TOPICS=()
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(echo "$line" | tr -d '[:space:]')"
    [ -n "$line" ] && TOPICS+=("$line")
  done < "$TOPICS_FILE"
else
  # -e/--regex would drop everything if one topic is absent, so name topics
  # explicitly: a rig often comes up piecemeal and a partial bag beats none.
  TOPICS=(/tf /tf_static /imu/data /odom /joint_states
          /camera/color/image_raw /camera/color/camera_info
          /camera/depth/image_raw)
fi

CMD=(ros2 bag record)
if [ "$HAS_OUTPUT_OPT" = "1" ]; then CMD+=(-o "$OUT"); else CMD+=("$OUT"); fi
CMD+=(--storage "$STORAGE" --max-cache-size "$MAX_CACHE")
[ "$HAS_PRESET" = "1" ] && CMD+=(--storage-preset-profile "$PRESET")
[ "$HAS_CHUNK" = "1" ] && CMD+=(--max-chunk-size "$MAX_CACHE")
if [ "$COMPRESSION" != "none" ] && [ "$HAS_COMPMODE" = "1" ]; then
  CMD+=(--compression-mode message --compression-format zstd)
fi
[ "$MAX_BAG_SIZE" -gt 0 ] && has_opt "-b " && CMD+=(-b "$MAX_BAG_SIZE")
# Jazzy's `--topics` and its positional `[Topic ...]` are mutually exclusive
# under this argparse, and the positional form is what Humble wants anyway, so
# topics always go positional (deprecated on Jazzy, but functional).
CMD+=("${TOPICS[@]}")
[ "${#EXTRA[@]}" -gt 0 ] && CMD+=("${EXTRA[@]}")

echo "[rec] out      : $OUT"
echo "[rec] storage  : $STORAGE   (output-opt=$HAS_OUTPUT_OPT preset=$HAS_PRESET chunk=$HAS_CHUNK comp=$HAS_COMPMODE)"
echo "[rec] topics   : ${#TOPICS[@]}"
printf '[rec]           %s\n' "${TOPICS[@]}"
echo "[rec] duration : ${DURATION:-until Ctrl-C}"
echo "[rec] cmd      : ${CMD[*]}"

"${CMD[@]}" &
BAG_PID=$!

# Forward interrupts so ros2 can close the bag properly.
trap 'echo; echo "[rec] interrupt -> closing bag..."; kill -INT "$BAG_PID" 2>/dev/null; wait "$BAG_PID" 2>/dev/null; exit 130' INT TERM
if [ "$DURATION" -gt 0 ] 2>/dev/null; then
  ( sleep "$DURATION"; kill -INT "$BAG_PID" 2>/dev/null ) &
  TIMER_PID=$!
fi

wait "$BAG_PID"; RC=$?
[ -n "${TIMER_PID:-}" ] && kill "$TIMER_PID" 2>/dev/null
echo "[rec] ros2 bag record exited rc=$RC"

# Jazzy writes metadata.yaml; Humble writes <name>.metadata.yaml.
if [ -f "$OUT/metadata.yaml" ] || compgen -G "$OUT/*.metadata.yaml" >/dev/null; then
  echo "[rec] OK — metadata written"
  python3 "$HERE/verify_bag.py" "$OUT" || echo "[rec] WARNING: bag did not pass verification"
  exit 0
fi
echo "[rec] FAIL — no metadata in $OUT (bag not finalized)"
exit 1
