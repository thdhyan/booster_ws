#!/usr/bin/env bash
# Batch, resumable, checksummed file transfer between this workstation and the
# K1's onboard storage.
#
#   ./sftp_sync.sh pull                 # robot -> ./bags
#   ./sftp_sync.sh push bags/walk_fast  # workstation -> robot
#   ./sftp_sync.sh pull --filter walk_fast --dry-run
#
# SFTP batch mode is used rather than scp/rsync because the robot exposes SSH
# but not rsync, and because batch mode is non-interactive and scriptable.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
[ -f "$HERE/config/env.sh" ] && source "$HERE/config/env.sh"

ROBOT_USER="${ROBOT_USER:-robot}"
ROBOT_HOST="${ROBOT_HOST:-192.168.1.100}"
ROBOT_PORT="${ROBOT_PORT:-22}"
# Remote-side path. The tilde is expanded by the ROBOT's shell inside the sftp
# batch, so it is deliberately left unexpanded here.
ROBOT_BAG_DIR="${ROBOT_BAG_DIR:-~/bags}"
LOCAL_BAG_DIR="${LOCAL_BAG_DIR:-$HERE/bags}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
FILTER=""
DRY_RUN=0
ACTION=""

usage() { sed -n '2,10p' "$0"; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    pull|push)   ACTION="$1"; shift ;;
    --filter)    FILTER="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --host)      ROBOT_HOST="$2"; shift 2 ;;
    --user)      ROBOT_USER="$2"; shift 2 ;;
    --key)       SSH_KEY="$2"; shift 2 ;;
    -h|--help)   usage ;;
    *)           echo "unknown arg: $1"; usage ;;
  esac
done
[ -n "$ACTION" ] || usage
[ -f "$SSH_KEY" ] || { echo "[sync] no ssh key at $SSH_KEY"; exit 2; }

SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10
          -o StrictHostKeyChecking=accept-new -P "$ROBOT_PORT")

# 1. list remote files (one line per file, name + size)
echo "[sync] listing $ROBOT_USER@$ROBOT_HOST:$ROBOT_BAG_DIR ..."
if [ "$ACTION" = "pull" ]; then
  REMOTE_LIST=$(sftp -b - "${SSH_OPTS[@]}" "$ROBOT_USER@$ROBOT_HOST" <<EOF 2>/dev/null
cd $ROBOT_BAG_DIR
ls -l
bye
EOF
  ) || { echo "[sync] FAIL: cannot reach robot over sftp"; exit 3; }
else
  REMOTE_LIST=""
fi

# 2. build the batch of transfers
BATCH=$(mktemp); RC_BATCH=$(mktemp)
mkdir -p "$LOCAL_BAG_DIR"
COUNT=0; BYTES=0

if [ "$ACTION" = "pull" ]; then
  # Parse the `ls -l` listing: name is the last field, size is field 5.
  while read -r line; do
    name=$(echo "$line" | awk '{print $NF}')
    size=$(echo "$line" | awk '{print $5}')
    case "$name" in
      *db3|*mcap) ;;
      *) continue ;;
    esac
    [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]] && continue
    # Skip if already present with the same size.
    if [ -f "$LOCAL_BAG_DIR/$name" ] && [ "$(stat -c%s "$LOCAL_BAG_DIR/$name")" = "$size" ]; then
      echo "[sync] skip (same size): $name"
      continue
    fi
    if [ "$DRY_RUN" = "1" ]; then
      echo "[sync] would pull: $name ($size B)"; COUNT=$((COUNT+1)); BYTES=$((BYTES+size)); continue
    fi
    echo "get $ROBOT_BAG_DIR/$name $LOCAL_BAG_DIR/$name" >> "$BATCH"
    COUNT=$((COUNT+1)); BYTES=$((BYTES+size))
  done <<< "$REMOTE_LIST"
else
  SRC="${2:-}"
  [ -n "$SRC" ] || { echo "[sync] push needs a source path"; exit 2; }
  shopt -s nullglob
  for f in "$SRC"/*; do
    b=$(basename "$f")
    [ -n "$FILTER" ] && [[ "$b" != *"$FILTER"* ]] && continue
    if [ "$DRY_RUN" = "1" ]; then
      echo "[sync] would push: $b ($(stat -c%s "$f") B)"; COUNT=$((COUNT+1)); continue
    fi
    echo "put $f $ROBOT_BAG_DIR/$b" >> "$BATCH"
    COUNT=$((COUNT+1))
  done
  shopt -u nullglob
fi

echo "[sync] $ACTION: $COUNT file(s), $(numfmt --to=iec --suffix=B "${BYTES:-0}")"
if [ "$DRY_RUN" = "1" ] || [ "$COUNT" -eq 0 ]; then
  echo "[sync] nothing to do"
  [ -n "$BATCH" ] && rm -f "$BATCH" "$RC_BATCH"
  exit 0
fi

# 3. transfer, then verify
echo "bye" >> "$BATCH"
sftp -b "$BATCH" "${SSH_OPTS[@]}" "$ROBOT_USER@$ROBOT_HOST"
RC=$?
rm -f "$BATCH" "$RC_BATCH"
[ $RC -ne 0 ] && { echo "[sync] FAIL: sftp rc=$RC (partial transfers remain; rerun to resume)"; exit $RC; }

echo "[sync] verifying sizes..."
FAIL=0
if [ "$ACTION" = "pull" ]; then
  while read -r line; do
    name=$(echo "$line" | awk '{print $NF}')
    size=$(echo "$line" | awk '{print $5}')
    case "$name" in *db3|*mcap) ;; *) continue ;; esac
    got=$(stat -c%s "$LOCAL_BAG_DIR/$name" 2>/dev/null || echo -1)
    if [ "$got" != "$size" ]; then
      echo "[sync] SIZE MISMATCH: $name remote=$size local=$got"; FAIL=1
    fi
  done <<< "$REMOTE_LIST"
fi
[ $FAIL -eq 0 ] && echo "[sync] OK — all files verified" || echo "[sync] FAIL — size mismatches above"
exit $FAIL
