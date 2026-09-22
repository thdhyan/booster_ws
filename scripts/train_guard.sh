#!/usr/bin/env bash
# train_guard.sh — run a training command under hard resource caps so the
# desktop never dies (Phase-6 PC-safety rule, PLAN_PHASE6_SOCCER_HRL §3.7).
#
# Usage:   ./scripts/train_guard.sh [--name RUN_NAME] -- <command...>
# Example: ./scripts/train_guard.sh --name p1_teacher -- "$VENV/bin/python train.py --task ... "
#
# Enforcement:
#   * systemd user scope: MemoryMax=9G, MemorySwapMax=4G  (OOM killer hits THIS
#     scope first; desktop RAM stays reserved)
#   * watchdog loop (5 s): kill the whole scope if
#       - GPU used memory  > $GPU_MAX_MB (default 7000) or
#       - free RAM         < $RAM_MIN_MB (default 1024) or
#       - free disk        < $DISK_MIN_GB (default 5)
#   * nice -n 10 so the desktop stays responsive
#   * only ONE guarded run at a time (lock file)
#
# Logs: logs/guard_<name>.log (stdout+stderr of the command)

set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$WS/logs"

GPU_MAX_MB="${GPU_MAX_MB:-7000}"
RAM_MIN_MB="${RAM_MIN_MB:-1024}"
DISK_MIN_GB="${DISK_MIN_GB:-5}"
LOCK=/tmp/k1_train_guard.lock

NAME="run"
if [ "${1:-}" = "--name" ]; then NAME="$2"; shift 2; fi
if [ "${1:-}" = "--" ]; then shift; fi
[ $# -ge 1 ] || { echo "usage: train_guard.sh [--name N] -- <cmd...>" >&2; exit 2; }

# single-run lock (flock-free, atomic via mkdir)
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "ERROR: another guarded run holds $LOCK (remove it if stale: rmdir $LOCK)" >&2
    exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

LOG="$WS/logs/guard_${NAME}.log"
SCOPE="k1train.${NAME}.$$"

echo "[guard] name=$NAME  limits: GPU<${GPU_MAX_MB}MB RAM>${RAM_MIN_MB}MB disk>${DISK_MIN_GB}GB"
echo "[guard] log: $LOG"

systemd-run --user --scope --quiet \
    -p MemoryMax=9G -p MemorySwapMax=4G \
    -p "ManagedOOMPreference=nodest" \
    nice -n 10 "$@" >"$LOG" 2>&1 &
RUNNER=$!

KILLED=""
while kill -0 "$RUNNER" 2>/dev/null; do
    sleep 5
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    RAM_FREE=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    DISK_FREE=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "$GPU_USED" ] && [ "$GPU_USED" -gt "$GPU_MAX_MB" ]; then
        KILLED="GPU ${GPU_USED}MB > ${GPU_MAX_MB}MB"
    elif [ "$RAM_FREE" -lt "$RAM_MIN_MB" ]; then
        KILLED="RAM free ${RAM_FREE}MB < ${RAM_MIN_MB}MB"
    elif [ -n "$DISK_FREE" ] && [ "$DISK_FREE" -lt "$DISK_MIN_GB" ]; then
        KILLED="disk ${DISK_FREE}GB < ${DISK_MIN_GB}GB"
    fi
    if [ -n "$KILLED" ]; then
        echo "[guard] LIMIT HIT: $KILLED — killing training scope" | tee -a "$LOG"
        pkill -f "systemd-run --user --scope" 2>/dev/null # best effort
        systemctl --user stop "$SCOPE" 2>/dev/null
        kill "$RUNNER" 2>/dev/null
        # ensure descendants (python) die too
        pkill -KILL -f "$$" 2>/dev/null
        break
    fi
done

wait "$RUNNER"; RC=$?
if [ -n "$KILLED" ]; then
    echo "[guard] TERMINATED by resource guard: $KILLED" | tee -a "$LOG"
    exit 99
fi
echo "[guard] command finished rc=$RC"
exit "$RC"
