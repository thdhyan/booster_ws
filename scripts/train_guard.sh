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
#   * watchdog loop (5 s): kill the whole scope if a limit is breached for
#     $GUARD_HITS consecutive samples (default 3 = 15 s sustained):
#       - GPU used memory  > $GPU_MAX_MB (default 7000) or
#       - available RAM    < $RAM_MIN_MB (default 1024) or
#       - free disk        < $DISK_MIN_GB (default 5)
#     (debounce: first-run RTX shader compilation + Kit bring-up cause brief
#      available-RAM dips that must not kill an otherwise-healthy run)
#   * nice -n 10 so the desktop stays responsive
#   * only ONE guarded run at a time (lock file)
#
# Logs: logs/guard_<name>.log (stdout+stderr of the command)

set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$WS/logs"

GPU_MAX_MB="${GPU_MAX_MB:-7000}"   # used-MB cap on the WATCHED gpu (total, incl. baseline)
GPU_IDX="${GPU_IDX:-0}"            # 0-based gpu to watch — set to match CUDA_VISIBLE_DEVICES
MEM_MAX_GB="${MEM_MAX_GB:-9}"      # systemd MemoryMax for the run scope (raise on big-RAM boxes)
SWAP_MAX_GB="${SWAP_MAX_GB:-4}"    # systemd MemorySwapMax for the run scope
DISK_PATH="${DISK_PATH:-/}"        # filesystem to watch for free space (e.g. $HOME on servers)
RAM_MIN_MB="${RAM_MIN_MB:-1024}"
DISK_MIN_GB="${DISK_MIN_GB:-5}"
GUARD_HITS="${GUARD_HITS:-3}"
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

echo "[guard] name=$NAME  limits: gpu$GPU_IDX<${GPU_MAX_MB}MB RAM>${RAM_MIN_MB}MB disk($DISK_PATH)>${DISK_MIN_GB}GB scope Mem=${MEM_MAX_GB}G/swap=${SWAP_MAX_GB}G"
echo "[guard] log: $LOG"

systemd-run --user --scope --quiet \
    -p MemoryMax=${MEM_MAX_GB}G -p MemorySwapMax=${SWAP_MAX_GB}G \
    nice -n 10 "$@" >"$LOG" 2>&1 &
RUNNER=$!

KILLED=""
BREACH=""
HITS=0
while kill -0 "$RUNNER" 2>/dev/null; do
    sleep 5
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n "$((GPU_IDX + 1))p")
    RAM_FREE=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    DISK_FREE=$(df -BG --output=avail "$DISK_PATH" 2>/dev/null | tail -1 | tr -dc '0-9')
    THIS_BREACH=""
    if [ -n "$GPU_USED" ] && [ "$GPU_USED" -gt "$GPU_MAX_MB" ]; then
        THIS_BREACH="gpu$GPU_IDX ${GPU_USED}MB > ${GPU_MAX_MB}MB"
    elif [ "$RAM_FREE" -lt "$RAM_MIN_MB" ]; then
        THIS_BREACH="RAM free ${RAM_FREE}MB < ${RAM_MIN_MB}MB"
    elif [ -n "$DISK_FREE" ] && [ "$DISK_FREE" -lt "$DISK_MIN_GB" ]; then
        THIS_BREACH="disk ${DISK_FREE}GB < ${DISK_MIN_GB}GB"
    fi
    if [ -n "$THIS_BREACH" ]; then
        if [ "$THIS_BREACH" = "$BREACH" ]; then
            HITS=$((HITS + 1))
        else
            BREACH="$THIS_BREACH"
            HITS=1
        fi
        if [ "$HITS" -ge "$GUARD_HITS" ]; then
            KILLED="$BREACH (${HITS} consecutive samples)"
        fi
    else
        BREACH=""
        HITS=0
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
