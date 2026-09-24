#!/bin/bash
# Host launcher for the revised P2 gait campaign on a GB10 Spark.
# The container script owns the smoke/full gates and writes scripts/p2_gait.*.log.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/p2_gait.log"
CACHE="$HERE/k1_isaac_cache_p2_gait"
: > "$LOG"
mkdir -p "$CACHE"
RUN_CMD="docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx -v $HERE/spark_p2_gait_container.sh:/p2.sh:ro -v $CACHE:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /p2.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
if [ "${WAIT_FOR_P3:-0}" = "1" ]; then
  P3_LOG="$REPO/scripts/p3.soccer.log"
  P4_LOG="$REPO/scripts/p4t.soccer.log"
  RUN_CMD="echo P2_GAIT_QUEUE=WAITING_FOR_TRACK_A; until grep -qE 'SOC_FULL_MARKER=(OK|FAIL)' '$P3_LOG'; do sleep 60; done; grep -q 'SOC_FULL_MARKER=OK' '$P3_LOG' || exit 21; until grep -qE 'SOC_FULL_MARKER=(OK|FAIL)' '$P4_LOG'; do sleep 60; done; grep -q 'SOC_FULL_MARKER=OK' '$P4_LOG' || exit 22; $RUN_CMD"
fi
tmux new-session -d -s k1_spark_p2_gait "$RUN_CMD"
sleep 2
tmux ls | grep k1_spark_p2_gait || { echo TMUX_SESSION_MISSING; exit 1; }
echo "P2 gait log: $LOG"
