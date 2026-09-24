#!/bin/bash
# Soccer host launcher (run on a spark): p3 | p4t
#   p3  : P3 head tracking (detection-based, cameras, 512x2000)
#   p4t : P4 kick TEACHER (privileged ball GT, 512x3000)
# Live tail: scripts/$WHICH.soccer.log
set -e
WHICH="${1:?usage: spark_soccer_host.sh p3|p4t}"
case "$WHICH" in p3|p4t) ;; *) echo "usage: spark_soccer_host.sh p3|p4t"; exit 2 ;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/$WHICH.soccer.log"
: > "$LOG"
CACHE="$HERE/k1_isaac_cache_$WHICH"
mkdir -p "$CACHE"
tmux new-session -d -s "k1_spark_$WHICH" "docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e WHICH=$WHICH -v $HERE/spark_soccer_container.sh:/soc.sh:ro -v $CACHE:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /soc.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep "k1_spark_$WHICH" || echo "TMUX_SESSION_MISSING"
