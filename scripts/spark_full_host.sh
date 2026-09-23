#!/bin/bash
# Full-student host launcher (run on spark04): p1|p2.
# S1/S2 smokes already PASSED on this image, so this goes straight to the FULL run
# (queue_student.sh's ENVS/ITERS recipe). One tmux session per run, live session log.
set -e
WHICH="${1:?usage: spark_full_host.sh p1|p2}"
case "$WHICH" in p1|p2) ;; *) echo "usage: spark_full_host.sh p1|p2" >&2; exit 2 ;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/$WHICH.full.log"
: > "$LOG"
mkdir -p "$HERE/k1_isaac_cache"
tmux new-session -d -s "k1_spark_$WHICH" "docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e WHICH=$WHICH -v $HERE/spark_full_container.sh:/full.sh:ro -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /full.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep "k1_spark_$WHICH" || echo "TMUX_SESSION_MISSING"
