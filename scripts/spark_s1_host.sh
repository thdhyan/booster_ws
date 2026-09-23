#!/bin/bash
# S1 host launcher (run on spark04): bind-mounts the staged repo + warp/kit cache and
# runs the S1 container script (our-stack student smoke) inside tmux.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/s1.log"
: > "$LOG"
mkdir -p "$HERE/k1_isaac_cache"
tmux new-session -d -s k1_spark_s1 "docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/spark_s1_container.sh:/s1.sh:ro -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /s1.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep k1_spark_s1 || echo "TMUX_SESSION_MISSING"
