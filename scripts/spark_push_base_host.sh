#!/bin/bash
# Host launcher (spark04): Track B frozen-base retrain on the gait-v2 velocity
# config.  Smoke-gated FULL partial-control run in tmux + docker — survives
# laptop disconnect/reboot.  Live tail: scripts/push_base.full.log
# Dashboard: wandb booster_k1_soccer_hrl / k1_partialctrl_base.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/push_base.host.log"
: > "$LOG"
mkdir -p "$HERE/k1_isaac_cache"
tmux new-session -d -s "k1_spark_push_base" "docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/spark_push_base_container.sh:/base.sh:ro -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /base.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep "k1_spark_push_base" || { echo "TMUX_SESSION_MISSING"; exit 1; }
echo "PUSH BASE RETRAIN LOG: $LOG"
