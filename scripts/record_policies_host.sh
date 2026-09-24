#!/bin/bash
# REC host launcher (run on spark04): bind-mounts the staged repo + warp/kit
# cache and runs record_policies_container.sh inside tmux session k1_spark_rec.
# Live tail: scripts/record.log
#   SKIP_CORE=1 scripts/record_policies_host.sh   # only the partial video (later)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/record.log"
: > "$LOG"
mkdir -p "$HERE/k1_isaac_cache"
tmux new-session -d -s "k1_spark_rec" "docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e SKIP_CORE=${SKIP_CORE:-0} -v $HERE/record_policies_container.sh:/rec.sh:ro -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /rec.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep "k1_spark_rec" || echo "TMUX_SESSION_MISSING"
