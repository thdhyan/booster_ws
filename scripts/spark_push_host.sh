#!/bin/bash
# Push host launcher (run on a spark): reach | push
#   reach : P6 stage 1 — walk up + wrist contact (256 envs x 1500 iters)
#   push  : P6 stage 2 — corner-goal pushing, warm-start from reach ckpt (256x3000)
# Live tail: scripts/$WHICH.push.log
set -e
WHICH="${1:?usage: spark_push_host.sh reach|push}"
case "$WHICH" in reach|push) ;; *) echo "usage: spark_push_host.sh reach|push"; exit 2 ;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/$WHICH.push.log"
: > "$LOG"
CACHE="$HERE/k1_isaac_cache"
mkdir -p "$CACHE"
tmux new-session -d -s "k1_spark_push_$WHICH" "docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e WHICH=$WHICH -v $HERE/spark_push_container.sh:/push.sh:ro -v $CACHE:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /push.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep "k1_spark_push_$WHICH" || echo "TMUX_SESSION_MISSING"
