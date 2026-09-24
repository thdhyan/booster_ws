#!/bin/bash
# F-variant host launcher (run on spark04): p1f|p2f — smoke-gated teacher PPO
# then CHAINED student distillation from the teacher's final checkpoint.
# Live tail: scripts/$WHICH.f.log; wandb = booster_k1_soccer_hrl dashboard.
set -e
WHICH="${1:?usage: spark_force_host.sh p1f|p2f}"
case "$WHICH" in p1f|p2f) ;; *) echo "usage: spark_force_host.sh p1f|p2f"; exit 2 ;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/$WHICH.f.log"
: > "$LOG"
mkdir -p "$HERE/k1_isaac_cache"
tmux new-session -d -s "k1_spark_$WHICH" "docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e WHICH=$WHICH -e RESUME_CKPT=${RESUME_CKPT:-} -v $HERE/spark_force_container.sh:/full.sh:ro -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /full.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep "k1_spark_$WHICH" || echo "TMUX_SESSION_MISSING"
