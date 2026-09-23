#!/bin/bash
# S0 host launcher (run on the spark): boots the S0 container smoke inside tmux.
# Everything lives at home ROOT (no subdirectory) — see spark notes.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/s0.log"
: > "$LOG"
tmux new-session -d -s k1_spark_s0 "docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e WANDB_MODE=disabled -v $HERE/spark_s0_container.sh:/s0.sh:ro -v $HERE/k1_isaac_cache:/root/.cache nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /s0.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep k1_spark_s0 || echo "TMUX_SESSION_MISSING"
