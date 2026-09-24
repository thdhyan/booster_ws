#!/bin/bash
# HP host launcher (run on spark02): bind-mounts the staged repo + warp/kit cache
# and runs spark_partial_container.sh (SMOKE-gated FULL partial-control run)
# inside tmux session k1_spark_hp.  Live tail: scripts/hp.full.log
# Dashboard: wandb booster_k1_soccer_hrl / k1_partialctrl_base.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/hp.full.log"
: > "$LOG"
mkdir -p "$HERE/k1_isaac_cache"
tmux new-session -d -s "k1_spark_hp" "docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/spark_partial_container.sh:/hp.sh:ro -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /hp.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep "k1_spark_hp" || echo "TMUX_SESSION_MISSING"
