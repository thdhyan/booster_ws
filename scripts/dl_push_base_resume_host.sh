#!/bin/bash
# Host launcher (dl, GPU 1): Track B gait-v2 base RESUME training in tmux+docker
# — laptop-independent. Live tail: scripts/push_base.resume.host.log
# Dashboard: wandb booster_k1_soccer_hrl / k1_partialctrl_base.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/push_base.resume.host.log"
: > "$LOG"
mkdir -p "$HERE/k1_isaac_cache"
tmux new-session -d -s "k1_dl_push_base_resume" "docker run --rm --gpus device=1 --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /workspace/booster_ws/scripts/dl_push_base_resume_container.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep "k1_dl_push_base_resume" || { echo "TMUX_SESSION_MISSING"; exit 1; }
echo "PUSH BASE RESUME LOG: $LOG"
