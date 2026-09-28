#!/bin/bash
# Host launcher (dl, GPU 2): render the gait-v2 base PROGRESS VIDEO at the last
# surviving checkpoint (iter 800/3000, spark04 died at ~iter 840 on 2026-09-24).
# Output: isaac_tasks/k1_velocity/videos/partial_gaitv2_iter800.mp4
# Uses its own cache dir (built in parallel with the resume training's cache).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/dl_partial800_record.log"
CKPT="$REPO/logs/spark04/rsl_rl/k1_partialctrl_base/2026-09-24_18-44-39_k1_partialctrl_base/model_800.pt"
OUT="$REPO/isaac_tasks/k1_velocity/videos/partial_gaitv2_iter800.mp4"
[ -s "$CKPT" ] || { echo "NO_CKPT $CKPT"; exit 1; }
mkdir -p "$HERE/k1_isaac_cache_rec" "$(dirname "$OUT")"
tmux new-session -d -s k1_dl_partial800_rec "docker run --rm --gpus device=2 --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/k1_isaac_cache_rec:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /workspace/booster_ws/scripts/play_partial_diag.sh /workspace/booster_ws/logs/spark04/rsl_rl/k1_partialctrl_base/2026-09-24_18-44-39_k1_partialctrl_base/model_800.pt /workspace/booster_ws/isaac_tasks/k1_velocity/videos/partial_gaitv2_iter800.mp4 > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep k1_dl_partial800_rec || { echo "TMUX_SESSION_MISSING"; exit 1; }
echo "RECORD LOG: $LOG"
