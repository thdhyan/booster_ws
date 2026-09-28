#!/bin/bash
# Host launcher for the P2 velocity resume: verify the checkpoint on video first,
# then resume training with a loosened curriculum gate.
#
# Pass a checkpoint path, or leave it off to use the most recent model_2999.pt.
# The container script refuses to resume if the video did not render, so compute
# is never spent reinforcing a checkpoint nobody has watched.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/p2_vel_resume.log"
CACHE="$HERE/k1_isaac_cache_p2_vel_resume"
SESSION=k1_spark_p2_velresume

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "TMUX_SESSION_ALREADY_EXISTS=$SESSION"
  exit 1
fi
# A SIGKILLed container can leave its CUDA context wedged, which makes the next
# Isaac boot hang silently at 0% CPU. Refuse to stack runs instead.
if docker ps --format '{{.Config.Image}}' | grep -q 'isaac-lab'; then
  echo "An isaac-lab container is already running. Refusing to double-book the GPU."
  docker ps --format '{{.Names}} {{.Image}}'
  exit 2
fi

CKPT="${1:-}"
if [ -z "$CKPT" ]; then
  CKPT=$(find "$REPO/logs/rsl_rl" -type f -name model_2999.pt -printf '%T@ %p\n' 2>/dev/null \
         | sort -nr | head -1 | cut -d' ' -f2-)
fi
if [ -z "$CKPT" ] || [ ! -s "$CKPT" ]; then
  echo "NO_CHECKPOINT_FOUND (looked for logs/rsl_rl/*/model_2999.pt)"
  exit 3
fi
echo "checkpoint: $CKPT"

: > "$LOG"
rm -rf "$CACHE" 2>/dev/null || true
mkdir -p "$CACHE"
RUN_CMD="docker run --rm --gpus all --user 0 --entrypoint bash \
 -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
 -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx \
 -v $HERE/spark_p2_velresume_container.sh:/p2resume.sh:ro \
 -v $CACHE:/root/.cache -v $REPO:/workspace/booster_ws \
 nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /p2resume.sh '$CKPT' > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"

tmux new-session -d -s "$SESSION" "$RUN_CMD"
sleep 3
tmux ls | grep "$SESSION" || { echo TMUX_SESSION_MISSING; exit 1; }
echo "log: $LOG"
echo "watch: tail -f $LOG"
