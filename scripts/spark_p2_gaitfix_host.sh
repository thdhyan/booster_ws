#!/bin/bash
# Host launcher for the P2 gait-structured campaign.
#
# Runs TWO containers in sequence, never two Kit launches in one:
#   1. preflight  - builds the env, checks the robot stands and every reward term
#                   is live. Exits without training.
#   2. train      - smoke, 3000 iterations, then the movement and gait gates.
#
# AppLauncher deadlocks on a second Kit launch inside one container. That cost
# several cycles to diagnose: train.py sat at 0% CPU and 43 MiB with zero bytes
# of output, right after the preflight had booted Kit successfully in the same
# container. One launch per container removes the whole failure mode.
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/p2_gaits.log"
SESSION=k1_spark_p2_gaits

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "TMUX_SESSION_ALREADY_EXISTS=$SESSION"
  exit 1
fi
if docker ps --format '{{.Image}}' | grep -q 'isaac-lab'; then
  echo "An isaac-lab container is already running. Refusing to double-book the GPU."
  docker ps --format '{{.Names}} {{.Image}}'
  exit 2
fi

: > "$LOG"

# Fresh cache per container. The container runs as --user 0 so its cache files are
# root-owned and cannot be removed from the host, and a stale ov/_cache.lock wedges
# the next Kit boot at 0% CPU with no output. Never deleting is the robust answer.
CACHE_A="$HERE/k1_isaac_cache_gaits_pre_$(date +%Y%m%d_%H%M%S)"
CACHE_B="$HERE/k1_isaac_cache_gaits_trn_$(date +%H%M%S)"
mkdir -p "$CACHE_A" "$CACHE_B"
echo "caches: $CACHE_A  $CACHE_B"

run_stage() {   # $1 = container script, $2 = cache dir, $3 = label
  echo "=== STAGE $3 $(date +%H:%M:%S)"
  docker run --rm --gpus all --user 0 --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
    -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx \
    -v "$1":/stage.sh:ro \
    -v "$2":/root/.cache -v "$REPO":/workspace/booster_ws \
    nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /stage.sh 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_stage "$HERE/spark_p2_gaitfix_preflight_container.sh" "$CACHE_A" preflight
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "P2_GAITS_GATE=PREFLIGHT_FAILED_RC_$rc_NO_TRAINING"
  echo "DOCKER_RC=$rc" >> "$LOG"
  exit 30
fi
# The GPU must be free between containers; the previous one is --rm so it is gone,
# but confirm rather than assume.
sleep 5
if docker ps --format '{{.Image}}' | grep -q 'isaac-lab'; then
  echo "P2_GAITS_GATE=PREVIOUS_CONTAINER_STILL_RUNNING"
  exit 31
fi

run_stage "$HERE/spark_p2_gaitfix_train_container.sh" "$CACHE_B" train
rc=$?
echo "DOCKER_RC=$rc" >> "$LOG"
echo "log: $LOG"
exit "$rc"
