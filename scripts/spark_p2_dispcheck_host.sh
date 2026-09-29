#!/bin/bash
# Wait for the P2 velocity-resume training to finish, then automatically run the
# displacement check so the GPU is never idle waiting for a human to notice.
#
# The trigger is DOCKER_RC= appearing in the resume log, which is written when the
# training container exits. It also requires no isaac-lab container to be running,
# because eval and training on one GPU corrupt both -- the rule that matters most
# here, since this script exists precisely to start work the instant training ends.
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
RESUME_LOG="$HERE/p2_vel_resume.log"
OUT_LOG="$HERE/p2_dispcheck.log"
SESSION=k1_spark_p2_dispwait
MAX_WAIT="${MAX_WAIT:-21600}"   # 6 h hard stop, so this cannot wait forever

: > "$OUT_LOG"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT_LOG"; }

log "watching $RESUME_LOG for the training container to exit (max ${MAX_WAIT}s)"

waited=0
while [ "$waited" -lt "$MAX_WAIT" ]; do
  if grep -q 'DOCKER_RC=' "$RESUME_LOG" 2>/dev/null; then
    log "training container exited: $(grep -o 'DOCKER_RC=[0-9]*' "$RESUME_LOG" | tail -1)"
    break
  fi
  sleep 30
  waited=$((waited + 30))
  if [ $((waited % 1800)) -eq 0 ]; then
    log "still waiting (${waited}s); iteration: $(grep -oE 'Learning iteration [0-9]+/5999' "$RESUME_LOG" 2>/dev/null | tail -1)"
  fi
done

if ! grep -q 'DOCKER_RC=' "$RESUME_LOG" 2>/dev/null; then
  log "TIMEOUT after ${waited}s without the training container exiting; not starting eval"
  exit 1
fi

# The GPU must actually be free. Do not trust the log alone.
for i in $(seq 1 40); do
  running=$(docker ps --format '{{.Image}}' | grep -c 'isaac-lab')
  [ "$running" -eq 0 ] && break
  log "waiting for isaac-lab container to clear ($running still up)"
  sleep 15
done
if [ "$(docker ps --format '{{.Image}}' | grep -c 'isaac-lab')" -ne 0 ]; then
  log "ABORT: an isaac-lab container is still running; refusing to eval on a shared GPU"
  exit 2
fi
log "GPU is free; launching displacement check"

# Take the most recent checkpoint the resume run wrote. Guessing an exact
# model_<N>.pt is fragile: the run resumed at iteration 3000, so the final
# filename depends on how many further iterations it completed.
CKPT=$(find "$REPO/logs/rsl_rl" -type f -name 'model_*.pt' -newermt '-8 hours' \
       -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { log "no checkpoint found to evaluate"; exit 3; }
CKPT_REL="${CKPT#$REPO/}"
log "evaluating $CKPT_REL"

# Fresh dir per run: the container runs as root so old cache files cannot be
# deleted from here, and a stale ov/_cache.lock wedges the next Kit boot
# at 0% CPU with no output.
CACHE="$HERE/k1_isaac_cache_$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CACHE"
docker run --rm --gpus all --user 0 --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx \
  -v "$HERE/spark_p2_dispcheck_container.sh:/disp.sh:ro" \
  -v "$CACHE:/root/.cache" -v "$REPO:/workspace/booster_ws" \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /disp.sh "$CKPT_REL" >> "$OUT_LOG" 2>&1
log "DISPCHECK_DOCKER_RC=$?"
