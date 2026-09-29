#!/bin/bash
# Host launcher for the P2 velocity-curriculum campaign on a GB10 Spark.
# The container script owns the preflight/smoke/full gates and the curriculum
# marker; this only mounts the repo and runs it in tmux.
#
#   ./spark_p2_velcurriculum_host.sh          # background in tmux, returns at once
#   tail -f scripts/p2_gaits.log                # watch it
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/p2_gaits.log"
SESSION=k1_spark_p2_gaits

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "TMUX_SESSION_ALREADY_EXISTS=$SESSION -- not starting a second run"
  exit 1
fi
# A previous run's container outlives its tmux session, so check the host too.
# Two training containers on one GPU would corrupt both runs.
# Match on the image, not the name: container names are auto-generated
# (e.g. "vibrant_rubin"), so grepping .Names never matches anything.
if docker ps --format '{{.Image}}' | grep -q 'isaac-lab'; then
  echo "An isaac-lab container is already running. Refusing to double-book the GPU."
  docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
  exit 2
fi

: > "$LOG"
# A FRESH cache directory per run, and never try to delete an old one.
#
# The container runs as --user 0, so everything it writes into the mounted cache
# is root-owned and `rm -rf` as thakk100 fails silently. A stale
# ov/_cache.lock left by the previous run then blocks the next Kit boot forever:
# the process sits at 0% CPU and ~43 MiB with no output and no exception, which
# cost several cycles to diagnose. A unique directory means there is never a
# stale lock to inherit, which is more robust than trying to clean one.
CACHE="${K1_CACHE_DIR:-$HERE/k1_isaac_cache_gaits_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$CACHE"
echo "cache: $CACHE"
RUN_CMD="docker run --rm --gpus all --user 0 --entrypoint bash \
 -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
 -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx \
 -v $HERE/spark_p2_gaitfix_container.sh:/p2vel.sh:ro \
 -v $CACHE:/root/.cache -v $REPO:/workspace/booster_ws \
 nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /p2vel.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> \$LOG"

tmux new-session -d -s "$SESSION" "$RUN_CMD"
sleep 3
tmux ls | grep "$SESSION" || { echo TMUX_SESSION_MISSING; exit 1; }
echo "P2 velocity-curriculum log: $LOG"
echo "watch:  tail -f $LOG"
echo "stop:   tmux kill-session -t $SESSION   # then: docker kill \$(docker ps -q -f isaac-lab)"
