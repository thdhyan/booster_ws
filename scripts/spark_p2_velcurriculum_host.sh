#!/bin/bash
# Host launcher for the P2 velocity-curriculum campaign on a GB10 Spark.
# The container script owns the preflight/smoke/full gates and the curriculum
# marker; this only mounts the repo and runs it in tmux.
#
#   ./spark_p2_velcurriculum_host.sh          # background in tmux, returns at once
#   tail -f scripts/p2_vel.log                # watch it
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/p2_vel.log"
CACHE="$HERE/k1_isaac_cache_p2_vel"   # reset below before use
SESSION=k1_spark_p2_vel

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
# Run-specific cache. A cache shared with a previous run can wedge Kit at boot
# (0% CPU, ~29 MiB, zero output) if the earlier container was SIGKILLed rather
# than stopped, and the resulting files are root-owned from inside `--user 0` so
# they cannot be cleaned from the host afterwards.
CACHE="${K1_CACHE_DIR:-$HERE/k1_isaac_cache_p2_vel}"
rm -rf "$CACHE" 2>/dev/null || true
mkdir -p "$CACHE"
echo "cache: $CACHE"
RUN_CMD="docker run --rm --gpus all --user 0 --entrypoint bash \
 -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
 -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx \
 -v $HERE/spark_p2_velcurriculum_container.sh:/p2vel.sh:ro \
 -v $CACHE:/root/.cache -v $REPO:/workspace/booster_ws \
 nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /p2vel.sh > $LOG 2>&1; echo DOCKER_RC=\$? >> \$LOG"

tmux new-session -d -s "$SESSION" "$RUN_CMD"
sleep 3
tmux ls | grep "$SESSION" || { echo TMUX_SESSION_MISSING; exit 1; }
echo "P2 velocity-curriculum log: $LOG"
echo "watch:  tail -f $LOG"
echo "stop:   tmux kill-session -t $SESSION   # then: docker kill \$(docker ps -q -f isaac-lab)"
