#!/bin/bash
set +e
REPO="$HOME/Projects/booster_ws"
CACHE="$REPO/scripts/k1_isaac_cache_sweep"
mkdir -p "$CACHE"
docker run --rm --gpus all --user 0 --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx \
  -e SWEEP_MULTS="${SWEEP_MULTS:-1,2,3,4,6,8}" \
  -e SWEEP_STEPS="${SWEEP_STEPS:-180}" -e SWEEP_ENVS="${SWEEP_ENVS:-8}" \
  -v "$REPO/scripts/sweep_standing_container.sh:/sweep.sh:ro" \
  -v "$CACHE:/root/.cache" -v "$REPO:/workspace/booster_ws" \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /sweep.sh
echo "SWEEP_DOCKER_RC=$?"
