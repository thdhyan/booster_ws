#!/bin/bash
# Run the reward probe for one task inside the Isaac Lab container.
#   PROBE_TASK=Isaac-Velocity-Rough-K1-Teacher-v0 bash scripts/probe_rewards_host.sh
set +e
REPO="$HOME/Projects/booster_ws"
CACHE="$REPO/scripts/k1_isaac_cache_probe"
mkdir -p "$CACHE"
docker run --rm --gpus all --user 0 --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx \
  -e PROBE_TASK="${PROBE_TASK:-Isaac-Velocity-Rough-K1-Teacher-v0}" \
  -e PROBE_ENVS="${PROBE_ENVS:-16}" -e PROBE_STEPS="${PROBE_STEPS:-60}" \
  -v "$REPO/scripts/probe_rewards_container.sh:/probe.sh:ro" \
  -v "$CACHE:/root/.cache" -v "$REPO:/workspace/booster_ws" \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /probe.sh
echo "PROBE_DOCKER_RC=$?"
