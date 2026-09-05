#!/bin/bash
# Launch Isaac Sim GUI in Docker with X11 forwarding.
# Works on Wayland — Docker + X11 forwarding bypasses the viewport issue.
#
# First run: ~90s (shader cache compilation)
# Cached runs: ~35s
#
# Usage:
#   bash scripts/docker_isaac_fleet.sh
#
# After Isaac Sim loads, run the fleet script via Window > Script Editor:
#   exec(open('/workspace/booster_ws/src/k1_sim_isaac/scripts/isaac_fleet_vis.py').read())
#
# Prerequisites:
#   1. xhost +local:  (run once per login session, done automatically)
#   2. Docker with NVIDIA Container Toolkit

set -euo pipefail

IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

xhost +local: 2>/dev/null || true

# Persistent cache dirs
CACHE="$HOME/docker/isaac-sim/cache"
mkdir -p "$CACHE"/{ov/shaders,ov/ogn_generated,ov/texturecache,warp,matplotlib}
sudo chmod -R 777 "$CACHE/ov" 2>/dev/null || true

echo "╔══════════════════════════════════════════════════╗"
echo "║  Isaac Sim Fleet (Docker + X11 on Wayland)      ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Image:     $IMAGE"
echo "║  Workspace: $WORKSPACE"
echo "║  DISPLAY:   $DISPLAY"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "After Isaac Sim loads (~35s), open Window > Script Editor"
echo "and paste this to run the fleet visualization:"
echo ""
echo "  exec(open('/workspace/booster_ws/src/k1_sim_isaac/scripts/isaac_fleet_vis.py').read())"
echo ""

docker run --rm -i \
    --entrypoint /isaac-sim/runapp.sh \
    --name isaac-fleet \
    --gpus all \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=${DISPLAY}" \
    -e "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json" \
    -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
    --network=host \
    -v "$WORKSPACE:/workspace/booster_ws:rw" \
    -v "$CACHE/ov:/root/.cache/ov:rw" \
    -v "$CACHE/warp:/root/.cache/warp:rw" \
    -v "$CACHE/matplotlib:/root/.cache/matplotlib:rw" \
    -v "$HOME/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw" \
    nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 \
    --no-remote \
    --/app/livestream/enabled=false
