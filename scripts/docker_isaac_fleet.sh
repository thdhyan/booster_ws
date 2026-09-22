#!/bin/bash
# Launch Isaac Sim Python standalone in Docker with X11 forwarding.
# Runs the K1 fleet visualization automatically via --exec.
#
# First run: ~90s (shader cache compilation + pyyaml install)
# Cached runs: ~35s
#
# Usage:
#   bash scripts/docker_isaac_fleet.sh [--n_robots N] [--cmd-vx V] [--headless]
#
# Prerequisites:
#   1. xhost +local:  (run once per login session, done automatically)
#   2. Docker with NVIDIA Container Toolkit

set -euo pipefail

IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

# Parse args
N_ROBOTS=2
CMD_VX=0.3
HEADLESS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_robots) N_ROBOTS="$2"; shift 2 ;;
        --cmd-vx) CMD_VX="$2"; shift 2 ;;
        --headless) HEADLESS="--headless"; shift ;;
        *) shift ;;
    esac
done

xhost +local: 2>/dev/null || true

# Persistent cache dirs
CACHE="$HOME/docker/isaac-sim/cache"
mkdir -p "$CACHE"/{ov/shaders,ov/ogn_generated,ov/texturecache,warp,matplotlib}
sudo chmod -R 777 "$CACHE/ov" 2>/dev/null || true

SCRIPT_PATH="/workspace/booster_ws/src/k1_sim_isaac/scripts/isaac_fleet_vis.py"
FLEET_ARGS="--n_robots $N_ROBOTS --cmd-vx $CMD_VX $HEADLESS"

echo "╔══════════════════════════════════════════════════╗"
echo "║  K1 Fleet — Isaac Sim Python (Docker + X11)     ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Image:     $IMAGE"
echo "║  Workspace: $WORKSPACE"
echo "║  DISPLAY:   $DISPLAY"
echo "║  Robots:    $N_ROBOTS"
echo "║  Cmd vel:   ${CMD_VX} m/s"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Build the command to run inside the container
# 1. Install pyyaml + typing_extensions if missing
# 2. Patch Warp ProxyArray bug in articulation_data.py
# 3. Set bundled ROS2 env vars
# 4. Run fleet script via Kit --exec
INNER_CMD="
    /isaac-sim/kit/python/bin/pip3 install -q pyyaml typing_extensions 2>/dev/null || true

    # Fix Warp kernel ProxyArray bug: body_com_pose_w passes ProxyArray to kernel
    # that expects wp.array. Patch two lines in articulation_data.py.
    ARTDATA=/workspace/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py
    if [ -f \"\$ARTDATA\" ] && ! grep -q '_PATCHED_WARP_FIX' \"\$ARTDATA\" 2>/dev/null; then
        echo \"[patch] Fixing Warp ProxyArray bug in articulation_data.py ...\"
        /isaac-sim/kit/python/bin/python3 -c \"
import re
path = '\$ARTDATA'
with open(path) as f: src = f.read()
# Fix 1: In body_com_pose_w wp.launch, use raw wp.array instead of ProxyArray
src = src.replace(
    'self.body_link_pose_w,\\n                    self.body_com_pose_b,',
    'self._body_link_pose_w.data,\\n                    self.body_com_pose_b,',
    1
)
# Fix 2: Return wp.array instead of ProxyArray from body_com_pose_w
src = src.replace(
    'return self._body_com_pose_w_ta',
    'return self._body_com_pose_w.data',
    1
)
# Mark as patched
src = src.replace(
    'class ArticulationData',
    '# _PATCHED_WARP_FIX\\nclass ArticulationData',
    1
)
with open(path, 'w') as f: f.write(src)
print('[patch] Done')
\"
    fi

    export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/rclpy:\$PYTHONPATH
    export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:\$LD_LIBRARY_PATH
    export AMENT_PREFIX_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy
    cd /workspace/booster_ws
    /isaac-sim/kit/kit /isaac-sim/apps/isaacsim.exp.base.python.kit \
        --no-remote \
        --exec $SCRIPT_PATH $FLEET_ARGS
"

docker run --rm -i \
    --entrypoint /bin/dash \
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
    "$IMAGE" \
    -c "$INNER_CMD"
