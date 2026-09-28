#!/bin/bash
# Build / start / stop the K1 VR teleop stack on dl (sim + Quest XR node + YOLO).
# Plan and wiring: docs/vr_teleop_k1_plan.md.
#
#   scripts/dl_k1_teleop_up.sh build     # teleop image (k1-teleop-xr:$TAG), once per host
#   scripts/dl_k1_teleop_up.sh all       # sim + teleop + yolo
#   scripts/dl_k1_teleop_up.sh sim|teleop|yolo
#   scripts/dl_k1_teleop_up.sh down      # stop everything (shared machine: always do this)
#   scripts/dl_k1_teleop_up.sh status
#
# Containers (host network, ROS domain $K1_ROS_DOMAIN_ID, discovery localhost only):
#   k1-teleop-sim   Isaac Lab: src/k1_sim_isaac/scripts/k1_teleop_sim.py (debug video in logs/k1_teleop_sim/videos)
#   k1-teleop-xr    CloudXR + Televiz: stereo ZED / chase / YOLO panels, Quest controls
#                   (ports 48322/tcp 49100/tcp 47998/udp; headset: https://nvidia.github.io/IsaacTeleop/client/)
#   k1-teleop-yolo  YOLOv8n (CPU) on the left ZED eye -> /k1_0/yolo/image_raw
# Overrides: TAG, SIM_GPU, XR_GPU, K1_ROS_DOMAIN_ID, TELEOP_ARGS (e.g. "--arm-src soma"), SIM_ARGS, REPO.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
TAG=${TAG:-dl}
SIM_GPU=${SIM_GPU:-2} XR_GPU=${XR_GPU:-2}
DOMAIN=${K1_ROS_DOMAIN_ID:-45}                       # G1 teleop uses 44
TELEOP_ARGS=${TELEOP_ARGS:---arm-src ik}
SIM_ARGS=${SIM_ARGS:-}
ISAAC_IMAGE=${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1}
XR_IMAGE=k1-teleop-xr:$TAG
ROS_ENV=(-e ROS_DOMAIN_ID="$DOMAIN" -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST -e ROS_LOG_DIR=/tmp/ros_log)

start() {  # name, docker run args...
    docker rm -f "$1" >/dev/null 2>&1 || true
    docker run -d --name "$@" >/dev/null && echo "started $1"
}
case "${1:-all}" in
    build)
        docker build -t "$XR_IMAGE" -f "$REPO/docker/Dockerfile.teleop-xr" "$REPO/docker"
        docker pull "$ISAAC_IMAGE"
        mkdir -p "$HOME/.cloudxr" "$REPO/logs/k1_teleop_cache" ;;
    sim)
        start k1-teleop-sim --gpus "device=$SIM_GPU" --network host --ipc host --user 0 --entrypoint bash \
            -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 "${ROS_ENV[@]}" \
            -v "$REPO/logs/k1_teleop_cache:/root/.cache" -v "$REPO:/workspace/booster_ws" "$ISAAC_IMAGE" \
            -c "cd /workspace/booster_ws && /isaac-sim/python.sh -m pip install --no-deps -e src/k1_description/assets >/dev/null 2>&1;
                exec /isaac-sim/python.sh src/k1_sim_isaac/scripts/k1_teleop_sim.py $SIM_ARGS" ;;
    teleop)
        start k1-teleop-xr --gpus "device=$XR_GPU" --network host --ipc host --user "$(id -u):$(id -g)" \
            -e NVIDIA_DRIVER_CAPABILITIES=all -e HOME=/tmp "${ROS_ENV[@]}" \
            -v "$REPO:/workspace/booster_ws" -v "$HOME/.cloudxr:/cloudxr" -w /workspace/booster_ws "$XR_IMAGE" \
            bash -lc "source /opt/ros/jazzy/setup.bash && exec python3 -u -m k1_teleop.xr_teleop_node \
                --cloudxr-install-dir /cloudxr --accept-eula $TELEOP_ARGS" ;;
    yolo)
        start k1-teleop-yolo --network host --ipc host --user "$(id -u):$(id -g)" -e HOME=/tmp \
            -e YOLO_CONFIG_DIR=/tmp -e PYTHONPATH=/workspace/booster_ws "${ROS_ENV[@]}" \
            -v "$REPO:/workspace/booster_ws:ro" -w /tmp "$XR_IMAGE" \
            bash -lc "source /opt/ros/jazzy/setup.bash && exec python3 -u -m k1_teleop.yolo_node --device cpu" ;;
    all) "$0" sim; "$0" teleop; "$0" yolo ;;
    down) docker rm -f k1-teleop-sim k1-teleop-xr k1-teleop-yolo 2>/dev/null || true ;;
    status) docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -E '^k1-teleop-' || true ;;
    *) echo "usage: $0 [build|all|sim|teleop|yolo|down|status]" >&2; exit 2 ;;
esac
