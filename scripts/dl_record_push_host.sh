#!/bin/bash
# P6 play+record host wrapper (dl, GPU 2 = pinned record/eval GPU).
#   usage: dl_record_push_host.sh <task> <ckpt> <out.mp4> [label] [panel]
# e.g. dl_record_push_host.sh Isaac-Push-K1-v0 logs/rsl_rl/p6_push/<run>/model_4498.pt \
#        videos/push_policy.mp4 "P6 push stage-2" panel
# Runs synchronously (~5-10 min incl. Kit boot); grep RECORD_PUSH_OK /
# RECORD_PUSH_DONE in the printed output (rc of python.sh unreliable).
# NEVER run while training occupies GPU 2 — dl pins training to GPU 1.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
TASK="${1:?usage: dl_record_push_host.sh <task> <ckpt> <out.mp4> [label]}"
shift
docker run --rm --gpus device=2 --user 0 --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$HERE/k1_isaac_cache_rec:/root/.cache" -v "$REPO:/workspace/booster_ws" \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 \
  -c '/workspace/booster_ws/scripts/record_push.sh "$@"' _ "$TASK" "$@"
