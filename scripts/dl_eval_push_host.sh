#!/bin/bash
# P6 metrics eval host wrapper (dl, GPU 2 = pinned record/eval GPU).
#   usage: dl_eval_push_host.sh <task> <ckpt|-> [envs] [steps] [extra...]
# e.g. dl_eval_push_host.sh Isaac-Push-Reach-K1-v0 - 8 400 --zero_actions
#      dl_eval_push_host.sh Isaac-Push-K1-v0 logs/rsl_rl/p6_push/<run>/model_4498.pt
# Runs synchronously (~2-6 min incl. Kit boot); grep EVAL_PUSH_RESULT=OK /
# EVAL_METRIC in the printed output (python.sh rc is unreliable).
# NEVER run while training occupies GPU 2 — dl pins training to GPU 1.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
TASK="${1:?usage: dl_eval_push_host.sh <task> <ckpt|-> [envs] [steps] [extra...]}"
shift
docker run --rm --gpus device=2 --user 0 --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$HERE/k1_isaac_cache_rec:/root/.cache" -v "$REPO:/workspace/booster_ws" \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 \
  -c '/workspace/booster_ws/scripts/eval_push.sh "$@"' _ "$TASK" "$@"
