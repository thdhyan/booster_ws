#!/bin/bash
# Track B A/B hunt step C (dl, GPU 2): ground-friction parity re-gate.
# Evidence chain:
#   1. chain2 gate DIAG_FAILED_BASE_STILL_FALLS (pass-B done_rate=0.0425 >= 0.02).
#   2. Home @ cmd=0 STANDS 300/300 (push_ab_home_play_cmd0.mp4, same weights
#      model_3799/export) -> defect is env-side, matched cmd.
#   3. Diff: P6 ground was GroundPlaneCfg() default friction 0.5/0.5 vs the
#      velocity terrain plane the base trained on (1.0/1.0, multiply).
#      push_env_cfg.py now authors the trained material.
# This re-runs the two-pass frozen diag on the SAME export; gate: pass-B
# done_rate < 0.02 (eval_push.sh prints done_rate=...). GPU 2 = pinned
# record/eval GPU; training on GPU 1 is finished, never share a GPU.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/dl_ab_push_frict.log"
DOCKER="docker run --rm --gpus device=2 --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/k1_isaac_cache_rec:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"
tmux new-session -d -s k1_dl_ab_push_frict \
  "timeout 3600 $DOCKER /workspace/booster_ws/scripts/diag_frozen.sh > $LOG 2>&1; echo DIAG_RC=\$? >> $LOG"
sleep 2
tmux ls | grep k1_dl_ab_push_frict || { echo "TMUX_SESSION_MISSING"; exit 1; }
echo "FRICTION A/B LOG: $LOG"
