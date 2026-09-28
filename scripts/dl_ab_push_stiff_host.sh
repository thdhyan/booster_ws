#!/bin/bash
# Track B A/B hunt step D (dl, GPU 2): stiff-arms test of the frozen base.
# Evidence chain:
#   1. chain2 gate DIAG_FAILED_BASE_STILL_FALLS (pass-B done_rate=0.0425).
#   2. Home @ cmd=0 STANDS 300/300 with the same weights -> env-side diff.
#   3. Friction parity (ground 1.0/1.0 multiply, trained material): done_rate
#      0.0434 -> friction REFUTED.
#   4. Home step-1 actions ≈ P6 frozen out0 (same ±4 first command) -> the
#      policy emits the same command; the difference is physical.
#   5. Static diff: partial (home) has NO arm action term (PD targets pinned
#      by reset event = stiff hold). P6 wrist IK uses stock relative-mode
#      DifferentialIKController whose set_command REBASES ee_pos_des to the
#      current EE pose every control step -> zero wrist actions = floppy arms.
# This run passes PUSH_DIAG_STIFF_ARMS=1 into the container: cfg drops the
# wrist terms so arm PD targets persist at default (home-like stiff hold).
# Gate: pass-B done_rate < 0.02 (eval_push.sh prints done_rate=...).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/dl_ab_push_stiff.log"
DOCKER="docker run --rm --gpus device=2 --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e PUSH_DIAG_STIFF_ARMS=1 -v $HERE/k1_isaac_cache_rec:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"
tmux new-session -d -s k1_dl_ab_push_stiff \
  "timeout 3600 $DOCKER /workspace/booster_ws/scripts/diag_frozen.sh > $LOG 2>&1; echo DIAG_RC=\$? >> $LOG"
sleep 2
tmux ls | grep k1_dl_ab_push_stiff || { echo "TMUX_SESSION_MISSING"; exit 1; }
echo "STIFF-ARMS A/B LOG: $LOG"
