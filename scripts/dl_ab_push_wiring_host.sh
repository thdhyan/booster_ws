#!/bin/bash
# Track B A/B hunt step E (dl, GPU 2): joint-wiring FIX re-gate.
# Evidence chain (the full hunt):
#   1. chain2 gate DIAG_FAILED_BASE_STILL_FALLS (pass-B done_rate 0.0425).
#   2. Home @ cmd=0 STANDS 300/300 with the SAME weights/first output
#      (push_ab_home_play_cmd0.mp4 vs [frozenout] step=0) -> env-side diff.
#   3. Friction parity (ground 1.0/1.0 multiply): 0.0434 -> REFUTED.
#   4. Stiff/dumb arm runs: invalid stiff 0.0294, full home-reset mimic
#      (legs scaled + arms randomized + pinned PD targets): 0.0431 -> REFUTED.
#   5. ROOT CAUSE: FrozenBaseVelocityAction used find_joints(preserve_order
#      =True) = left-then-right K1_*_JOINTS list order, but the training env
#      resolved BOTH action terms (JointActionCfg.preserve_order=False) and
#      obs terms (SceneEntityCfg.preserve_order=False) in ARTICULATION order
#      (interleaved L/R: legs [3,4,8,9,...], arms [1,2,6,7,...]) -> 11/12
#      leg dims wired to the wrong joints in P6, arm/leg obs blocks permuted.
# push_mdp.py now resolves preserve_order=False everywhere (arms: one find
# over all 8). This re-runs the two-pass diag on the DEFAULT config (wrist
# IK terms present, no diag env vars) = exactly what chain2's gate runs.
# Gate: pass-B done_rate < 0.02. GPU 2 = pinned record/eval GPU.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/dl_ab_push_wiring.log"
DOCKER="docker run --rm --gpus device=2 --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/k1_isaac_cache_rec:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"
tmux new-session -d -s k1_dl_ab_push_wiring \
  "timeout 3600 $DOCKER /workspace/booster_ws/scripts/diag_frozen.sh > $LOG 2>&1; echo DIAG_RC=\$? >> $LOG"
sleep 2
tmux ls | grep k1_dl_ab_push_wiring || { echo "TMUX_SESSION_MISSING"; exit 1; }
echo "WIRING-FIX A/B LOG: $LOG"
