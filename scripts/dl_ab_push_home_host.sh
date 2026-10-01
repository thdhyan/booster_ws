#!/bin/bash
# Host launcher (dl, GPU 2): Track B A/B hunt step A (HANDOFF "in-env A/B hunt",
# prescribed after chain2 fired DIAG_FAILED_BASE_STILL_FALLS 2026-09-27).
# Plays the FINAL gait-v2 base checkpoint model_3799.pt (resume run dir
# 2026-09-28_01-12-26) in its HOME env — play_record --checkpoint needs a
# training state dict; the TorchScript export fails runner.load() with
# NotImplementedError (export == actor bit-exact per push_export_parity.py,
# so weights are equivalent). HOME env, 4 envs, 300 steps, debug video.
# RUN 2 (cmd-matched): pass 1 ran WITHOUT --cmd, so the play cfg's pinned
# lin_vel_x=(1.0,1.0) applied while HUD wrongly labelled it "none (stand
# task)" (play_record.py:266 default label) -> that only proved walking at
# cmd=1.0. P6 pass-B was cmd=0, so this run pins --cmd 0 0 0 to match:
#   stands there -> env-side diff P6 push vs partial home (friction/spawn/arms);
#   falls there  -> base cannot stand at cmd=0 anywhere (25% rel_standing_envs
#                   trained) -> policy-level stand failure, not an env bug.
# GPU discipline: GPU 2 = pinned record/eval GPU, never concurrent with
# training on the same GPU (training was pinned to GPU 1).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/dl_ab_push_home.log"
OUT="$REPO/isaac_tasks/k1_velocity/videos/push_ab_home_play_cmd0.mp4"
CKPT="$REPO/logs/rsl_rl/k1_partialctrl_base/2026-09-28_01-12-26_k1_partialctrl_base/model_3799.pt"
[ -s "$CKPT" ] || { echo "NO_CKPT $CKPT"; exit 1; }
mkdir -p "$HERE/k1_isaac_cache_rec" "$(dirname "$OUT")"
tmux new-session -d -s k1_dl_ab_push_home "docker run --rm --gpus device=2 --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/k1_isaac_cache_rec:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /workspace/booster_ws/scripts/play_partial_diag.sh /workspace/booster_ws/logs/rsl_rl/k1_partialctrl_base/2026-09-28_01-12-26_k1_partialctrl_base/model_3799.pt /workspace/booster_ws/isaac_tasks/k1_velocity/videos/push_ab_home_play_cmd0.mp4 --cmd 0 0 0 > $LOG 2>&1; echo DOCKER_RC=\$? >> $LOG"
sleep 2
tmux ls | grep k1_dl_ab_push_home || { echo "TMUX_SESSION_MISSING"; exit 1; }
echo "A/B HOME PLAY LOG: $LOG"
