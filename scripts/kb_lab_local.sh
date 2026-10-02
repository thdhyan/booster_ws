#!/usr/bin/env bash
# Local keyboard-control lab for the Booster K1 teacher policy (laptop, DISPLAY=:0).
#
# Runs the 237-dim TEACHER policy, which is EVALUATION ONLY (it reads a 187-ray
# privileged height scan). A deployable policy must come from the distilled
# 50-dim student via play_keyboard_fixed.py.
#
# The script prints live velocity tracking so W/A/S/D/Q/E can be checked against
# the commanded velocity, and root height so posture is visible.
set -u
WS=/home/thakk100/Projects/booster_ws
CKPT="$WS/models/k1_teacher_6321_gains.pt"
cd "$WS" || exit 1
source scripts/phase6_env.sh
exec "$PHASE6_VENV/bin/python" isaac_tasks/k1_velocity/scripts/play_keyboard_teacher.py \
  --task Isaac-Velocity-Rough-K1-Teacher-Play-v0 \
  --checkpoint "$CKPT" \
  --num_envs 1 \
  --viz kit
