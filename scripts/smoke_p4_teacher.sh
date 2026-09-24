#!/bin/bash
# P4 kick TEACHER SMOKE ONLY (16x3). Run on a spark inside the Isaac image.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] || export WANDB_MODE=disabled
$PY -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
echo "=== P4T SMOKE Isaac-Kick-Ball-K1-Teacher-v0 16x3"
timeout 2400 $PY isaac_tasks/k1_velocity/scripts/train.py \
  --task Isaac-Kick-Ball-K1-Teacher-v0 --num_envs 16 --max_iterations 3 --seed 42 \
  --headless --video --video_length 300 --video_interval 48
echo "P4T_SMOKE_RC=$?"
echo "P4T_SMOKE_DONE"
