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
LOG="$REPO/scripts/p4tsmoke.train.log"
timeout 2400 $PY -u isaac_tasks/k1_velocity/scripts/train.py \
  --task Isaac-Kick-Ball-K1-Teacher-v0 --num_envs 16 --max_iterations 3 --seed 42 \
  --viz none --video --video_length 48 --video_interval 24 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "P4T_SMOKE_RC=$RC"
grep -q "Learning iteration 2/3" "$LOG" && echo P4T_SMOKE_MARKER=OK || echo P4T_SMOKE_MARKER=FAIL
echo "P4T_SMOKE_DONE"
[ "$RC" -eq 0 ] && grep -q "Learning iteration 2/3" "$LOG"
