#!/bin/bash
# Track B gait-v2 base RESUME (container, runs INSIDE the isaac-lab image on
# dl): smoke 16x3 (marker-gated) then FULL 512x3000 resuming from spark04's
# model_800.pt — the run died with spark04 (offline since 2026-09-24 ~00:32
# UTC) at ~iter 840. Resume counts from the ckpt iteration (--max_iterations
# is absolute). Gates: PUSH_BASE_RESUME_FULL_MARKER=OK + model_2999.pt.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo "PUSH_BASE_CD_FAIL"; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_KEY_LOADED"; else echo "WANDB_KEY_MISSING"; fi

CKPT="$REPO/logs/spark04/rsl_rl/k1_partialctrl_base/2026-09-24_18-44-39_k1_partialctrl_base/model_800.pt"
[ -s "$CKPT" ] || { echo "PUSH_BASE_RESUME_GATE=NO_CKPT"; exit 9; }
TASK=Isaac-Velocity-PartialCtrl-K1-v0

echo "=== PUSH BASE INSTALL booster_train (rsl_rl fork)"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -3
echo "=== PUSH BASE INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -3
echo "=== PUSH BASE INSTALL booster_assets (K1 URDF package)"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -3

SMOKE_LOG="$REPO/scripts/push_base.resume.smoke.train.log"
FULL_LOG="$REPO/scripts/push_base.resume.full.train.log"

echo "=== PUSH BASE RESUME SMOKE $TASK 16x3"
timeout 3600 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 16 --max_iterations 3 --seed 42 \
  --headless --video --video_length 500 2>&1 | tee "$SMOKE_LOG"
RC=${PIPESTATUS[0]}
echo "PUSH_BASE_RESUME_SMOKE_RC=$RC"
if [ "$RC" -ne 0 ] || ! grep -q "Learning iteration 2/3" "$SMOKE_LOG"; then
  echo "PUSH_BASE_RESUME_GATE=SMOKE_FAILED_NO_FULL_RUN"
  exit 10
fi

echo "=== PUSH BASE RESUME FULL $TASK 512x3000 from $(basename "$CKPT") (timeout 43200s)"
timeout 43200 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 512 --max_iterations 3000 --seed 42 \
  --checkpoint "$CKPT" \
  --headless --video --video_length 1500 --video_interval 4800 2>&1 | tee "$FULL_LOG"
RC=${PIPESTATUS[0]}
echo "PUSH_BASE_RESUME_FULL_RC=$RC"
CKPT2999=$(find "$REPO/logs/rsl_rl/k1_partialctrl_base" -type f -name model_2999.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ "$RC" -eq 0 ] && grep -q "Learning iteration 2999/3000" "$FULL_LOG" && [ -s "$CKPT2999" ]; then
  echo "PUSH_BASE_CKPT=$CKPT2999"
  echo "PUSH_BASE_RESUME_FULL_MARKER=OK"
  exit 0
fi
echo "PUSH_BASE_RESUME_FULL_MARKER=FAIL"
exit 11
