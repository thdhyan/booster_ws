#!/bin/bash
# Track B frozen-base retrain (container side, runs INSIDE
# nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 via spark_push_base_host.sh):
#   1. install our 3 editable pkgs,
#   2. SMOKE 16 envs x 3 iters of Isaac-Velocity-PartialCtrl-K1-v0,
#   3. gate on the smoke MARKER (rc alone is unreliable on python.sh),
#   4. FULL 512x3000 -> PUSH_BASE_FULL_MARKER=OK + model_2999.pt exists.
# Rationale: the shipped k1_partialctrl_base.pt was trained on the pre-gait-v2
# config AND its TorchScript export falls in the push env (falls every ~17
# steps with zero commands; hold-mode env is perfect).  Retraining the base on
# the reviewed gait-v2 config (feet_air_time +0.5 / feet_slide -0.25 /
# stand_still / undesired_contacts, direct Cartesian commands) gives Track B a
# fresh base + a fresh export to re-verify the push path end to end.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo "PUSH_BASE_CD_FAIL"; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_KEY_LOADED"; else echo "WANDB_KEY_MISSING"; fi

TASK=Isaac-Velocity-PartialCtrl-K1-v0

echo "=== PUSH BASE INSTALL booster_train (rsl_rl fork)"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -3
echo "=== PUSH BASE INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -3
echo "=== PUSH BASE INSTALL booster_assets (K1 URDF package)"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -3

SMOKE_LOG="$REPO/scripts/push_base.smoke.train.log"
FULL_LOG="$REPO/scripts/push_base.full.train.log"

echo "=== PUSH BASE SMOKE $TASK 16x3"
timeout 3600 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 16 --max_iterations 3 --seed 42 \
  --headless --video --video_length 500 2>&1 | tee "$SMOKE_LOG"
RC=${PIPESTATUS[0]}
echo "PUSH_BASE_SMOKE_RC=$RC"
if [ "$RC" -ne 0 ] || ! grep -q "Learning iteration 2/3" "$SMOKE_LOG"; then
  echo "PUSH_BASE_GATE=SMOKE_FAILED_NO_FULL_RUN"
  exit 10
fi

echo "=== PUSH BASE FULL $TASK 512x3000 (timeout 43200s)"
timeout 43200 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 512 --max_iterations 3000 --seed 42 \
  --headless --video --video_length 1500 --video_interval 4800 2>&1 | tee "$FULL_LOG"
RC=${PIPESTATUS[0]}
echo "PUSH_BASE_FULL_RC=$RC"
CKPT=$(find "$REPO/logs/rsl_rl/k1_partialctrl_base" -type f -name model_2999.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ "$RC" -eq 0 ] && grep -q "Learning iteration 2999/3000" "$FULL_LOG" && [ -s "$CKPT" ]; then
  echo "PUSH_BASE_CKPT=$CKPT"
  echo "PUSH_BASE_FULL_MARKER=OK"
  echo "NEXT: scripts/export_base_policy.sh (re-export models/k1_partialctrl_base.pt), then diag_frozen.sh, then spark_push_host.sh reach"
  exit 0
fi
echo "PUSH_BASE_FULL_MARKER=FAIL"
exit 11
