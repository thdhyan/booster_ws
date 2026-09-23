#!/bin/bash
# FULL (container-side): full P1/P2 student runs on spark04 — the exact queue_student.sh
# full recipe (mirrors dl byte-for-byte). S1/S2 smokes green on this image, no smoke gate.
# Runs INSIDE nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 via:
#   --entrypoint bash -e WHICH=p1|p2 IMG /full.sh
# Output streams straight to the session log (live tail-able; wandb = dashboard).
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo "FULL_CD_FAIL"; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_KEY_LOADED"; else echo "WANDB_KEY_MISSING"; fi

echo "=== FULL INSTALL booster_train (rsl_rl fork)"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -3
echo "=== FULL INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -3
echo "=== FULL INSTALL booster_assets (K1 URDF package)"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -3

case "${WHICH:?set WHICH=p1|p2}" in
  p1) TASK=Isaac-Basic-Student-K1-v0
      TEACHER="$REPO/logs/rsl_rl/p1_basic_teacher/2026-09-22_13-41-33_p1_basic_teacher/model_6498.pt"
      ENVS=256; ITERS=1500; TMO=14400 ;;
  p2) TASK=Isaac-Velocity-Distill-K1-v0
      TEACHER="$REPO/logs/rsl_rl/k1_velocity_teacher/2026-09-22_13-14-06/model_2999.pt"
      ENVS=512; ITERS=3000; TMO=28800 ;;
  *) echo "FULL_BAD_WHICH=$WHICH"; exit 2 ;;
esac
if [ -f "$TEACHER" ]; then echo "TEACHER_OK $TEACHER"; else echo "TEACHER_MISSING $TEACHER"; exit 3; fi

echo "=== FULL RUN $TASK ${ENVS}x$ITERS (timeout ${TMO}s)"
timeout "$TMO" "$PY" isaac_tasks/k1_velocity/scripts/train_student.py \
  --task "$TASK" --teacher_checkpoint "$TEACHER" \
  --num_envs "$ENVS" --max_iterations "$ITERS" --seed 42 \
  --headless --enable_cameras --video --video_length 1500 --video_interval 4800
rc=$?
echo "FULL_RC=$rc WHICH=$WHICH"
echo "FULL_CONTAINER_DONE WHICH=$WHICH"
