#!/bin/bash
# S2 (container-side): install our booster_ws stack into the isaac-lab image and run
# the P2 student (velocity distill) smoke (16 envs x 3 iters) exactly as queue_student.sh
# does on dl for p2. Runs INSIDE nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 via:
#   --entrypoint bash IMG /s2.sh
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo "S2_CD_FAIL"; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_KEY_LOADED"; else echo "WANDB_KEY_MISSING"; fi

echo "=== S2 INSTALL booster_train (rsl_rl fork)"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -4
echo "=== S2 INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -4
echo "=== S2 INSTALL booster_assets (K1 URDF package)"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -4
echo "=== S2 VERIFY import"
if "$PY" -c "import k1_velocity" >/tmp/s2_imp.log 2>&1; then
  echo "IMPORT_k1_velocity_OK"
else
  echo "IMPORT_FAIL"; tail -6 /tmp/s2_imp.log
fi

TEACHER="$REPO/logs/rsl_rl/k1_velocity_teacher/2026-09-22_13-14-06/model_2999.pt"
[ -f "$TEACHER" ] && echo "TEACHER_OK $TEACHER" || echo "TEACHER_MISSING $TEACHER"

echo "=== S2 SMOKE Isaac-Velocity-Distill-K1-v0 16x3"
# Redirect to a file so the captured rc is python's, not the pipeline's tail.
timeout 1800 "$PY" isaac_tasks/k1_velocity/scripts/train_student.py \
  --task Isaac-Velocity-Distill-K1-v0 --teacher_checkpoint "$TEACHER" \
  --num_envs 16 --max_iterations 3 --seed 42 \
  --headless --enable_cameras --video --video_length 500 --video_interval 4800 \
  > /tmp/s2_train.log 2>&1
rc=$?
tail -100 /tmp/s2_train.log
echo "S2_RC=$rc"
echo "S2_CONTAINER_DONE"
