#!/bin/bash
# Soccer container launcher: WHICH=p3 (head track) | p4t (kick teacher).
# 1. install editable pkgs (+ ultralytics + yolov8n for p3's detector),
# 2. SMOKE 16 envs x 3 iters (cameras + video for p3),
# 3. gate: FULL run only on smoke rc 0.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo "SOC_CD_FAIL"; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_KEY_LOADED"; else echo "WANDB_KEY_MISSING"; fi

echo "=== SOC INSTALL booster_train (rsl_rl fork)"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -2
echo "=== SOC INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -2
echo "=== SOC INSTALL booster_assets (K1 URDF package)"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -2

case "${WHICH:-}" in
p3)
  TASK=Isaac-HeadTrack-K1-v0
  ENVS=512; ITERS=2000
  echo "=== SOC INSTALL ultralytics (detector) + yolov8n weights"
  "$PY" -m pip install --no-input ultralytics 2>&1 | tail -2
  YOLO=/workspace/booster_ws/logs/yolov8n.pt
  [ -f "$YOLO" ] || curl -fsSL -o "$YOLO" \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt \
    || { echo "YOLO_WEIGHTS_DOWNLOAD_FAIL (geometric proxy will be used)"; }
  export YOLO_WEIGHTS="$YOLO"
  echo "=== SOC SMOKE $TASK 16x3 (cameras + YOLO)"
  SMOKE_LOG="$REPO/scripts/${WHICH}.smoke.train.log"
  timeout 3600 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
    --task "$TASK" --num_envs 16 --max_iterations 3 --seed 42 \
    --viz none --video --video_length 64 --video_interval 32 2>&1 | tee "$SMOKE_LOG"
  RC=${PIPESTATUS[0]}
  echo "SOC_SMOKE_RC=$RC"
  if [ "$RC" -ne 0 ] || ! grep -q "Learning iteration 2/3" "$SMOKE_LOG"; then
    echo "SOC_GATE=SMOKE_FAILED_NO_FULL_RUN"
    exit 10
  fi
  echo "=== SOC FULL $TASK ${ENVS}x${ITERS} (timeout 28800s)"
  FULL_LOG="$REPO/scripts/${WHICH}.full.train.log"
  timeout 28800 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
    --task "$TASK" --num_envs "$ENVS" --max_iterations "$ITERS" --seed 42 \
    --viz none --video --video_length 1500 --video_interval 6400 2>&1 | tee "$FULL_LOG"
  RC=${PIPESTATUS[0]}
  if grep -q "Learning iteration $((ITERS - 1))/$ITERS" "$FULL_LOG"; then
    echo "SOC_FULL_MARKER=OK"
  else
    echo "SOC_FULL_MARKER=FAIL"
    RC=1
  fi
  echo "SOC_FULL_RC=$RC"
  echo "SOC_CONTAINER_DONE WHICH=$WHICH RC=$RC"
  exit $RC
  ;;
p4t)
  TASK=Isaac-Kick-Ball-K1-Teacher-v0
  ENVS=512; ITERS=3000
  echo "=== SOC SMOKE $TASK 16x3"
  SMOKE_LOG="$REPO/scripts/${WHICH}.smoke.train.log"
  timeout 3600 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
    --task "$TASK" --num_envs 16 --max_iterations 3 --seed 42 \
    --viz none --video --video_length 48 --video_interval 24 2>&1 | tee "$SMOKE_LOG"
  RC=${PIPESTATUS[0]}
  echo "SOC_SMOKE_RC=$RC"
  if [ "$RC" -ne 0 ] || ! grep -q "Learning iteration 2/3" "$SMOKE_LOG"; then
    echo "SOC_GATE=SMOKE_FAILED_NO_FULL_RUN"
    exit 10
  fi
  echo "=== SOC FULL $TASK ${ENVS}x${ITERS} (timeout 43200s)"
  FULL_LOG="$REPO/scripts/${WHICH}.full.train.log"
  timeout 43200 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
    --task "$TASK" --num_envs "$ENVS" --max_iterations "$ITERS" --seed 42 \
    --viz none --video --video_length 1500 --video_interval 4800 2>&1 | tee "$FULL_LOG"
  RC=${PIPESTATUS[0]}
  if grep -q "Learning iteration $((ITERS - 1))/$ITERS" "$FULL_LOG"; then
    echo "SOC_FULL_MARKER=OK"
  else
    echo "SOC_FULL_MARKER=FAIL"
    RC=1
  fi
  echo "SOC_FULL_RC=$RC"
  echo "SOC_CONTAINER_DONE WHICH=$WHICH RC=$RC"
  exit $RC
  ;;
*)
  echo "usage: WHICH=p3|p4t"; exit 2 ;;
esac
