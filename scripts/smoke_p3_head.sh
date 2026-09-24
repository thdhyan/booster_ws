#!/bin/bash
# P3 head-track SMOKE ONLY (16x3, cameras + YOLO) — used to validate the env
# before committing a full 512-env run. Run on spark02 inside the Isaac image.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] || export WANDB_MODE=disabled
export YOLO_WEIGHTS="$REPO/logs/yolov8n.pt"
$PY -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
$PY -m pip install --no-input ultralytics 2>&1 | tail -1
[ -f "$YOLO_WEIGHTS" ] || curl -fsSL -o "$YOLO_WEIGHTS" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt || echo YOLO_DL_FAIL
echo "=== P3 SMOKE Isaac-HeadTrack-K1-v0 16x3 (cameras + YOLO)"
LOG="$REPO/scripts/p3smoke.train.log"
timeout 2400 $PY -u isaac_tasks/k1_velocity/scripts/train.py \
  --task Isaac-HeadTrack-K1-v0 --num_envs 16 --max_iterations 3 --seed 42 \
  --viz none --video --video_length 64 --video_interval 32 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "P3_SMOKE_RC=$RC"
grep -q "Learning iteration 2/3" "$LOG" && echo P3_SMOKE_MARKER=OK || echo P3_SMOKE_MARKER=FAIL
echo "P3_SMOKE_DONE"
[ "$RC" -eq 0 ] && grep -q "Learning iteration 2/3" "$LOG"
