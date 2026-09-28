#!/bin/bash
# Recover P3's clean iteration-1900 checkpoint to the required iteration 1999.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo P3_RECOVERY_CD_FAIL; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
export YOLO_WEIGHTS="$REPO/logs/yolov8n.pt"
[ -s "$YOLO_WEIGHTS" ] || { echo P3_RECOVERY_YOLO_MISSING; exit 2; }

"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train >/tmp/p3_recovery_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity >>/tmp/p3_recovery_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets >>/tmp/p3_recovery_install.log 2>&1
"$PY" -m pip install -q ultralytics >>/tmp/p3_recovery_install.log 2>&1

LOG="$REPO/scripts/p3.recover.full.train.log"
CKPT="$REPO/logs/rsl_rl/p3_head_track/2026-09-24_14-58-47_p3_head_track/model_1900_logstd.pt"
timeout 14400 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task Isaac-HeadTrack-K1-v0 --num_envs 512 --max_iterations 100 --seed 42 \
  --viz none --enable_cameras --safe_resume --checkpoint "$CKPT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "P3_RECOVERY_RC=$rc"
final_ckpt=$(find "$REPO/logs/rsl_rl/p3_head_track" -type f -name model_1999.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ "$rc" -eq 0 ] && grep -q "Learning iteration 1999/2000" "$LOG" && [ -s "$final_ckpt" ]; then
  echo "P3_RECOVERY_CKPT=$final_ckpt"
  echo "P3_RECOVERY_MARKER=OK"
  exit 0
fi
echo "P3_RECOVERY_MARKER=FAIL"
exit 3
