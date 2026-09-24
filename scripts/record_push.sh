#!/bin/bash
# P6 play + RECORD (inside the Isaac image): HUD video of a push/reach ckpt.
#   usage: record_push.sh <task> <checkpoint.pt> <out.mp4> [label]
# Output: $3 written under the repo; grep RECORD_PUSH_DONE.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1
$PY -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
TASK="${1:?usage: record_push.sh <task> <checkpoint> <out.mp4> [label]}"
CKPT="${2:?missing checkpoint path}"
OUT="${3:?missing output mp4 path}"
LABEL="${4:-P6 push}"
echo "=== PUSH RECORD $TASK ckpt=$CKPT -> $OUT"
"$PY" isaac_tasks/k1_velocity/scripts/play_record.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num_envs 4 --steps 750 --headless \
  --eye 3.2,-2.8,1.9 --lookat 0.7,0.2,0.55 \
  --video_out "$OUT" --label "$LABEL"
RC=$?
# artifact gate: python.sh's rc is unreliable — the mp4 must exist and be fresh
if [ -s "$OUT" ] && [ "$(find "$OUT" -mmin -5 | wc -l)" -gt 0 ]; then
  echo "RECORD_PUSH_OK size=$(stat -c%s "$OUT")"
else
  echo "RECORD_PUSH_FAIL no fresh mp4"
  RC=1
fi
echo "RECORD_PUSH_DONE rc=$RC"
