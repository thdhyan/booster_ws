#!/bin/bash
# DIAG (inside the Isaac image): play a partial-control checkpoint in its OWN
# env (Isaac-Velocity-PartialCtrl-K1-Play-v0) to verify the frozen base walks.
#   usage: play_partial_diag.sh <checkpoint.pt> <out.mp4>
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1
$PY -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
CKPT="${1:?usage: play_partial_diag.sh <ckpt> <out.mp4>}"
OUT="${2:?missing out.mp4}"
echo "=== PARTIAL DIAG ckpt=$CKPT"
"$PY" isaac_tasks/k1_velocity/scripts/play_record.py \
  --task Isaac-Velocity-PartialCtrl-K1-Play-v0 \
  --checkpoint "$CKPT" \
  --num_envs 4 --steps 300 --headless \
  --eye 6,-6,3 --lookat 3,0,0.5 \
  --video_out "$OUT" --label "partial diag"
if [ -s "$OUT" ] && [ "$(find "$OUT" -mmin -5 | wc -l)" -gt 0 ]; then
  echo "PARTIAL_DIAG_OK size=$(stat -c%s "$OUT")"
else
  echo "PARTIAL_DIAG_FAIL"
fi
echo "PARTIAL_DIAG_DONE"
