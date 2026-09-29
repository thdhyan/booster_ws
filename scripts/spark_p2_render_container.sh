#!/bin/bash
# Render the delivery video for the verified P2 checkpoint.
#
# Uses model_5998.pt, the checkpoint whose displacement was actually measured
# rather than inferred from reward: 2.664 m net at 0.5 m/s and 3.246 m at
# 0.6 m/s, with zero falls in 750 steps at both speeds. The earlier
# model_2999.pt video is kept for comparison but is not the deliverable.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo RENDER_CD_FAIL; exit 1; }

TASK=Isaac-Velocity-Rough-K1-Teacher-v0
CKPT="$1"
SPEED="${2:-0.5}"
# Label comes from the caller. It was previously hardcoded to describe one
# specific checkpoint, so re-rendering a different policy produced a video whose
# HUD confidently stated the wrong checkpoint and the wrong measurements. The
# label is the only thing a viewer uses to judge what they are looking at, so it
# must never be stale.
LABEL="${3:-K1 P2 velocity policy | cmd vx=+${SPEED} | ${CKPT##*/}}"
[ -s "$CKPT" ] || { echo "RENDER_CKPT_MISSING=$CKPT"; exit 1; }
echo "RENDER_CKPT=$CKPT speed=$SPEED"
echo "RENDER_LABEL=$LABEL"

"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -1

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$REPO/isaac_tasks/k1_velocity/videos/p2_vel_verified_${STAMP}_panels.mp4"
TRACE="$REPO/isaac_tasks/k1_velocity/videos/p2_vel_verified_${STAMP}.npz"

# 4 labelled panels + status header. cmd held at the measured-good speed so the
# HUD shows a speed the policy actually achieves rather than one it under-tracks.
timeout 2400 "$PY" -u isaac_tasks/k1_velocity/scripts/play_record.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num_envs 8 --steps 750 --cmd "$SPEED" 0.0 0.0 \
  --panel_video --video_out "$OUT" --trace_out "$TRACE" \
  --label "$LABEL" \
  2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "RENDER_RC=$rc"
if [ -s "$OUT" ]; then
  echo "RENDER_VIDEO=OK $OUT"
  ls -lh "$OUT" | awk '{print "  size:", $5}'
else
  echo "RENDER_VIDEO=FAIL"
  exit 2
fi
[ -s "$TRACE" ] && echo "RENDER_TRACE=OK $TRACE"
echo "RENDER_ALL_DONE"
