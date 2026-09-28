#!/bin/bash
# Smoke-test the tiled debug panel recorder for any Track A task.
#   PANEL_TASK=Isaac-HeadTrack-K1-v0 PANEL_CKPT=<path> PANEL_NAME=p3 \
#   PANEL_STEPS=30 docker run ... /panel.sh
# Writes $PANEL_OUT_DIR/<name>_panels.mp4 plus a trace, and prints markers.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1

TASK="${PANEL_TASK:?PANEL_TASK required}"
CKPT="${PANEL_CKPT:?PANEL_CKPT required}"
NAME="${PANEL_NAME:-panel}"
STEPS="${PANEL_STEPS:-30}"
NUM_ENVS="${PANEL_NUM_ENVS:-1}"
OUT_DIR="${PANEL_OUT_DIR:-$REPO/isaac_tasks/k1_velocity/videos}"
mkdir -p "$OUT_DIR"

"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train >/tmp/panel_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity >>/tmp/panel_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets >>/tmp/panel_install.log 2>&1
"$PY" -m pip install -q ultralytics >>/tmp/panel_install.log 2>&1

timeout 1200 "$PY" -u isaac_tasks/k1_velocity/scripts/play_record.py \
  --task "$TASK" --checkpoint "$CKPT" --num_envs "$NUM_ENVS" --steps "$STEPS" \
  --headless --panel_video \
  --video_out "$OUT_DIR/${NAME}_panels.mp4" \
  --trace_out "$OUT_DIR/${NAME}_panels_trace.npz" \
  --label "$NAME" 2>&1 | tee "/tmp/${NAME}_panel.log"
rc=${PIPESTATUS[0]}
echo "PANEL_SMOKE_RC=$rc"
if [ "$rc" -eq 0 ] && [ -s "$OUT_DIR/${NAME}_panels.mp4" ]; then
  echo "PANEL_SMOKE_MARKER=OK"
  exit 0
fi
echo "PANEL_SMOKE_MARKER=FAIL"
exit 1
