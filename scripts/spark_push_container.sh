#!/bin/bash
# Push container launcher: WHICH=reach (stage 1) | push (stage 2, warm-start).
# 1. install editable pkgs,
# 2. SMOKE 16 envs x 3 iters (gated),
# 3. FULL run only on smoke green (grep marker, never rc alone).
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo "PUSH_CD_FAIL"; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_KEY_LOADED"; else echo "WANDB_KEY_MISSING"; fi

echo "=== PUSH INSTALL booster_train (rsl_rl fork)"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -2
echo "=== PUSH INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -2
echo "=== PUSH INSTALL booster_assets (K1 URDF package)"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -2

case "${WHICH:-}" in
reach)
  TASK=Isaac-Push-Reach-K1-v0
  ENVS=256; ITERS=1500; TIMEOUT=43200
  EXTRA=""
  ;;
push)
  TASK=Isaac-Push-K1-v0
  ENVS=256; ITERS=3000; TIMEOUT=86400
  WARM=$(ls -t "$REPO"/logs/rsl_rl/p6_push_reach/*/model_*.pt 2>/dev/null | head -1)
  if [ -z "$WARM" ]; then
    echo "PUSH_GATE=NO_REACH_CHECKPOINT (run spark_push_host.sh reach first)"
    exit 11
  fi
  EXTRA="--checkpoint $WARM"
  echo "PUSH_WARM_START=$WARM"
  ;;
*)
  echo "usage: WHICH=reach|push"; exit 2 ;;
esac

echo "=== PUSH SMOKE $TASK 16x3"
SMOKE_LOG="$REPO/scripts/${WHICH}.push.smoke.train.log"
timeout 3600 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 16 --max_iterations 3 --seed 42 \
  --viz none --video --video_length 48 --video_interval 24 $EXTRA 2>&1 | tee "$SMOKE_LOG"
RC=${PIPESTATUS[0]}
echo "PUSH_SMOKE_RC=$RC"
if [ "$RC" -ne 0 ] || ! grep -q "Learning iteration 2/3" "$SMOKE_LOG"; then
  echo "PUSH_GATE=SMOKE_FAILED_NO_FULL_RUN"
  exit 10
fi

echo "=== PUSH FULL $TASK ${ENVS}x${ITERS} (timeout ${TIMEOUT}s) $EXTRA"
FULL_LOG="$REPO/scripts/${WHICH}.push.full.train.log"
timeout "$TIMEOUT" "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs "$ENVS" --max_iterations "$ITERS" --seed 42 \
  --viz none --video --video_length 1500 --video_interval 6400 $EXTRA 2>&1 | tee "$FULL_LOG"
RC=${PIPESTATUS[0]}
if grep -q "Learning iteration $((ITERS - 1))/$ITERS" "$FULL_LOG"; then
  echo "PUSH_FULL_MARKER=OK"
else
  echo "PUSH_FULL_MARKER=FAIL"
  RC=1
fi
echo "PUSH_FULL_RC=$RC"
echo "PUSH_CONTAINER_DONE WHICH=$WHICH RC=$RC"
exit $RC
