#!/bin/bash
# Box-push SMOKE (16x3, video) inside the Isaac image. Usage: smoke_push.sh reach|push
set +e
WHICH="${1:-push}"
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] || export WANDB_MODE=disabled
$PY -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
case "$WHICH" in
reach) TASK=Isaac-Push-Reach-K1-v0 ;;
push)  TASK=Isaac-Push-K1-v0 ;;
*) echo "usage: smoke_push.sh reach|push"; exit 2 ;;
esac
echo "=== PUSH SMOKE $TASK 16x3"
timeout 2400 $PY isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 16 --max_iterations 3 --seed 42 \
  --headless --video --video_length 300 --video_interval 48
echo "PUSH_SMOKE_RC=$?"
echo "PUSH_SMOKE_DONE"
