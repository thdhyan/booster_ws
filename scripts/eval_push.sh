#!/bin/bash
# P6 metrics eval (inside the Isaac image).
#   usage: eval_push.sh <task> <checkpoint.pt | -> [num_envs] [steps] [extra args...]
#   e.g.   eval_push.sh Isaac-Push-Reach-K1-v0 logs/rsl_rl/p6_push_reach/<run>/model_1499.pt
#          eval_push.sh Isaac-Push-Reach-K1-v0 - 8 400 --zero_actions
#          eval_push.sh Isaac-Push-Reach-K1-v0 model_600.pt 8 400 --zero_vel
# Greps: EVAL_PUSH_RESULT=OK (rc of python.sh is unreliable).
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1
$PY -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
TASK="${1:?usage: eval_push.sh <task> <checkpoint|-> [envs] [steps] [extra...]}"
CKPT="${2:?missing checkpoint or -}"
ENVS="${3:-8}"
STEPS="${4:-400}"
shift 4 2>/dev/null
echo "=== PUSH EVAL $TASK ckpt=$CKPT envs=$ENVS steps=$STEPS extra=$*"
if [ "$CKPT" = "-" ]; then
  $PY scripts/eval_push_check.py --task "$TASK" --num_envs "$ENVS" --steps "$STEPS" "$@"
else
  $PY scripts/eval_push_check.py --task "$TASK" --checkpoint "$CKPT" --num_envs "$ENVS" --steps "$STEPS" "$@"
fi
echo "EVAL_PUSH_DONE rc=$?"
