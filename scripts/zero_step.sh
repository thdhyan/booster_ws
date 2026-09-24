#!/bin/bash
# Zero-agent single-step verification wrapper (inside the Isaac image).
# Usage: zero_step.sh <task> [cameras]
set +e
REPO=/workspace/booster_ws
cd "$REPO" || exit 1
TASK="${1:?usage: zero_step.sh <task> [cameras]}"
EXTRA=""
[ "${2:-}" = "cameras" ] && EXTRA="--cameras"
/isaac-sim/python.sh -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
timeout 1800 /isaac-sim/python.sh scripts/zero_step_check.py --task "$TASK" --num_envs 4 --steps 3 $EXTRA
grep -q "ZERO_STEP_RESULT=OK" && echo ZERO_STEP_DONE=OK || echo ZERO_STEP_DONE=FAIL
