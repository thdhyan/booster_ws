#!/bin/bash
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1

"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train >/tmp/probe_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity >>/tmp/probe_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets >>/tmp/probe_install.log 2>&1

timeout 1500 "$PY" -u isaac_tasks/k1_velocity/scripts/probe_rewards.py \
  --task "$PROBE_TASK" --num_envs "$PROBE_ENVS" --steps "$PROBE_STEPS" --headless
exit $?
