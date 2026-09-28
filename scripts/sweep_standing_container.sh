#!/bin/bash
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train >/tmp/sweep_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity >>/tmp/sweep_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets >>/tmp/sweep_install.log 2>&1
timeout 1800 "$PY" -u isaac_tasks/k1_velocity/scripts/sweep_standing_gains.py \
  --num_envs "$SWEEP_ENVS" --steps "$SWEEP_STEPS" --mults "$SWEEP_MULTS" --viz none
exit $?
