#!/bin/bash
# Reward-term probe wrapper (inside the Isaac image): builds each task, runs
# ~120 random-action steps, prints per-term raw reward means + foot-contact
# sensor diagnostics. Gates on REWARD_PROBE_RESULT=OK (never on rc).
# Usage: reward_probe.sh [task ...]   (default: velocity rough + partial ctrl)
set +e
REPO=/workspace/booster_ws
cd "$REPO" || exit 1
PY=/isaac-sim/python.sh
"$PY" -m pip install --no-deps --no-build-isolation \
  -e isaac_tasks/booster_train_ref/source/booster_train \
  -e isaac_tasks/k1_velocity \
  -e src/k1_description/assets >/dev/null 2>&1

TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(Isaac-Velocity-Rough-K1-v0 Isaac-Velocity-PartialCtrl-K1-v0)
FAIL=0
for T in "${TASKS[@]}"; do
  echo "=== REWARD PROBE $T"
  LOG=/tmp/reward_probe_$(echo "$T" | tr '/' '_').log
  timeout 1800 "$PY" -u scripts/reward_probe_check.py --task "$T" --num_envs 16 --steps 120 2>&1 | tee "$LOG"
  if grep -q "REWARD_PROBE_RESULT=OK" "$LOG"; then
    echo "REWARD_PROBE=OK TASK=$T"
  else
    echo "REWARD_PROBE=FAIL TASK=$T"
    FAIL=1
  fi
done
echo "REWARD_PROBE_ALL_DONE FAIL=$FAIL"
exit $FAIL
