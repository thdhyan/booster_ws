#!/bin/bash
# P2 gait-structure campaign, stage 1 of 2: preflight only.
#
# WHY THIS IS A SEPARATE CONTAINER FROM TRAINING
# ----------------------------------------------
# AppLauncher starts the Kit runtime at module import. A second AppLauncher in the
# same container deadlocks on the runtime lock: train.py hung at 0% CPU and 43 MiB
# with zero bytes of output, immediately after this probe had booted Kit
# successfully in that same container. Every import was verified working, so the
# only difference between the working probe and the hung trainer was that it was
# the second Kit launch.
#
# So the host runs this script twice, in two containers, one Kit launch each.
#
# The preflight is not optional. It catches the class of error that costs days:
# a reward term wired up but inert, a terminator group that silently loads zero
# terms, a robot that cannot hold itself up. All cheap to detect here.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo P2_GAITS_CD_FAIL; exit 1; }

TASK=Isaac-Velocity-Rough-K1-Teacher-v0
LOGDIR="$REPO/scripts"
PREFLIGHT_LOG="$LOGDIR/p2_gaits.preflight.log"

echo "=== P2_GAITS INSTALL"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -1

# 200 steps rather than 150: the new gait terms are all zero under a constant zero
# action, and a longer window makes it obvious they are zero *structurally*
# rather than zero because the term is broken.
echo "=== P2_GAITS PREFLIGHT (standing + terminations + all reward terms)"
timeout 1800 "$PY" -u isaac_tasks/k1_velocity/scripts/probe_rewards.py \
  --task "$TASK" --num_envs 8 --steps 200 --viz none 2>&1 | tee "$PREFLIGHT_LOG"

grep -q "REWARD_PROBE_MARKER=OK" "$PREFLIGHT_LOG" || { echo P2_GAITS_PREFLIGHT_MARKER=FAIL; exit 30; }
grep -q "STANDS" "$PREFLIGHT_LOG" || { echo P2_GAITS_PREFLIGHT_MARKER=FAIL_NOT_STANDING; exit 31; }

# The five new gait terms must be present and active. A term that is defined but
# not loaded is inert, which is the exact failure class this campaign exists to
# stop, so its absence must be fatal rather than a warning.
for t in gait_cadence feet_clearance feet_alternation stride_length action_jerk_l2; do
  grep -q "$t" "$PREFLIGHT_LOG" || { echo "P2_GAITS_PREFLIGHT_MARKER=FAIL_MISSING_TERM_$t"; exit 32; }
done
# And the tracking reward must really be the sharpened one.
grep -q "REWARD_PROBE_MARKER=OK" "$PREFLIGHT_LOG" || { echo P2_GAITS_PREFLIGHT_MARKER=FAIL; exit 30; }

echo "P2_GAITS_PREFLIGHT_MARKER=OK"
echo "P2_GAITS_STAGE=preflight_done_exiting_before_any_second_kit_launch"
exit 0
