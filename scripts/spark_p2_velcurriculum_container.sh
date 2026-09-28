#!/bin/bash
# P2 velocity-curriculum campaign.
#
# Why a new campaign rather than resuming p2 gait-v2: that policy was trained
# with a fixed +/-0.5 m/s command range, so it has never been asked to go faster
# and cannot have learned to. Widening the range on a converged narrow-range
# policy mostly produces falls. Starting fresh at 0.5 and letting the curriculum
# widen is the honest version, and it costs the same wall clock as one more run.
#
# Teacher only. Distillation to a student happens after the teacher is shown to
# actually widen its range, because a student can only imitate what the teacher
# can do and distilling a narrow-range teacher wastes a second 3000-iteration run.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo P2_VEL_CD_FAIL; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] || { echo P2_VEL_WANDB_KEY_MISSING; exit 2; }

echo "=== P2_VEL INSTALL booster_train"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -2
echo "=== P2_VEL INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -2
echo "=== P2_VEL INSTALL k1_assets"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -2

TASK=Isaac-Velocity-Rough-K1-Teacher-v0
LOGDIR="$REPO/scripts"
PREFLIGHT_LOG="$LOGDIR/p2_vel.preflight.log"
TEACHER_LOG="$LOGDIR/p2_vel.teacher.train.log"

# Pre-flight: the K1 must stand under zero action and the terminations must be
# live. Both were silent failures before (trunk sank 0.589 -> 0.076 m; a
# TerminationsCfg missing @configclass loaded zero terms). Cheap to catch here,
# fatal to discover at iteration 3000.
echo "=== P2_VEL PREFLIGHT"
timeout 1200 "$PY" -u isaac_tasks/k1_velocity/scripts/probe_rewards.py \
  --task "$TASK" --num_envs 8 --steps 150 --viz none 2>&1 | tee "$PREFLIGHT_LOG"
grep -q "REWARD_PROBE_MARKER=OK" "$PREFLIGHT_LOG" || { echo P2_VEL_PREFLIGHT_MARKER=FAIL; exit 30; }
grep -q "STANDS" "$PREFLIGHT_LOG" || { echo P2_VEL_PREFLIGHT_MARKER=FAIL_NOT_STANDING; exit 31; }
echo "P2_VEL_PREFLIGHT_MARKER=OK"

echo "=== P2_VEL SMOKE 16x3"
timeout 1800 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 16 --max_iterations 3 --seed 42 --viz none 2>&1 | tee "$LOGDIR/p2_vel.smoke.log"
rc=${PIPESTATUS[0]}
echo "P2_VEL_SMOKE_RC=$rc"
[ "$rc" -eq 0 ] && grep -q "Learning iteration 2/3" "$LOGDIR/p2_vel.smoke.log" \
  || { echo P2_VEL_GATE=SMOKE_FAILED_NO_FULL_RUN; exit 10; }

echo "=== P2_VEL FULL TEACHER 512x3000 (velocity curriculum active)"
timeout 43200 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 512 --max_iterations 3000 --seed 42 --viz none \
  2>&1 | tee "$TEACHER_LOG"
rc=${PIPESTATUS[0]}
echo "P2_VEL_TEACHER_RC=$rc"

# The curriculum must have been *observed* to act, not merely present in the
# config. A CurrTerm that silently loads nothing would leave a 3000-iteration run
# training at a fixed 0.5 m/s while looking healthy -- the same class of failure
# as the missing @configclass terminations.
if grep -q "vel-curriculum" "$TEACHER_LOG"; then
  echo "P2_VEL_CURRICULUM_MARKER=OK"
  echo "--- curriculum decisions ---"
  grep "vel-curriculum" "$TEACHER_LOG" | head -20
  echo "--- final range seen ---"
  grep "vel-curriculum" "$TEACHER_LOG" | tail -1
else
  echo "P2_VEL_CURRICULUM_MARKER=FAIL_NEVER_FIRED"
  echo "P2_VEL_GATE=CURRICULUM_INERT_NO_TRUST_IN_CHECKPOINT"
  exit 40
fi

ckpt=$(find "$REPO/logs/rsl_rl" -type f -name model_2999.pt -newermt '-3 hours' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ "$rc" -eq 0 ] && grep -q "Learning iteration 2999/3000" "$TEACHER_LOG" && [ -s "$ckpt" ]; then
  echo "P2_VEL_TEACHER_CKPT=$ckpt"
  echo "P2_VEL_TEACHER_MARKER=OK"
  echo P2_VEL_ALL_DONE
  exit 0
fi
echo "P2_VEL_TEACHER_MARKER=FAIL"
exit 11
