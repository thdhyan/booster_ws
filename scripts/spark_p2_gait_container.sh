#!/bin/bash
# P2 gait-v2 campaign: revised command distribution + H1/G1-style shaping.
# Runs smoke -> full teacher -> smoke -> full student.  Every stage must emit
# both its process marker and checkpoint before the next stage starts.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo P2_GAIT_CD_FAIL; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] || { echo P2_GAIT_WANDB_KEY_MISSING; exit 2; }

echo "=== P2_GAIT INSTALL booster_train"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -3
echo "=== P2_GAIT INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -3
echo "=== P2_GAIT INSTALL k1_assets"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -3

TEACHER_TASK=Isaac-Velocity-Rough-K1-Teacher-v0
STUDENT_TASK=Isaac-Velocity-Distill-K1-v0

SMOKE_TEACHER_LOG="$REPO/scripts/p2_gait.teacher.smoke.train.log"
SMOKE_STUDENT_LOG="$REPO/scripts/p2_gait.student.smoke.train.log"
FULL_TEACHER_LOG="$REPO/scripts/p2_gait.teacher.full.train.log"
FULL_STUDENT_LOG="$REPO/scripts/p2_gait.student.full.train.log"

run_smoke_teacher() {
  echo "=== P2_GAIT SMOKE TEACHER 16x3"
  timeout 1800 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
    --task "$TEACHER_TASK" --num_envs 16 --max_iterations 3 --seed 42 --viz none \
    2>&1 | tee "$SMOKE_TEACHER_LOG"
  local rc=${PIPESTATUS[0]}
  echo "P2_GAIT_TEACHER_SMOKE_RC=$rc"
  [ "$rc" -eq 0 ] && grep -q "Learning iteration 2/3" "$SMOKE_TEACHER_LOG" && return 0
  return 1
}

run_full_teacher() {
  echo "=== P2_GAIT FULL TEACHER 512x3000"
  timeout 43200 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
    --task "$TEACHER_TASK" --num_envs 512 --max_iterations 3000 --seed 42 --viz none \
    2>&1 | tee "$FULL_TEACHER_LOG"
  local rc=${PIPESTATUS[0]}
  echo "P2_GAIT_TEACHER_FULL_RC=$rc"
  local ckpt
  ckpt=$(find "$REPO/logs/rsl_rl/p2_move_teacher" -type f -name model_2999.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  if [ "$rc" -eq 0 ] && grep -q "Learning iteration 2999/3000" "$FULL_TEACHER_LOG" && [ -s "$ckpt" ]; then
    echo "P2_GAIT_TEACHER_CKPT=$ckpt"
    echo "P2_GAIT_TEACHER_MARKER=OK"
    return 0
  fi
  echo "P2_GAIT_TEACHER_MARKER=FAIL"
  return 1
}

run_smoke_student() {
  local teacher="$1"
  echo "=== P2_GAIT SMOKE STUDENT 16x3 teacher=$teacher"
  timeout 1800 "$PY" -u isaac_tasks/k1_velocity/scripts/train_student.py \
    --task "$STUDENT_TASK" --teacher_checkpoint "$teacher" \
    --num_envs 16 --max_iterations 3 --seed 42 --viz none \
    2>&1 | tee "$SMOKE_STUDENT_LOG"
  local rc=${PIPESTATUS[0]}
  echo "P2_GAIT_STUDENT_SMOKE_RC=$rc"
  [ "$rc" -eq 0 ] && grep -q "Learning iteration 2/3" "$SMOKE_STUDENT_LOG" && return 0
  return 1
}

run_full_student() {
  local teacher="$1"
  echo "=== P2_GAIT FULL STUDENT 512x3000 teacher=$teacher"
  timeout 43200 "$PY" -u isaac_tasks/k1_velocity/scripts/train_student.py \
    --task "$STUDENT_TASK" --teacher_checkpoint "$teacher" \
    --num_envs 512 --max_iterations 3000 --seed 42 --viz none \
    2>&1 | tee "$FULL_STUDENT_LOG"
  local rc=${PIPESTATUS[0]}
  echo "P2_GAIT_STUDENT_FULL_RC=$rc"
  local ckpt
  ckpt=$(find "$REPO/logs/rsl_rl/p2_move_student" -type f -name model_2999.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  if [ "$rc" -eq 0 ] && grep -q "Learning iteration 2999/3000" "$FULL_STUDENT_LOG" && [ -s "$ckpt" ]; then
    echo "P2_GAIT_STUDENT_CKPT=$ckpt"
    echo "P2_GAIT_STUDENT_MARKER=OK"
    return 0
  fi
  echo "P2_GAIT_STUDENT_MARKER=FAIL"
  return 1
}

if ! run_smoke_teacher; then
  echo P2_GAIT_GATE=TEACHER_SMOKE_FAILED_NO_FULL_RUN
  exit 10
fi
if ! run_full_teacher; then
  echo P2_GAIT_GATE=TEACHER_FULL_FAILED_NO_STUDENT
  exit 11
fi
TEACHER=$(find "$REPO/logs/rsl_rl/p2_move_teacher" -type f -name model_2999.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ -z "$TEACHER" ] || [ ! -s "$TEACHER" ]; then
  echo P2_GAIT_GATE=TEACHER_CHECKPOINT_MISSING
  exit 12
fi
if ! run_smoke_student "$TEACHER"; then
  echo P2_GAIT_GATE=STUDENT_SMOKE_FAILED_NO_FULL_RUN
  exit 13
fi
if ! run_full_student "$TEACHER"; then
  echo P2_GAIT_GATE=STUDENT_FULL_FAILED
  exit 14
fi
echo P2_GAIT_ALL_DONE
exit 0
