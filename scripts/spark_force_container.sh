#!/bin/bash
# F (container-side): P1f/P2f force-variant trainings, SMOKE-gated and CHAINED:
#   SMOKE teacher 16x3 -> FULL teacher -> SMOKE student 16x3 (fresh teacher
#   ckpt) -> FULL student. Runs INSIDE nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1
#   via spark_force_host.sh (WHICH=p1f|p2f).
#
# Tasks (registered in tasks/{basic,velocity}/__init__.py):
#   p1f: Isaac-Basic-Teacher-K1-F-v0      -> Isaac-Basic-Student-K1-F-v0
#        exp p1f_basic_teacher            exp p1f_basic_student
#   p2f: Isaac-Velocity-Rough-K1-Teacher-F-v0 -> Isaac-Velocity-Distill-K1-F-v0
#        exp p2f_move_teacher                 exp p2f_move_student
# Both envs use Isaac Lab's native apply_external_force_torque shoves; only the
# privileged teacher obs group sees the applied wrench (student stays blind).
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo FULL_CD_FAIL; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_KEY_LOADED"; else echo "WANDB_KEY_MISSING"; fi

case "${WHICH:?set WHICH=p1f|p2f}" in
  p1f) T_TASK=Isaac-Basic-Teacher-K1-F-v0
       S_TASK=Isaac-Basic-Student-K1-F-v0
       EXP=p1f_basic_teacher
       T_ENVS=256 T_ITERS=2000 T_TMO=14400
       S_ENVS=256 S_ITERS=1500 S_TMO=14400 ;;
  p2f) T_TASK=Isaac-Velocity-Rough-K1-Teacher-F-v0
       S_TASK=Isaac-Velocity-Distill-K1-F-v0
       EXP=p2f_move_teacher
       T_ENVS=512 T_ITERS=3000 T_TMO=36000
       S_ENVS=512 S_ITERS=3000 S_TMO=28800 ;;
  *) echo "FULL_BAD_WHICH=$WHICH"; exit 2 ;;
esac

echo "=== FULL INSTALL booster_train (rsl_rl fork)"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -3
echo "=== FULL INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -3
echo "=== FULL INSTALL booster_assets (K1 URDF package)"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -3

echo "=== F SMOKE teacher $T_TASK 16x3"
timeout 3600 "$PY" isaac_tasks/k1_velocity/scripts/train.py \
  --task "$T_TASK" --num_envs 16 --max_iterations 3 --seed 42 --headless --enable_cameras
TS_RC=$?
echo "F_TEACHER_SMOKE_RC=$TS_RC"
[ "$TS_RC" -ne 0 ] && { echo "F_GATE=TEACHER_SMOKE_FAILED"; exit 10; }

T_EXTRA=""
[ -n "${RESUME_CKPT:-}" ] && T_EXTRA="--checkpoint $RESUME_CKPT" && echo "F_RESUME=$RESUME_CKPT"
echo "=== F FULL teacher $T_TASK ${T_ENVS}x$T_ITERS (timeout ${T_TMO}s) $T_EXTRA"
timeout "$T_TMO" "$PY" isaac_tasks/k1_velocity/scripts/train.py \
  --task "$T_TASK" --num_envs "$T_ENVS" --max_iterations "$T_ITERS" --seed 42 \
  --headless --enable_cameras --video --video_length 1500 --video_interval 4800 $T_EXTRA
T_RC=$?
echo "F_TEACHER_RC=$T_RC"

# newest run dir under the teacher experiment -> highest-iter checkpoint
RUN_DIR="$(ls -1t "$REPO/logs/rsl_rl/$EXP" 2>/dev/null | head -1)"
TEACHER_CKPT="$(ls -V "$REPO/logs/rsl_rl/$EXP/$RUN_DIR"/model_*.pt 2>/dev/null | tail -1)"
echo "TEACHER_CKPT=$TEACHER_CKPT"
if [ -z "$TEACHER_CKPT" ] || [ ! -f "$TEACHER_CKPT" ]; then
  echo "F_GATE=TEACHER_CKPT_MISSING"; exit 11
fi

echo "=== F SMOKE student $S_TASK 16x3 (teacher $TEACHER_CKPT)"
timeout 3600 "$PY" isaac_tasks/k1_velocity/scripts/train_student.py \
  --task "$S_TASK" --teacher_checkpoint "$TEACHER_CKPT" \
  --num_envs 16 --max_iterations 3 --seed 42 --headless --enable_cameras
SS_RC=$?
echo "F_STUDENT_SMOKE_RC=$SS_RC"
[ "$SS_RC" -ne 0 ] && { echo "F_GATE=STUDENT_SMOKE_FAILED"; exit 13; }

echo "=== F FULL student $S_TASK ${S_ENVS}x$S_ITERS (timeout ${S_TMO}s)"
timeout "$S_TMO" "$PY" isaac_tasks/k1_velocity/scripts/train_student.py \
  --task "$S_TASK" --teacher_checkpoint "$TEACHER_CKPT" \
  --num_envs "$S_ENVS" --max_iterations "$S_ITERS" --seed 42 \
  --headless --enable_cameras --video --video_length 1500 --video_interval 4800
S_RC=$?
echo "F_STUDENT_RC=$S_RC"
echo "F_CONTAINER_DONE TEACHER_RC=$T_RC STUDENT_RC=$S_RC"
exit "$S_RC"
