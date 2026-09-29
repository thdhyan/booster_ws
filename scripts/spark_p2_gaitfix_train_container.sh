#!/bin/bash
# P2 gait-structure campaign, stage 2 of 2: smoke, full training, then BOTH gates.
#
# Runs in its own container because AppLauncher deadlocks on a second Kit launch
# in the same container. The preflight already ran in a separate container and
# passed, so this container performs exactly one Kit launch sequence per process
# it starts -- and the smoke and the full run are separate processes, which is
# fine because Kit is torn down with each process.
#
# Certification requires BOTH gates on real footage from the new checkpoint:
# MOVEMENT_GATE (net displacement > 0.5 m) and GAIT_GATE (cadence, stride, jerk).
# Movement alone is not sufficient -- it passed an 8 Hz chatter that covered
# 2.66 m without taking a real step.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo P2_GAITS_CD_FAIL; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] || { echo P2_GAITS_WANDB_KEY_MISSING; exit 2; }

TASK=Isaac-Velocity-Rough-K1-Teacher-v0
LOGDIR="$REPO/scripts"
TRAIN_LOG="$LOGDIR/p2_gaits.teacher.train.log"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== P2_GAITS INSTALL"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -1

echo "=== P2_GAITS SMOKE 16x3"
timeout 1800 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 16 --max_iterations 3 --seed 44 --viz none \
  2>&1 | tee "$LOGDIR/p2_gaits.smoke.log"
rc=${PIPESTATUS[0]}
echo "P2_GAITS_SMOKE_RC=$rc"
[ "$rc" -eq 0 ] && grep -q "Learning iteration 2/3" "$LOGDIR/p2_gaits.smoke.log" \
  || { echo P2_GAITS_GATE=SMOKE_FAILED_NO_FULL_RUN; exit 10; }

echo "=== P2_GAITS FULL TEACHER 512x3000 (gait-structured rewards)"
timeout 43200 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 512 --max_iterations 3000 --seed 44 --viz none \
  2>&1 | tee "$TRAIN_LOG"
rc=${PIPESTATUS[0]}
echo "P2_GAITS_TEACHER_RC=$rc"

ckpt=$(find "$REPO/logs/rsl_rl" -type f -name model_2999.pt -newermt '-5 hours' \
       -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ "$rc" -eq 0 ] && [ -s "$ckpt" ]; then
  echo "P2_GAITS_TEACHER_CKPT=$ckpt"
  echo "P2_GAITS_TEACHER_MARKER=OK"
else
  echo "P2_GAITS_TEACHER_MARKER=FAIL"
  exit 11
fi

# ---- both gates, on real footage from this checkpoint ----------------------
echo "=== P2_GAITS EVAL: movement gate then gait gate"
EVALOUT="$LOGDIR/p2_gaits.eval.log"
: > "$EVALOUT"
TRACE="$REPO/isaac_tasks/k1_velocity/videos/p2_gaits_${STAMP}.npz"
timeout 2400 "$PY" -u isaac_tasks/k1_velocity/scripts/play_record.py \
  --task "$TASK" --checkpoint "$ckpt" \
  --num_envs 8 --steps 750 --cmd 0.5 0.0 0.0 \
  --trace_out "$TRACE" \
  --label "P2 gait-structured ckpt 2999 | cmd vx=+0.5" \
  2>&1 | tail -6

if [ ! -s "$TRACE" ]; then
  echo "P2_GAITS_EVAL_MARKER=FAIL_NO_TRACE"
  echo "P2_GAITS_GATE=UNEVALUATED_NO_TRUST_IN_CHECKPOINT"
  exit 20
fi

"$PY" - "$TRACE" <<'PYEOF' | tee -a "$EVALOUT"
import sys
import numpy as np
d = np.load(sys.argv[1])
xy = d["root_pos"][..., :2]
net = np.linalg.norm(xy[-1] - xy[0], axis=-1)
print(f"MOVEMENT net_displacement mean={net.mean():.3f} m  min={net.min():.3f}")
print(f"MOVEMENT_GATE={'PASS' if net.mean() > 0.5 else 'FAIL'} (0.5 m bar)")
PYEOF

"$PY" -u "$REPO/isaac_tasks/k1_velocity/scripts/gait_gate.py" "$TRACE" 2>&1 | tee -a "$EVALOUT"
gait_rc=${PIPESTATUS[0]}

if ! grep -q "MOVEMENT_GATE=PASS" "$EVALOUT"; then
  echo "P2_GAITS_EVAL_MARKER=FAIL_NO_MOVEMENT"
  exit 21
fi
if [ "$gait_rc" -ne 0 ]; then
  echo "P2_GAITS_EVAL_MARKER=FAIL_MOVES_BUT_DOES_NOT_WALK"
  echo "P2_GAITS_GATE=WALKS_BUT_DOES_NOT_WALK_FIX_REWARD_NOT_NETWORK"
  exit 22
fi
echo "P2_GAITS_EVAL_MARKER=OK_WALKS"
echo P2_GAITS_ALL_DONE
exit 0
