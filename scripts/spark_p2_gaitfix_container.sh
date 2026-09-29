#!/bin/bash
# P2 gait-structure campaign: train with rewards that distinguish a walk from a
# jitter, then require BOTH a movement gate and a gait gate before trusting the
# checkpoint.
#
# WHY A FRESH RUN
# ---------------
# The previous checkpoint is a high-frequency shuffle: 6.75 steps/s against 1.8-2.2
# for human walking, 3-5 cm strides, 8/8 envs failing the gait gate. Resuming from
# it would inherit that gait as the local optimum the policy sits in. Sharpening
# the tracking reward and adding gait terms changes the objective, so the policy
# needs to re-find its solution from a standing start.
#
# TWO GATES, BOTH REQUIRED
# ------------------------
# Movement is necessary but not sufficient -- the displacement gate passed the
# shuffle at 2.664 m with zero falls. Only the gait gate (cadence, stride, jerk)
# decides whether this walks. The run is not certified unless both pass.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo P2_GAITS_CD_FAIL; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] || { echo P2_GAITS_WANDB_KEY_MISSING; exit 2; }

TASK=Isaac-Velocity-Rough-K1-Teacher-v0
LOGDIR="$REPO/scripts"
PREFLIGHT_LOG="$LOGDIR/p2_gaits.preflight.log"
TRAIN_LOG="$LOGDIR/p2_gaits.teacher.train.log"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== P2_GAITS INSTALL"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -1

# Pre-flight. Now doing more than standing and terminations: the five new reward
# terms call env.action_manager.action, env.scene[...].body_pos and
# env.scene.sensors[...].data.compute_contact_sensor_data(), and any of those
# being wrong crashes a 3000-iteration run on its first step. The probe is
# cheap and it exercises every reward term.
echo "=== P2_GAITS PREFLIGHT (standing + terminations + all reward terms)"
timeout 1800 "$PY" -u isaac_tasks/k1_velocity/scripts/probe_rewards.py \
  --task "$TASK" --num_envs 8 --steps 200 --viz none 2>&1 | tee "$PREFLIGHT_LOG"
grep -q "REWARD_PROBE_MARKER=OK" "$PREFLIGHT_LOG" || { echo P2_GAITS_PREFLIGHT_MARKER=FAIL; exit 30; }
grep -q "STANDS" "$PREFLIGHT_LOG" || { echo P2_GAITS_PREFLIGHT_MARKER=FAIL_NOT_STANDING; exit 31; }
# The new terms must appear and must be finite. A NaN or an absent term means
# the gait shaping is silently not running, which is exactly the failure class
# this whole campaign exists to stop.
for t in gait_cadence feet_clearance feet_alternation stride_length action_jerk_l2; do
  if ! grep -q "$t" "$PREFLIGHT_LOG"; then
    echo "P2_GAITS_PREFLIGHT_MARKER=FAIL_MISSING_TERM_$t"
    exit 32
  fi
done
echo "P2_GAITS_PREFLIGHT_MARKER=OK"

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

ckpt=$(find "$REPO/logs/rsl_rl" -type f -name model_2999.pt -newermt '-4 hours' \
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
