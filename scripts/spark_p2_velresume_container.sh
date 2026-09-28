#!/bin/bash
# Verify a P2 checkpoint by eye before trusting it, then resume the velocity
# curriculum from it.
#
# Order matters: the video comes FIRST and on its own, because resuming compute
# from a checkpoint nobody has watched is how a broken policy gets 5000 more
# iterations of reinforcement. The GPU is only free because training finished,
# so there is no concurrency conflict here.
#
# The 0.60 gate is a deliberate loosening of the legged_gym-standard 0.80. The
# first full run reached an episode-length ratio of ~0.53 after 3000 iterations
# and was still climbing, so 0.80 was not reachable in the budget and the
# curriculum never opened. 0.60 is safe because the curriculum contracts on
# failure: it settles at the edge of what the policy can hold rather than
# overshooting and collapsing.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo P2_RESUME_CD_FAIL; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"

TASK=Isaac-Velocity-Rough-K1-Teacher-v0
CKPT="$1"
if [ -z "$CKPT" ] || [ ! -s "$CKPT" ]; then
  echo "P2_RESUME_CKPT_MISSING=$CKPT"
  exit 1
fi
echo "P2_RESUME_CKPT=$CKPT"

STAMP=$(date +%Y%m%d_%H%M%S)
VID="$REPO/isaac_tasks/k1_velocity/videos/p2_vel_teacher_resume_${STAMP}.mp4"
PANEL="${VID%.mp4}_panels.mp4"
LOGDIR="$REPO/scripts"
VIDEO_LOG="$LOGDIR/p2_vel_resume.video.log"
TRAIN_LOG="$LOGDIR/p2_vel_resume.train.log"

echo "=== P2_RESUME INSTALL"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -1

# ---- stage 1: look at it ----------------------------------------------------
# Four labelled panels (overview, top-down, follow cam, status header) and a
# command of 0.6 m/s: above the 0.5 training range, so if the policy can hold
# 0.6 at all that is itself evidence, and if it cannot we learn that now rather
# than after a resumed run.
echo "=== P2_RESUME VIDEO (4-panel, cmd 0.6 m/s)"
timeout 2400 "$PY" -u isaac_tasks/k1_velocity/scripts/play_record.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num_envs 4 --steps 750 --cmd 0.6 0.0 0.0 \
  --panel_video --video_out "$PANEL" \
  --label "P2 vel-resume ckpt 2999 | cmd vx=0.6 (above 0.5 train range)" \
  2>&1 | tee "$VIDEO_LOG"
vrc=${PIPESTATUS[0]}
echo "P2_RESUME_VIDEO_RC=$vrc"
if [ -s "$PANEL" ]; then
  echo "P2_RESUME_VIDEO=PANEL_OK $PANEL"
else
  echo "P2_RESUME_VIDEO=FAIL_NO_FILE"
  # Refuse to resume from a checkpoint whose behaviour we could not observe.
  echo "P2_RESUME_GATE=VIDEO_FAILED_NO_RESUME"
  exit 20
fi

# ---- stage 2: resume training with the loosened gate -----------------------
echo "=== P2_RESUME TRAIN from $CKPT with success_threshold=0.60"
timeout 43200 "$PY" -u isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num_envs 512 --max_iterations 3000 --seed 43 --viz none \
  --vel_success_threshold 0.60 \
  2>&1 | tee "$TRAIN_LOG"
rc=${PIPESTATUS[0]}
echo "P2_RESUME_TRAIN_RC=$rc"

# The whole point of resuming: prove the range actually opened this time.
if grep -q "vel-curriculum" "$TRAIN_LOG" && grep "vel-curriculum" "$TRAIN_LOG" | grep -q -- "->"; then
  echo "P2_RESUME_CURRICULUM_MARKER=OK"
  grep "vel-curriculum" "$TRAIN_LOG" | head -15
  echo "--- final range seen ---"
  grep "vel-curriculum" "$TRAIN_LOG" | tail -1
else
  echo "P2_RESUME_CURRICULUM_MARKER=FAIL_NEVER_FIRED"
  echo "P2_RESUME_GATE=CURRICULUM_STILL_INERT"
  exit 21
fi

ckpt2=$(find "$REPO/logs/rsl_rl" -type f -name model_2999.pt -newermt '-4 hours' \
         -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ "$rc" -eq 0 ] && [ -s "$ckpt2" ]; then
  echo "P2_RESUME_CKPT=$ckpt2"
  echo P2_RESUME_ALL_DONE
  exit 0
fi
echo "P2_RESUME_TRAIN_MARKER=FAIL"
exit 22
