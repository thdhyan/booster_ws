#!/bin/bash
# =============================================================================
# P6 push SMOKE on zz-bw bare metal (blocker-6 port of scripts/smoke_push.sh
# to the run_k1_train.sh + entrypoint.sh flow).
#
#   zzbw_push_smoke.sh reach   Isaac-Push-Reach-K1-v0, 16 envs x 3 iters
#   zzbw_push_smoke.sh push    Isaac-Push-K1-v0,       16 envs x 3 iters
#
# WITH debug video (compulsory): --video --video_length 48 --video_interval 24.
# Gates on log MARKERS, never rc (python.sh returns 0 on Traceback):
#   PUSH_SMOKE_GATE=OK requires
#     [k1-entrypoint] workspace packages import OK   (v3 push pkg imports)
#     Learning iteration 2/3                          (3 iters actually ran)
#     no "Traceback"
#
# STANDING RULE: after PUSH_SMOKE_GATE=OK, extract a PNG frame from the video
# (imageio inside the SIF - no ffmpeg on host) and LOOK AT IT before calling
# the smoke done. Video lands at <clone>/videos/train/rl-video-step-*.mp4.
#
# Usage: zzbw_push_smoke.sh reach|push
#   Env: K1_SMOKE_LOG (default /export/scratch/thakk100/k1/<stage>_push_smoke.log)
#        K1_GPU (run_k1_train default 1), SMOKE_TIMEOUT (default 3600 s)
# =============================================================================
set -u

WHICH="${1:-}"
case "$WHICH" in
    reach) TASK=Isaac-Push-Reach-K1-v0 ;;
    push)  TASK=Isaac-Push-K1-v0 ;;
    *) echo "usage: zzbw_push_smoke.sh reach|push" >&2; exit 2 ;;
esac

LOG="${K1_SMOKE_LOG:-/export/scratch/thakk100/k1/${WHICH}_push_smoke.log}"
CLONE="${K1_CLONE:-/export/scratch/thakk100/k1/tmp/booster_ws}"
RUNNER="$HOME/run_k1_train.sh"

: > "$LOG"
echo "[smoke] task=$TASK num_envs=16 iters=3 video=48/24 log=$LOG"
timeout "${SMOKE_TIMEOUT:-3600}" \
    env K1_TRAIN_SCRIPT=run_unbuffered.py K1_GPU="${K1_GPU:-1}" \
    "$RUNNER" \
    --task "$TASK" --num_envs 16 --max_iterations 3 --seed 42 \
    --headless --video --video_length 48 --video_interval 24 \
    >> "$LOG" 2>&1
RC=$?
echo "[smoke] runner rc=$RC (informational - markers are authoritative)"

GATE=FAIL
if grep -q '\[k1-entrypoint\] workspace packages import OK' "$LOG" \
        && grep -q 'Learning iteration 2/3' "$LOG" \
        && ! grep -q 'Traceback' "$LOG"; then
    GATE=OK
fi
echo "PUSH_SMOKE_GATE=$GATE"
echo "PUSH_SMOKE_LOG=$LOG"

echo "--- video artifacts (extract a PNG frame + view it before done):"
ls -la "$CLONE/videos/train/" 2>/dev/null | tail -6

if [ "$GATE" != "OK" ]; then
    echo "--- last 40 log lines:"
    tail -40 "$LOG"
    exit 1
fi
exit 0
