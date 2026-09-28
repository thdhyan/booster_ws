#!/bin/bash
# =============================================================================
# P6 push HOST launcher for zz-bw (blocker-6 port of dl_push_host.sh /
# spark_push_container.sh).
#
#   zzbw_push_host.sh reach   stage 1: walk up + wrist contact   256 x 1500
#   zzbw_push_host.sh push    stage 2: corner-goal pushing       256 x 3000
#                             (warm-starts from the newest reach ckpt)
#
# What it does:
#   1. runs scripts/push_preflight.sh (blocker 5) and gates PREFLIGHT_RESULT
#   2. points the models/k1_push_base.pt symlink at the squat-base export
#      (blocker 2a switch: --containall strips PUSH_* env vars, so the
#      symlink basename IS the mode selector; relative -> container-safe)
#   3. stages the cross-experiment warm start for push (reach run as a
#      DIRECT child symlink of logs/rsl_rl/p6_push - only direct children
#      resolve through train.py's --checkpoint relpath mapping)
#   4. launches run_k1_train.sh in tmux (session k1_push_<stage>) and EXITS
#      - the caller waits on the session, then gates the log
#
# Log/ckpt paths: the clone's logs/ is a symlink to /workspace/mounts/logs
# (= host /export/scratch/thakk100/k1/logs) -> checkpoints live host-side
# under $LOGROOT, NOT under <clone>/logs.
#
# Usage: zzbw_push_host.sh reach|push
#   Env: K1_CLONE, K1_HOST_LOG, K1_ENVS, K1_ITERS, K1_GPU, PUSH_BASE_MODE
# Exit: 2 usage, 10 preflight, 11 no squat base, 12 no reach ckpt,
#       13 session already exists, 14 launch died
# =============================================================================
set -u

WHICH="${1:-}"
case "$WHICH" in
    reach|push) ;;
    *) echo "usage: zzbw_push_host.sh reach|push" >&2; exit 2 ;;
esac

CLONE="${K1_CLONE:-/export/scratch/thakk100/k1/tmp/booster_ws}"
LOGROOT=/export/scratch/thakk100/k1/logs
LOG="${K1_HOST_LOG:-/export/scratch/thakk100/k1/${WHICH}_push.host.log}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PRE="$HERE/push_preflight.sh"
[ -f "$PRE" ] || PRE="$CLONE/scripts/push_preflight.sh"
RUNNER="$HOME/run_k1_train.sh"

# ---- 1. preflight (blocker 5) ---------------------------------------------
PRE_LOG="${LOG%.log}.preflight.log"
if ! bash "$PRE" "$CLONE" | tee "$PRE_LOG"; then
    :
fi
if ! grep -q '^PREFLIGHT_RESULT=PASS' "$PRE_LOG"; then
    echo "PUSH_HOST_GATE=PREFLIGHT_FAIL (see $PRE_LOG)"
    exit 10
fi

# ---- 2. frozen base: require the squat export, retarget the symlink -------
#      (mode = basename of the resolved path: "k1_push_base.pt" -> squat;
#       relink to k1_partialctrl_base.pt manually for legacy A/B)
mkdir -p "$CLONE/models"
if [ ! -f "$CLONE/models/k1_squat_base.pt" ]; then
    echo "PUSH_HOST_GATE=NO_SQUAT_BASE ($CLONE/models/k1_squat_base.pt missing)"
    echo "  -> run the chain's export stage, or: K1_TRAIN_SCRIPT=isaac_tasks/k1_velocity/scripts/play_record.py ~/run_k1_train.sh --task Isaac-Velocity-Squat-K1-Play-v0 --checkpoint <ct model_*.pt> --num_envs 2 --steps 2 --headless --export models/k1_squat_base.pt --trace_out /tmp/ignore_trace.npz"
    exit 11
fi
ln -sfn k1_squat_base.pt "$CLONE/models/k1_push_base.pt"
echo "[host] models/k1_push_base.pt -> $(readlink "$CLONE/models/k1_push_base.pt")"

SESSION="k1_push_${WHICH}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "PUSH_HOST_GATE=SESSION_EXISTS ($SESSION - kill or wait first)"
    exit 13
fi

# ---- 3. stage parameters + warm start -------------------------------------
case "$WHICH" in
    reach)
        TASK=Isaac-Push-Reach-K1-v0
        ENVS="${K1_ENVS:-256}"
        ITERS="${K1_ITERS:-1500}"
        EXTRA=""
        ;;
    push)
        TASK=Isaac-Push-K1-v0
        ENVS="${K1_ENVS:-256}"
        ITERS="${K1_ITERS:-3000}"
        WARM=$(ls -t "$LOGROOT/rsl_rl/p6_push_reach/"*/model_*.pt 2>/dev/null | head -1)
        if [ -z "$WARM" ]; then
            echo "PUSH_HOST_GATE=NO_REACH_CHECKPOINT ($LOGROOT/rsl_rl/p6_push_reach/)"
            exit 12
        fi
        REACH_RUN=$(basename "$(dirname "$WARM")")
        ITER0=${WARM##*/model_}
        ITER0=${ITER0%.pt}
        # rsl_rl treats --max_iterations as ADDITIONAL on resume
        # (displayed total = ITER0 + ITERS; gate = "2999/3000" for a 3000 goal)
        ITERS=$(( ITERS - ITER0 ))
        mkdir -p "$LOGROOT/rsl_rl/p6_push"
        ln -sfn "../p6_push_reach/$REACH_RUN" "$LOGROOT/rsl_rl/p6_push/warm_from_reach"
        CKPT_CT="/tmp/booster_ws/logs/rsl_rl/p6_push/warm_from_reach/model_${ITER0}.pt"
        EXTRA="--checkpoint $CKPT_CT"
        echo "[host] warm start: $CKPT_CT (resume iter=$ITER0, target total=$((ITER0 + ITERS)))"
        ;;
esac

# ---- 4. launch + quick liveness check -------------------------------------
: > "$LOG"
tmux new-session -d -s "$SESSION" \
    "K1_TRAIN_SCRIPT=run_unbuffered.py K1_GPU=${K1_GPU:-1} $RUNNER --task $TASK --num_envs $ENVS --max_iterations $ITERS --seed 42 --headless $EXTRA >> '$LOG' 2>&1"
sleep 8
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "PUSH_HOST_GATE=LAUNCH_DIED (log=$LOG)"
    tail -20 "$LOG" 2>/dev/null
    exit 14
fi
echo "PUSH_HOST_STARTED=1 session=$SESSION task=$TASK ${ENVS}x${ITERS} log=$LOG"
echo "  gate after the session ends: final 'Learning iteration ...' + no Traceback"
exit 0
