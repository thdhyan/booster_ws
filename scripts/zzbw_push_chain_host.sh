#!/bin/bash
# =============================================================================
# P6 push FULL CHAIN on zz-bw (blocker 6) - run INSIDE tmux:
#   tmux new -s k1_push_chain 'scripts/zzbw_push_chain_host.sh'
#
# Sequential stages, gated on log MARKERS (never rc - python.sh exits 0 on
# Traceback). Any gate failure stops the chain with PUSH_CHAIN_RESULT=FAILED
# and the failing stage name.
#
#   0  preflight (blocker 5; PUSH_V3_CODE requires the v3 files rsynced here)
#   1  wait for the Phase-1 squat-teacher run (k1_squat_full) to END
#      - the clone stays on feat/velocity-squat the WHOLE chain: no branch
#        switch ever happens (branch switching under/around a running
#        training is what we're avoiding; v3 push files ride in via rsync)
#   2  gate squat_full.log: final iteration + no Traceback + newest ckpt
#   3  EXPORT frozen squat base -> models/k1_squat_base.pt (Play task,
#      K1_TRAIN_SCRIPT=play_record.py; also prints its own JIT parity)
#   4  OFFLINE PARITY (blocker 4 gate): shipped .pt vs runner actor, 236/12
#      -> PARITY_OK
#   5  SMOKE reach (16x3 + video)  -> PUSH_SMOKE_GATE=OK
#   6  HOST reach tmux k1_push_reach (256x1500) -> final "1499/1500"
#   7  SMOKE push  (16x3 + video)  -> PUSH_SMOKE_GATE=OK
#   8  HOST push tmux k1_push_push (256x3000, warm start) -> final "2999/3000"
#   9  summary: checkpoints, videos, PNG-frame reminder
#
# Prereqs (done by the authoring session BEFORE starting the chain):
#   - v3 push files + these scripts rsynced into the clone (marker-gated)
#   - clone on feat/velocity-squat (checked in stage 2)
#   - Phase-1 squat training running as tmux k1_squat_full (or already done)
#
# Env overrides: CHAIN_LOGROOT (/export/scratch/thakk100/k1),
#   SQUAT_EXPECT_MARKER (default "Learning iteration 4999/5000"),
#   EXPORT_TIMEOUT / PARITY_TIMEOUT / SMOKE_TIMEOUT seconds.
# =============================================================================
set -u

ROOT="${CHAIN_LOGROOT:-/export/scratch/thakk100/k1}"
CLONE="${K1_CLONE:-$ROOT/tmp/booster_ws}"
LOGROOT="$ROOT/logs"                 # clone logs/ symlinks here (container bind)
RUNNER="$HOME/run_k1_train.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
SQUAT_LOG="$ROOT/squat_full.log"
SQUAT_EXPECT_MARKER="${SQUAT_EXPECT_MARKER:-Learning iteration 4999/5000}"
STAGE=start

stage() { STAGE="$1"; shift; echo; echo "=== [chain:$STAGE] $* $(date '+%F %T')"; }
die()   { echo "PUSH_CHAIN_GATE=FAIL stage=$STAGE  # $*"; echo "PUSH_CHAIN_RESULT=FAILED"; exit 1; }

wait_session() { # wait_session <tmux-name> <timeout-s; 0=infinite>
    local s="$1" to="$2" t=0
    while tmux has-session -t "$s" 2>/dev/null; do
        sleep 60; t=$((t + 60))
        if [ "$to" -gt 0 ] && [ "$t" -ge "$to" ]; then
            tmux kill-session -t "$s" 2>/dev/null
            echo "[chain] session $s exceeded ${to}s - killed"
            return 1
        fi
        [ $((t % 600)) -eq 0 ] && echo "[chain] waiting on tmux $s (${t}s elapsed)"
    done
    echo "[chain] tmux $s ended after ${t}s"
    return 0
}

# --------------------------------------------------------------- stage 0 ----
stage 0 "preflight (blocker 5)"
PRE_LOG="$ROOT/chain_preflight.log"
bash "$HERE/push_preflight.sh" "$CLONE" | tee "$PRE_LOG" || true
grep -q '^PREFLIGHT_RESULT=PASS' "$PRE_LOG" || die "preflight failed (see $PRE_LOG)"

# --------------------------------------------------------------- stage 1 ----
stage 1 "wait for Phase-1 squat-teacher (tmux k1_squat_full) to end"
if tmux has-session -t k1_squat_full 2>/dev/null; then
    echo "[chain] k1_squat_full still running - waiting (prints every 10 min)"
    wait_session k1_squat_full 0 || die "wait for k1_squat_full interrupted"
else
    echo "[chain] k1_squat_full not running (already finished or never started)"
fi

# --------------------------------------------------------------- stage 2 ----
stage 2 "gate Phase-1 completion + clone branch + newest squat ckpt"
BR=$(git -C "$CLONE" branch --show-current 2>/dev/null)
[ "$BR" = "feat/velocity-squat" ] || die "clone on '$BR', need feat/velocity-squat (export task lives there; do NOT switch under a run)"
grep -qF "$SQUAT_EXPECT_MARKER" "$SQUAT_LOG" || die "missing final marker '$SQUAT_EXPECT_MARKER' in $SQUAT_LOG (run killed / config drift?)"
grep -q 'Traceback' "$SQUAT_LOG" && die "Traceback in $SQUAT_LOG"
SQUAT_CKPT=$(ls -t "$LOGROOT/rsl_rl/k1_squat_teacher/"*/model_*.pt 2>/dev/null | head -1)
[ -n "$SQUAT_CKPT" ] || die "no squat ckpt under $LOGROOT/rsl_rl/k1_squat_teacher/"
# host -> container path: clone/logs symlinks to the same bind-mounted tree
CT_CKPT="${SQUAT_CKPT/#$LOGROOT//tmp/booster_ws/logs}"
echo "[chain] squat ckpt: $SQUAT_CKPT"
echo "[chain] container : $CT_CKPT"

# --------------------------------------------------------------- stage 3 ----
stage 3 "export frozen squat base -> models/k1_squat_base.pt"
EXPORT_LOG="$ROOT/squat_export.log"
rm -f "$CLONE/models/k1_squat_base.pt"
timeout "${EXPORT_TIMEOUT:-3600}" \
    env K1_TRAIN_SCRIPT=isaac_tasks/k1_velocity/scripts/play_record.py K1_GPU="${K1_GPU:-1}" \
    "$RUNNER" \
    --task Isaac-Velocity-Squat-K1-Play-v0 \
    --checkpoint "$CT_CKPT" \
    --num_envs 2 --steps 2 --headless \
    --export models/k1_squat_base.pt \
    --trace_out /tmp/ignore_trace.npz \
    >> "$EXPORT_LOG" 2>&1
grep -q 'TorchScript saved' "$EXPORT_LOG" || { tail -40 "$EXPORT_LOG"; die "export marker missing (see $EXPORT_LOG)"; }
grep -q 'Traceback' "$EXPORT_LOG" && die "Traceback in export log"
[ -f "$CLONE/models/k1_squat_base.pt" ] || die "models/k1_squat_base.pt not created"
grep 'JIT parity' "$EXPORT_LOG" || true

# --------------------------------------------------------------- stage 4 ----
stage 4 "offline parity: shipped .pt vs runner actor (236 -> 12)"
PARITY_LOG="$ROOT/squat_parity.log"
timeout "${PARITY_TIMEOUT:-900}" \
    env K1_TRAIN_SCRIPT=scripts/push_export_parity.py K1_GPU="${K1_GPU:-1}" \
    "$RUNNER" \
    --shipped models/k1_squat_base.pt \
    --ckpt "$CT_CKPT" \
    --layout squat \
    >> "$PARITY_LOG" 2>&1
grep 'PARITY_MAX_DIFF' "$PARITY_LOG" || true
grep -q '^PARITY_OK' "$PARITY_LOG" || { tail -30 "$PARITY_LOG"; die "PARITY_OK missing (see $PARITY_LOG)"; }

# --------------------------------------------------------------- stage 5 ----
stage 5 "SMOKE reach (16x3 + debug video)"
K1_SMOKE_LOG="$ROOT/reach_push_smoke.log" bash "$HERE/zzbw_push_smoke.sh" reach \
    | tee "$ROOT/chain_smoke_reach.log"
grep -q 'PUSH_SMOKE_GATE=OK' "$ROOT/chain_smoke_reach.log" || die "reach smoke failed"
echo "[chain] ACTION REQUIRED when you next look: extract a PNG frame from" \
     "$CLONE/videos/train/ and view it (standing rule)"

# --------------------------------------------------------------- stage 6 ----
stage 6 "HOST reach 256x1500 (tmux k1_push_reach)"
bash "$HERE/zzbw_push_host.sh" reach || die "reach host launch failed"
wait_session k1_push_reach 0 || die "reach wait interrupted"
grep -q 'Learning iteration 1499/1500' "$ROOT/reach_push.host.log" \
    || { tail -30 "$ROOT/reach_push.host.log"; die "reach final marker 1499/1500 missing"; }
grep -q 'Traceback' "$ROOT/reach_push.host.log" && die "Traceback in reach log"
ls -t "$LOGROOT/rsl_rl/p6_push_reach/"*/model_1499.pt >/dev/null 2>&1 \
    || die "reach model_1499.pt missing"

# --------------------------------------------------------------- stage 7 ----
stage 7 "SMOKE push (16x3 + debug video)"
K1_SMOKE_LOG="$ROOT/push_push_smoke.log" bash "$HERE/zzbw_push_smoke.sh" push \
    | tee "$ROOT/chain_smoke_push.log"
grep -q 'PUSH_SMOKE_GATE=OK' "$ROOT/chain_smoke_push.log" || die "push smoke failed"

# --------------------------------------------------------------- stage 8 ----
stage 8 "HOST push 256x3000 warm-started (tmux k1_push_push)"
bash "$HERE/zzbw_push_host.sh" push || die "push host launch failed"
wait_session k1_push_push 0 || die "push wait interrupted"
grep -q 'Learning iteration 2999/3000' "$ROOT/push_push.host.log" \
    || { tail -30 "$ROOT/push_push.host.log"; die "push final marker 2999/3000 missing"; }
grep -q 'Traceback' "$ROOT/push_push.host.log" && die "Traceback in push log"
PUSH_CKPT=$(ls -t "$LOGROOT/rsl_rl/p6_push/"*/model_*.pt 2>/dev/null | head -1)
[ -n "$PUSH_CKPT" ] || die "push ckpt missing under $LOGROOT/rsl_rl/p6_push/"

# --------------------------------------------------------------- stage 9 ----
stage 9 "ALL STAGES DONE"
echo "PUSH_CHAIN_RESULT=ALL_OK"
echo "  reach ckpt: $(ls -t "$LOGROOT/rsl_rl/p6_push_reach/"*/model_*.pt | head -1)"
echo "  push  ckpt: $PUSH_CKPT"
echo "  base     : $CLONE/models/k1_squat_base.pt"
echo "  logs     : $ROOT/{reach,push}_push.host.log  chain stage logs $ROOT/chain_*"
echo "  videos   : $CLONE/videos/train/  (smoke clips)"
echo "  REMINDER : view a PNG frame of every smoke video before declaring it" \
     "done; then Phase 4 = play-run eval videos (corner-error/success report)."
exit 0
