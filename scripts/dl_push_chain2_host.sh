#!/bin/bash
# Auto-chain AFTER the dl base RESUME — v2 (corrected gates).
# v1 gated on "Learning iteration 2999/3000" + model_2999.pt, but rsl_rl treats
# --max_iterations as ADDITIONAL iterations on resume (800 + 3000 = 3800
# displayed), so those markers never appear. v2 gates on what actually happens:
#   1. wait for tmux k1_dl_push_base_resume to end
#   2. gate: newest run (non-copied) under logs/rsl_rl/k1_partialctrl_base has
#      a model_*.pt with iteration >= 3700 AND the log shows it reached /3800
#   3. re-export models/k1_partialctrl_base.pt (gate: RC=0 + fresh mtime)
#   4. two-pass frozen diag (diag_frozen.sh)
#   5. gate on pass-B done_rate (< 0.02: fresh base must STAND in P6)
#   6. launch reach training (dl_push_host.sh reach, GPU 1)
# Log: scripts/dl_push_chain2.log
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/dl_push_chain2.log"
: > "$LOG"
say() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
DOCKER="docker run --rm --gpus device=1 --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"

say "DL_PUSH_CHAIN2_WAIT=k1_dl_push_base_resume"
while tmux has-session -t k1_dl_push_base_resume 2>/dev/null; do sleep 60; done
say "DL_PUSH_CHAIN2 base resume session ended"

# --- Gate 1: training actually reached the end of the 3800-iter resume run ---
HLOG="$HERE/push_base.resume.host.log"
if ! grep -q "Learning iteration 2/3" "$HLOG"; then
  say "DL_PUSH_CHAIN2_GATE=SMOKE_NEVER_OK"
  exit 10
fi
if ! grep -qE "Learning iteration (3[78][0-9][0-9])/3800" "$HLOG"; then
  say "DL_PUSH_CHAIN2_GATE=TRAINING_DID_NOT_REACH_3700+ (last iter line: $(grep -o 'Learning iteration [0-9]*/[0-9]*' "$HLOG" | tail -1))"
  exit 11
fi
# newest run dir, excluding the materialized 2026-09-24 copy
RUNDIR=$(ls -td "$REPO"/logs/rsl_rl/k1_partialctrl_base/*/ 2>/dev/null | grep -v '2026-09-24_18-44-39' | head -1)
LAST_MODEL=$(ls "$RUNDIR"model_*.pt 2>/dev/null | sort -V | tail -1)
LAST_ITER=$(echo "$LAST_MODEL" | grep -oE '[0-9]+\.pt$' | tr -d '.pt')
if [ -z "$LAST_MODEL" ] || [ "${LAST_ITER:-0}" -lt 3700 ]; then
  say "DL_PUSH_CHAIN2_GATE=NO_FINAL_CKPT (run=$RUNDIR last=${LAST_MODEL:-NONE})"
  exit 12
fi
say "DL_PUSH_CHAIN2_GATE=RESUME_OK run=$(basename "$RUNDIR") last=$(basename "$LAST_MODEL")"

# --- Stage: EXPORT -----------------------------------------------------------
say "DL_PUSH_CHAIN2_STAGE=EXPORT"
NOW=$(date +%s)
timeout 3600 $DOCKER /workspace/booster_ws/scripts/export_base_policy.sh >> "$LOG" 2>&1
MT=$(stat -c %Y "$REPO/models/k1_partialctrl_base.pt" 2>/dev/null || echo 0)
if ! grep -q "BASE_EXPORT_RC=0" "$LOG" || [ "$MT" -lt "$NOW" ]; then
  say "DL_PUSH_CHAIN2_GATE=EXPORT_FAILED (rc=$(grep -o 'BASE_EXPORT_RC=[0-9]*' "$LOG" | tail -1) mtime=$MT now=$NOW)"
  exit 13
fi
say "DL_PUSH_CHAIN2 export fresh: $(ls -la "$REPO/models/k1_partialctrl_base.pt" | awk '{print $5, $6, $7, $8}')"

# --- Stage: DIAG (two-pass frozen A/B) ---------------------------------------
say "DL_PUSH_CHAIN2_STAGE=DIAG"
timeout 3600 $DOCKER /workspace/booster_ws/scripts/diag_frozen.sh >> "$LOG" 2>&1
if ! grep -q "DIAG_FROZEN_ALL_DONE" "$LOG"; then
  say "DL_PUSH_CHAIN2_GATE=DIAG_DID_NOT_FINISH"
  exit 14
fi
DR=$(grep -o 'done_rate=[0-9.]*' "$LOG" | tail -1 | cut -d= -f2)
say "DL_PUSH_CHAIN2_DIAG pass-B done_rate=${DR:-PARSE_FAIL}"
if [ -z "$DR" ]; then
  say "DL_PUSH_CHAIN2_GATE=DIAG_PARSE_FAILED"
  exit 15
fi
if ! awk -v v="$DR" 'BEGIN{exit !(v < 0.02)}'; then
  say "DL_PUSH_CHAIN2_GATE=DIAG_FAILED_BASE_STILL_FALLS (done_rate=$DR >= 0.02) — NOT launching reach; next: in-env A/B hunt"
  exit 16
fi

# --- Stage: REACH ------------------------------------------------------------
say "DL_PUSH_CHAIN2_STAGE=REACH_LAUNCH (gate passed)"
DL_GPUS=device=1 bash "$HERE/dl_push_host.sh" reach >> "$LOG" 2>&1
say "DL_PUSH_CHAIN2_DONE (reach training live in tmux k1_spark_push_reach)"
