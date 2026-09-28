#!/bin/bash
# Auto-chain AFTER the dl base RESUME (host side, dl; run inside tmux):
#   1. wait for tmux k1_dl_push_base_resume to end
#   2. gate on PUSH_BASE_RESUME_FULL_MARKER=OK in push_base.resume.host.log
#   3. re-export models/k1_partialctrl_base.pt (export_base_policy.sh)
#   4. two-pass frozen diag (diag_frozen.sh)
#   5. gate on pass-B done_rate (< 0.02: the fresh base must STAND in P6)
#   6. launch reach training (dl_push_host.sh reach, own smoke gate, GPU 1)
# Any gate failure STOPS the chain with a grep-able marker.
# Log: scripts/dl_push_chain.log
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/dl_push_chain.log"
: > "$LOG"
say() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
DOCKER="docker run --rm --gpus device=1 --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"

say "DL_PUSH_CHAIN_WAIT=k1_dl_push_base_resume"
while tmux has-session -t k1_dl_push_base_resume 2>/dev/null; do sleep 60; done
say "DL_PUSH_CHAIN base resume session ended"

if ! grep -q "PUSH_BASE_RESUME_FULL_MARKER=OK" "$HERE/push_base.resume.host.log"; then
  say "DL_PUSH_CHAIN_GATE=BASE_RESUME_NOT_OK"
  exit 10
fi
say "DL_PUSH_CHAIN_STAGE=EXPORT (re-export frozen base)"
timeout 3600 $DOCKER /workspace/booster_ws/scripts/export_base_policy.sh >> "$LOG" 2>&1
if ! grep -q "BASE_EXPORT_DONE" "$LOG"; then
  say "DL_PUSH_CHAIN_GATE=EXPORT_FAILED"
  exit 11
fi
say "DL_PUSH_CHAIN_STAGE=DIAG (two-pass frozen A/B)"
timeout 3600 $DOCKER /workspace/booster_ws/scripts/diag_frozen.sh >> "$LOG" 2>&1
DR=$(grep -o 'done_rate=[0-9.]*' "$LOG" | tail -1 | cut -d= -f2)
say "DL_PUSH_CHAIN_DIAG pass-B done_rate=${DR:-PARSE_FAIL}"
if [ -z "$DR" ]; then
  say "DL_PUSH_CHAIN_GATE=DIAG_PARSE_FAILED"
  exit 12
fi
if ! awk -v v="$DR" 'BEGIN{exit !(v < 0.02)}'; then
  say "DL_PUSH_CHAIN_GATE=DIAG_FAILED_BASE_STILL_FALLS (done_rate=$DR >= 0.02) — NOT launching reach; next: in-env A/B hunt"
  exit 13
fi
say "DL_PUSH_CHAIN_STAGE=REACH_LAUNCH (gate passed)"
DL_GPUS=device=1 bash "$HERE/dl_push_host.sh" reach >> "$LOG" 2>&1
say "DL_PUSH_CHAIN_DONE (reach training live in tmux k1_spark_push_reach)"
