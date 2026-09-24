#!/bin/bash
# Auto-chain AFTER the gait-v2 base retrain (host side, spark04; run inside
# tmux — laptop-independent):
#   1. wait for tmux k1_spark_push_base to end
#   2. gate on PUSH_BASE_FULL_MARKER=OK in scripts/push_base.host.log
#   3. re-export models/k1_partialctrl_base.pt (export_base_policy.sh)
#   4. two-pass frozen diag (diag_frozen.sh)
#   5. gate on the pass-B done_rate (fresh base must STAND in P6: < 0.02)
#   6. launch reach training (spark_push_host.sh reach, own smoke gate)
# If any gate fails the chain STOPS and logs the marker — no wasted GPU.
# Log: scripts/push_base_chain.log
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/push_base_chain.log"
: > "$LOG"
say() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
DOCKER="docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -v $HERE/k1_isaac_cache:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"

say "PUSH_BASE_CHAIN_WAIT=k1_spark_push_base"
while tmux has-session -t k1_spark_push_base 2>/dev/null; do sleep 60; done
say "PUSH_BASE_CHAIN base session ended"

if ! grep -q "PUSH_BASE_FULL_MARKER=OK" "$HERE/push_base.host.log"; then
  say "PUSH_BASE_CHAIN_GATE=BASE_RUN_NOT_OK"
  exit 10
fi
say "PUSH_BASE_CHAIN_STAGE=EXPORT (re-export frozen base)"
timeout 3600 $DOCKER /workspace/booster_ws/scripts/export_base_policy.sh >> "$LOG" 2>&1
if ! grep -q "BASE_EXPORT_DONE" "$LOG"; then
  say "PUSH_BASE_CHAIN_GATE=EXPORT_FAILED"
  exit 11
fi
say "PUSH_BASE_CHAIN_STAGE=DIAG (two-pass frozen A/B)"
timeout 3600 $DOCKER /workspace/booster_ws/scripts/diag_frozen.sh >> "$LOG" 2>&1
DR=$(grep -o 'done_rate=[0-9.]*' "$LOG" | tail -1 | cut -d= -f2)
say "PUSH_BASE_CHAIN_DIAG pass-B done_rate=${DR:-PARSE_FAIL}"
if [ -z "$DR" ]; then
  say "PUSH_BASE_CHAIN_GATE=DIAG_PARSE_FAILED"
  exit 12
fi
if ! awk -v v="$DR" 'BEGIN{exit !(v < 0.02)}'; then
  say "PUSH_BASE_CHAIN_GATE=DIAG_FAILED_BASE_STILL_FALLS (done_rate=$DR >= 0.02) — NOT launching reach; next: in-env A/B hunt"
  exit 13
fi
say "PUSH_BASE_CHAIN_STAGE=REACH_LAUNCH (gate passed)"
bash "$HERE/spark_push_host.sh" reach >> "$LOG" 2>&1
say "PUSH_BASE_CHAIN_DONE (reach training live in tmux k1_spark_push_reach)"
