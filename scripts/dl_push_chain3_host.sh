#!/bin/bash
# Auto-chain reach -> push (host side, dl; detached):
#   1. wait for tmux k1_spark_push_reach to end
#   2. gate: PUSH_FULL_MARKER=OK in reach.push.log (1500 iters actually ran)
#   3. gate: a reach checkpoint exists (newest logs/rsl_rl/p6_push_reach model_*.pt)
#   4. launch push stage 2 (dl_push_host.sh push, GPU 1, warm-starts from
#      that reach ckpt inside spark_push_container.sh; own smoke gate first)
# Any gate failure STOPS the chain with a grep-able marker (never rc alone).
# Log: scripts/dl_push_chain3.log
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/dl_push_chain3.log"
: > "$LOG"
say() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "DL_PUSH_CHAIN3_WAIT=k1_spark_push_reach"
while tmux has-session -t k1_spark_push_reach 2>/dev/null; do sleep 60; done
say "DL_PUSH_CHAIN3 reach session ended"

if ! grep -q "PUSH_FULL_MARKER=OK" "$HERE/reach.push.log"; then
  say "DL_PUSH_CHAIN3_GATE=REACH_FULL_NOT_OK (last marker: $(grep -o 'PUSH_FULL_MARKER=[A-Z]*' "$HERE/reach.push.log" | tail -1))"
  exit 10
fi
WARM=$(ls -t "$REPO"/logs/rsl_rl/p6_push_reach/*/model_*.pt 2>/dev/null | head -1)
if [ -z "$WARM" ]; then
  say "DL_PUSH_CHAIN3_GATE=NO_REACH_CHECKPOINT"
  exit 11
fi
say "DL_PUSH_CHAIN3_GATE=REACH_OK warm=$(basename "$WARM") ($(dirname "$WARM" | xargs basename))"

say "DL_PUSH_CHAIN3_STAGE=PUSH_LAUNCH"
DL_GPUS=device=1 bash "$HERE/dl_push_host.sh" push >> "$LOG" 2>&1
if ! tmux has-session -t k1_spark_push_push 2>/dev/null; then
  say "DL_PUSH_CHAIN3_GATE=PUSH_SESSION_MISSING"
  exit 12
fi
say "DL_PUSH_CHAIN3_DONE (push training live in tmux k1_spark_push_push)"
