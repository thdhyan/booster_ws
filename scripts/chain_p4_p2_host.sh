#!/bin/bash
# Wait for the P4 kick-teacher full run to pass its gate, then launch the P2
# gait teacher->student campaign.  The dedicated final recorder already waits
# on P2_GAIT_STUDENT_MARKER, so nothing else needs to be started here.
#
#   scripts/chain_p4_p2_host.sh
#
# Markers land in scripts/p4_p2_chain.log.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/p4_p2_chain.log"
P4_LOG="$HERE/p4t.soccer.log"
: > "$LOG"

# P3 is already recovered and verified; assert it before moving on so a later
# stage never runs on top of a missing upstream checkpoint.
P3_CKPT="$REPO/logs/rsl_rl/p3_head_track/2026-09-28_00-42-16_p3_head_track/model_1999.pt"
if [ ! -s "$P3_CKPT" ]; then
  echo "CHAIN_P3_CKPT_MISSING=$P3_CKPT" | tee -a "$LOG"
  exit 30
fi
echo "CHAIN_P3_CKPT_OK=$P3_CKPT" | tee -a "$LOG"

echo "CHAIN_WAIT_P4=START" | tee -a "$LOG"
until grep -qE 'SOC_FULL_MARKER=(OK|FAIL)' "$P4_LOG" 2>/dev/null; do sleep 60; done
if ! grep -q 'SOC_FULL_MARKER=OK' "$P4_LOG"; then
  echo "CHAIN_P4_MARKER=FAIL" | tee -a "$LOG"
  exit 31
fi
echo "CHAIN_P4_MARKER=OK" | tee -a "$LOG"

# The stale P2 queue from 24 Sep would double-launch once the marker appears.
tmux kill-session -t k1_spark_p2_gait 2>/dev/null || true

echo "CHAIN_P2_LAUNCH=START" | tee -a "$LOG"
WAIT_FOR_P3=0 bash "$HERE/spark_p2_gait_host.sh" 2>&1 | tee -a "$LOG"
echo "CHAIN_P2_LAUNCH=DONE" | tee -a "$LOG"

# Mirror the student marker into the chain log for a single place to check.
until grep -qE 'P2_GAIT_STUDENT_MARKER=(OK|FAIL)' "$HERE/p2_gait.log" 2>/dev/null; do sleep 60; done
if grep -q 'P2_GAIT_STUDENT_MARKER=OK' "$HERE/p2_gait.log"; then
  echo "CHAIN_P2_STUDENT=OK" | tee -a "$LOG"
else
  echo "CHAIN_P2_STUDENT=FAIL" | tee -a "$LOG"
  exit 32
fi
