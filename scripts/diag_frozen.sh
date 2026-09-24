#!/bin/bash
# Frozen-base diagnostic (inside the Isaac image) — two passes:
#   A: PUSH_FROZEN_MODE=hold  -> env/physics only (default leg targets, no policy)
#   B: PUSH_FROZEN_MODE=debug -> frozen policy active + obs-block/out dump
# Both: zero actions, 8 envs x 400 steps. Grep EVAL_PUSH_RESULT / [frozenobs].
set +e
REPO=/workspace/booster_ws
cd "$REPO" || exit 1
echo "=== DIAG A: HOLD (env only, policy bypassed)"
PUSH_FROZEN_MODE=hold PUSH_FROZEN_DEBUG=1 \
  "$REPO/scripts/eval_push.sh" Isaac-Push-Reach-K1-v0 - 8 400 --zero_actions
echo "=== DIAG B: POLICY ACTIVE (cmd=0) + obs dump"
PUSH_FROZEN_MODE=debug PUSH_FROZEN_DEBUG=1 \
  "$REPO/scripts/eval_push.sh" Isaac-Push-Reach-K1-v0 - 8 400 --zero_actions
echo "DIAG_FROZEN_ALL_DONE"
