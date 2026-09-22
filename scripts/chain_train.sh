#!/usr/bin/env bash
# chain_train.sh — run inside tmux: wait for session $1 to end, then execute
# the remaining arguments as a command string via bash -c (so $(...) is
# evaluated AFTER the wait — e.g. picking the newest checkpoint).
#
# Usage: tmux new-session -d -s <chain> 'chain_train.sh <session> "<cmd>"'
# All waiting happens on the server in tmux — no client-side polling.
set -u
WAIT="${1:-}"; shift || true
[ -n "$WAIT" ] && [ $# -ge 1 ] || { echo "usage: chain_train.sh <session> <cmd...>" >&2; exit 2; }
while tmux has-session -t "$WAIT" 2>/dev/null; do sleep 15; done
exec bash -c "$*"
