#!/usr/bin/env bash
# Launch a guarded training run inside a DETACHED tmux session so training
# survives quitting OpenCode / closing terminals.
#
# usage: scripts/tmux_train.sh --name <n> -- <command...>
#   tmux session : k1_<n>   (attach: tmux attach -t k1_<n>, detach: Ctrl-b d)
#   guard log    : logs/guard_<n>.log
#   done marker  : logs/tmux_<n>.done  (contains GUARD_RC=<rc>; delete before relaunch)
#
# The pane prints GUARD_RC when finished and stays open so the output can be
# read after the fact.  Monitor:  tmux capture-pane -t k1_<n> -p | tail
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"

NAME="run"
if [ "${1:-}" = "--name" ]; then NAME="$2"; shift 2; fi
if [ "${1:-}" = "--" ]; then shift; fi
[ $# -ge 1 ] || { echo "usage: tmux_train.sh --name N -- <cmd...>" >&2; exit 2; }

SESSION="k1_${NAME}"
DONE="$WS/logs/tmux_${NAME}.done"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session $SESSION already exists (attach or kill it first)" >&2
    exit 3
fi
rm -f "$DONE"

# Make the tmux SERVER live in its OWN systemd scope: an OpenCode server
# restart kills its whole cgroup, taking any default-server tmux session with
# it (observed 2026-09-22 09:51: run died at iter ~1502). A scope-homed server
# survives restarts; later sessions join it over the existing socket.
if tmux list-sessions 2>&1 | grep -q "no server running"; then
    if systemd-run --user --scope tmux new-session -d -s k1_tmux_holder \
            'while :; do sleep 3600; done' 2>/dev/null; then
        echo "tmux server started inside a systemd scope (restart-proof)"
    else
        echo "WARN: systemd-run unavailable; tmux server NOT restart-proof" >&2
    fi
fi

# Built as a single shell string evaluated by tmux's default shell.
# HEADLESS=1: the unified train backend ignores a --headless CLI arg (verified
# 2026-09-22: window opened despite argv) — AppLauncher honors the HEADLESS
# env var instead. Override with HEADLESS=0 if you ever want the window.
INNER="cd $WS && export HEADLESS=\"\${HEADLESS:-1}\" && source scripts/phase6_env.sh && scripts/train_guard.sh --name $NAME -- $*; rc=\$?; echo GUARD_RC=\$rc | tee -a $WS/logs/guard_$NAME.log; echo GUARD_RC=\$rc > $DONE"

tmux new-session -d -s "$SESSION" "$INNER" || exit 4
echo "launched tmux session: $SESSION"
echo "  attach : tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  log    : logs/guard_${NAME}.log"
echo "  done   : $DONE (written when the run finishes)"
