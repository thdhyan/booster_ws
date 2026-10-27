#!/usr/bin/env bash
# Runs on the K1. Frees the mic by stopping the robot's booster-audio, runs k1_chat.robot_audio <args>,
# and ALWAYS starts booster-audio again on exit (normal end, SIGTERM, crash, ssh drop).
# usage: run.sh smoke "sentence" | run.sh listen
BA=/opt/booster/BoosterAudio
child=""
PIDFILE="$HOME/k1_chat/run.pid"

restore() {
    [ -n "$child" ] && kill "$child" 2>/dev/null && wait "$child" 2>/dev/null
    if ! pgrep -x booster-audio >/dev/null; then
        setsid nohup bash -lc "cd $BA/bin && nohup $BA/bin/booster-audio >> $BA/log/booster-audio.stdout.log 2>&1 &" >/dev/null 2>&1
        for _ in 1 2 3 4 5; do sleep 1; pgrep -x booster-audio >/dev/null && break; done
    fi
    pgrep -x booster-audio >/dev/null && echo "booster-audio restored (pid $(pgrep -x booster-audio))" || echo "WARNING: booster-audio NOT running" >&2
}
echo $$ > "$PIDFILE"
trap 'restore; rm -f "$PIDFILE"' EXIT
trap 'exit 143' TERM INT HUP

pkill -TERM -x booster-audio
for _ in 1 2 3 4 5 6 7 8; do pgrep -x booster-audio >/dev/null || break; sleep 1; done
pgrep -x booster-audio >/dev/null && { echo "booster-audio would not stop" >&2; exit 1; }

cd "$HOME/k1_chat" || exit 1
HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python -m k1_chat.robot_audio "$@" &
child=$!
wait "$child"
