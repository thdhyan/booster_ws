#!/usr/bin/env bash
# Laptop side: manage the on-robot local ASR/TTS. Stops the robot's audio service while ours runs
# and starts it again afterwards (robot/run.sh traps EXIT, fw 1.8.x only: containerd API on :20001, so it comes back even if this laptop drops).
#   k1voice.sh start   <robot> [listen] background voice chat: ASR -> OpenRouter (key from ./.env, sent over ssh, never written
#                                       on the robot) -> Kokoro. "listen" = transcripts only, no LLM. Log: ~/k1_chat/voice.log
#   k1voice.sh stop    <robot>          stop ours, start the audio service again
#   k1voice.sh smoke   <robot> [text]   say text, hear one utterance, say it back (foreground)
#   k1voice.sh calibrate <robot>        measure room noise + your voice, save the VAD threshold on the robot, show what ASR heard
#   k1voice.sh fetch   <robot>          copy the robot's raw audio dumps to audio_dumps/<robot>/ (calib_*, mictest_*, utt_* wav + txt)
#   k1voice.sh log     <robot>          follow the live transcripts of `start` (Ctrl-C leaves the listener running)
#   k1voice.sh status  <robot>
#   k1voice.sh restore <robot>          just make sure the audio service is running
set -euo pipefail
cmd=${1:?usage: k1voice.sh start|stop|smoke|log|fetch|mic-test|calibrate|status|restore <robot> [text]}
name=${2:?robot name, e.g. a1}
host=boosterk1.${name,,}
root=$(cd "$(dirname "$0")/.." && pwd)

sync() {
    ssh -o ConnectTimeout=5 "$host" 'mkdir -p ~/k1_chat/src/k1_chat'
    rsync -aq --delete --exclude __pycache__ "$root/src/k1_chat/" "$host:k1_chat/src/k1_chat/"
    rsync -aq "$root/robot/run.sh" "$host:k1_chat/run.sh"
    ssh "$host" 'chmod +x ~/k1_chat/run.sh'
}

sync
case $cmd in
  start)
    mode=chat; [ "${3:-}" = listen ] && mode=listen
    if [ $mode = chat ]; then
        set -a; . "$root/.env"; set +a
        [ -n "${OPENROUTER_API_KEY:-}" ] && [ -n "${K1_CHAT_MODEL:-}" ] || { echo ".env needs OPENROUTER_API_KEY and K1_CHAT_MODEL (or use: start $name listen)" >&2; exit 2; }
    fi
    # key reaches the robot over ssh stdin into a 0600 temp file that the launcher sources and deletes at once
    printf 'OPENROUTER_API_KEY=%s\nK1_CHAT_MODEL=%s\n' "${OPENROUTER_API_KEY:-}" "${K1_CHAT_MODEL:-}" |
      ssh "$host" "[ -f ~/k1_chat/run.pid ] && kill -0 \$(cat ~/k1_chat/run.pid) 2>/dev/null && { echo already running; exit 0; }
        umask 077; cat > ~/k1_chat/.chat_env
        setsid nohup bash -c 'cd ~/k1_chat && set -a && . ./.chat_env && set +a && rm -f ./.chat_env && exec ./run.sh $mode' > ~/k1_chat/voice.log 2>&1 < /dev/null &
        sleep 7; tail -3 ~/k1_chat/voice.log; ~/k1_chat/run.sh --status | grep -q stopped && echo 'audio service stopped, voice $mode running' || echo 'WARNING: audio service still running'"
    ;;
  stop)
    ssh "$host" 'if [ -f ~/k1_chat/run.pid ] && kill -0 $(cat ~/k1_chat/run.pid) 2>/dev/null; then kill -TERM $(cat ~/k1_chat/run.pid); for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do kill -0 $(cat ~/k1_chat/run.pid 2>/dev/null) 2>/dev/null || break; sleep 1; done; fi
      ~/k1_chat/run.sh --status; tail -2 ~/k1_chat/voice.log 2>/dev/null' ;;
  smoke)
    ssh -t "$host" "~/k1_chat/run.sh smoke ${3:+\"$3\"}" ;;
  log)
    ssh -t "$host" 'tail -n 20 -f ~/k1_chat/voice.log' ;;
  fetch)
    mkdir -p "$root/audio_dumps/${name,,}"
    rsync -av "$host:k1_chat/dump/" "$root/audio_dumps/${name,,}/" && ls -l "$root/audio_dumps/${name,,}" ;;
  mic-test)
    ssh -t "$host" "~/k1_chat/run.sh mic-test" ;;
  calibrate)
    ssh -t "$host" '~/k1_chat/run.sh calibrate' ;;
  status)
    ssh "$host" '[ -f ~/k1_chat/run.pid ] && kill -0 $(cat ~/k1_chat/run.pid) 2>/dev/null && echo "our ASR: running" || echo "our ASR: stopped"
      ~/k1_chat/run.sh --status; tail -3 ~/k1_chat/voice.log 2>/dev/null' ;;
  restore)
    ssh "$host" '~/k1_chat/run.sh --restore' ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
