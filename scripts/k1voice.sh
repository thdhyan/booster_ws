#!/usr/bin/env bash
# Laptop side: manage the on-robot local ASR/TTS. Stops the robot's booster-audio while ours runs
# and restarts it afterwards (robot/run.sh traps EXIT, so it comes back even if this laptop drops).
#   k1voice.sh start   <robot>          background listener, transcripts in ~/k1_chat/voice.log
#   k1voice.sh stop    <robot>          stop ours, restart booster-audio
#   k1voice.sh smoke   <robot> [text]   say text, hear one utterance, say it back (foreground)
#   k1voice.sh status  <robot>
#   k1voice.sh restore <robot>          just make sure booster-audio is running
set -euo pipefail
cmd=${1:?usage: k1voice.sh start|stop|smoke|status|restore <robot> [text]}
name=${2:?robot name, e.g. a1}
host=boosterk1.${name,,}
root=$(cd "$(dirname "$0")/.." && pwd)

sync() {
    ssh -o ConnectTimeout=5 "$host" 'mkdir -p ~/k1_chat/src/k1_chat'
    rsync -aq --delete --exclude __pycache__ "$root/src/k1_chat/" "$host:k1_chat/src/k1_chat/"
    rsync -aq "$root/robot/run.sh" "$host:k1_chat/run.sh"
    ssh "$host" 'chmod +x ~/k1_chat/run.sh'
}

case $cmd in
  start)
    sync
    ssh "$host" '[ -f ~/k1_chat/run.pid ] && kill -0 $(cat ~/k1_chat/run.pid) 2>/dev/null && { echo "already running"; exit 0; }
      setsid nohup bash -c "cd ~/k1_chat && exec ./run.sh listen" > ~/k1_chat/voice.log 2>&1 < /dev/null &
      sleep 6; tail -3 ~/k1_chat/voice.log; pgrep -x booster-audio >/dev/null && echo "WARNING: booster-audio still running" || echo "booster-audio stopped, ASR listening"' ;;
  stop)
    ssh "$host" 'if [ -f ~/k1_chat/run.pid ] && kill -0 $(cat ~/k1_chat/run.pid) 2>/dev/null; then kill -TERM $(cat ~/k1_chat/run.pid); for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do kill -0 $(cat ~/k1_chat/run.pid 2>/dev/null) 2>/dev/null || break; sleep 1; done; fi
      pgrep -x booster-audio >/dev/null && echo "booster-audio running (pid $(pgrep -x booster-audio))" || echo "WARNING: booster-audio NOT running"; tail -2 ~/k1_chat/voice.log 2>/dev/null' ;;
  smoke)
    sync
    ssh -t "$host" "~/k1_chat/run.sh smoke ${3:+\"$3\"}" ;;
  status)
    ssh "$host" '[ -f ~/k1_chat/run.pid ] && kill -0 $(cat ~/k1_chat/run.pid) 2>/dev/null && echo "our ASR: running" || echo "our ASR: stopped"
      pgrep -x booster-audio >/dev/null && echo "booster-audio: running" || echo "booster-audio: stopped"; tail -3 ~/k1_chat/voice.log 2>/dev/null' ;;
  restore)
    sync
    ssh "$host" 'pgrep -x booster-audio >/dev/null && echo "booster-audio already running" || { setsid nohup bash -lc "cd /opt/booster/BoosterAudio/bin && nohup /opt/booster/BoosterAudio/bin/booster-audio >> /opt/booster/BoosterAudio/log/booster-audio.stdout.log 2>&1 &" >/dev/null 2>&1; sleep 4; pgrep -x booster-audio >/dev/null && echo restored || echo "WARNING: still down"; }' ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
