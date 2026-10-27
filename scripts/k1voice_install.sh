#!/usr/bin/env bash
# Install the local voice stack (faster-whisper base.en + piper lessac) in ~/k1_chat on a K1. Idempotent.
# Needs internet on the robot (laptop NAT). Adds a non-persistent default route via the laptop if the
# robot has none (DHCP robots default to the router, which has no internet).
#   k1voice_install.sh <robot>...
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: k1voice_install.sh a1 a2 a3 b1 b2 b3" >&2; exit 2; }
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium

install_one() {
    local host=boosterk1.${1,,}
    ssh -o ConnectTimeout=6 "$host" 'bash -s' <<'REMOTE' 2>&1 | sed "s/^/[$1] /"
set -u
curl -sm6 -o /dev/null https://pypi.org || {
    sudo ip route add default via 192.168.50.250 dev wlP1p1s0 metric 100 2>/dev/null
    curl -sm6 -o /dev/null https://pypi.org || { echo "NO INTERNET (is the laptop NAT on?)"; exit 1; }
}
mkdir -p ~/k1_chat/models && cd ~/k1_chat
[ -x venv/bin/python ] || python3 -m venv venv || { echo "venv failed"; exit 1; }
venv/bin/pip install -q --upgrade pip && venv/bin/pip install -q faster-whisper piper-tts numpy || { echo "pip failed"; exit 1; }
[ -f models/en_US-lessac-medium.onnx ] || venv/bin/python -m piper.download_voices --data-dir models en_US-lessac-medium >/dev/null 2>&1
HF_HUB_DISABLE_PROGRESS_BARS=1 venv/bin/python -c 'from faster_whisper import WhisperModel; WhisperModel("base.en", device="cpu", compute_type="int8")' 2>&1 | grep -v Warning
ls models/en_US-lessac-medium.onnx >/dev/null && echo "INSTALL OK: $(grep Version /opt/booster/version.txt | tail -1)"
REMOTE
}

for r in "$@"; do install_one "$r" & done
wait
